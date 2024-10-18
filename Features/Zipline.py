# api
import queue
import threading
import time
import cv2
import numpy as np
from ultralytics import YOLO
from app.config import logger, global_thread, queues_dict, get_executor
from app.mqtt_handler import publish_message_mqtt as pub
from Features.sort import Sort
from app.utils import capture_image, start_feature_processing
from concurrent.futures import ThreadPoolExecutor
from collections import deque

executor = get_executor()


# Initialize SORT tracker
tracker = Sort(max_age=20, min_hits=4, iou_threshold=0.4)

class TrackableObject:
    def __init__(self, object_id, centroid):
        self.object_id = object_id
        self.centroids = deque([centroid], maxlen=50)
        self.counted = False

def set_roi_based_on_points(points, coordinates):
    x_offset, y_offset = coordinates["x"], coordinates["y"]
    return [(int(point[0] + x_offset), int(point[1] + y_offset)) for point in points]

def capture_and_publish(frame, c_id, s_id, typ, count):
    try:
        image_path = capture_image(frame)
        message = {
            "cameraId": c_id,
            "siteId": s_id,
            "type": typ,
            "image": image_path,
            "zipline_count": count
        }
        pub("zipline/detection", message)
        logger.info(f"Published zipline detection for camera {c_id}.")
    except Exception as e:
        logger.error(f"Error capturing image or publishing MQTT for camera {c_id}: {str(e)}")

def is_inside_roi(point, roi_points):
    return cv2.pointPolygonTest(np.array(roi_points, np.int32), point, False) >= 0

def is_crossing_line(p1, p2, line_start, line_end):
    def ccw(A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
    return ccw(p1, line_start, line_end) != ccw(p2, line_start, line_end) and \
           ccw(p1, p2, line_start) != ccw(p1, p2, line_end)

def is_movement_in_arrow_direction(prev_point, current_point, arrow_start, arrow_end):
    movement_vector = np.array(current_point) - np.array(prev_point)
    arrow_vector = np.array(arrow_end) - np.array(arrow_start)
    return np.dot(movement_vector, arrow_vector) > 0

def detect_zipline(camera_id, s_id, typ, coordinates, width, height, stop_event):
    try:
        count = 0
        frame_skip = 2
        frame_count = 0
        trackable_objects = {}
        last_count_time = {}
        debounce_time = 1
        model = YOLO('Model/yolov8n.pt')

        # roi_mask = np.zeros((height, width), dtype=np.uint8)

        if coordinates and "line" in coordinates and coordinates["line"]:

            roi_points = np.array(set_roi_based_on_points(coordinates["roi"]["points"], coordinates["roi"]),
                                  dtype=np.int32)
            roi_mask = np.zeros((height, width), dtype=np.uint8)
            cv2.fillPoly(roi_mask, [roi_points], 255)
            line_start, line_end = [tuple(p) for p in
                                    set_roi_based_on_points(coordinates["line"]["points"], coordinates["line"])]
            arrow_start, arrow_end = [tuple(p) for p in
                                      set_roi_based_on_points(coordinates["arrow"]["points"], coordinates["arrow"])]

        else:
            roi_mask = None

        while not stop_event.is_set():
            # start_time = time.time()
            frame = queues_dict[f"{camera_id}_{typ}"].get(timeout=10)
            if frame is None:
                logger.warning(f"Received None frame for camera {camera_id}")
                continue
            # # Log the queue size
            # queue_size = queues_dict[f"{camera_id}_{typ}"].qsize()
            # logger.info(f"zipline---: {queue_size}")

            frame_count += 1
            if frame_count % frame_skip != 0:
                continue

            masked_frame = cv2.bitwise_and(frame, frame, mask=roi_mask)
            results = model(masked_frame, stream=True, verbose=False)

            detections = np.empty((0, 5))
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    if cls == 0 and conf > 0.3:
                        detections = np.vstack((detections, [x1, y1, x2, y2, conf]))

            tracked_objects = tracker.update(detections)

            for obj in tracked_objects:
                x1, y1, x2, y2, object_id = map(int, obj)
                centroid = (int((x1 + x2) / 2), int((y1 + y2) / 2))

                if object_id not in trackable_objects:
                    to = TrackableObject(object_id, centroid)
                    trackable_objects[object_id] = to
                else:
                    to = trackable_objects[object_id]

                to.centroids.append(centroid)

                if is_inside_roi(centroid, roi_points):
                    if len(to.centroids) >= 2:
                        prev_centroid = to.centroids[-2]
                        current_centroid = to.centroids[-1]

                        if is_crossing_line(prev_centroid, current_centroid, line_start, line_end):
                            current_time = time.time()
                            if current_time - last_count_time.get(object_id, 0) > debounce_time:
                                if is_movement_in_arrow_direction(prev_centroid, current_centroid, arrow_start,
                                                                  arrow_end):
                                    count += 1
                                    last_count_time[object_id] = current_time
                                    executor.submit(capture_and_publish, frame, camera_id, s_id, typ, count)

            queues_dict[f"{camera_id}_{typ}"].task_done()
            # frame_end_time = time.time()
            # frame_processing_time_ms = (frame_end_time - start_time) * 1000
            # logger.info(f"zipline----- {frame_processing_time_ms:.2f} milliseconds.")

    except Exception as e:
        logger.error(f"Error during zipline detection: {str(e)}")



def zipline_start(c_id, s_id, typ, co, width, height, rtsp):
    """
    Start the motion detection process in a separate thread for the given camera task.
    """
    try:
        if f"{c_id}_{typ}_detect" in global_thread:
            zipline_stop(c_id, typ)
            time.sleep(2)
        executor.submit(start_feature_processing, c_id, typ, rtsp, width, height)
        stop_event = threading.Event()  # Create a stop event for each feature
        global_thread[f"{c_id}_{typ}_detect"] = stop_event
        executor.submit(detect_zipline, c_id, s_id, typ, co, width, height, stop_event)

        logger.info(f"Started motion detection for camera {c_id}.")

    except Exception as e:
        logger.error(f"Failed to start detection process for camera {c_id}: {str(e)}", exc_info=True)
        return False
    return True

def zipline_stop(camera_id, typ):
    stopped_tasks = []
    not_found_tasks = []

    key = f"{camera_id}_{typ}"  # Construct the key as used in the dictionary
    key2 = f"{camera_id}_{typ}_detect"

    try:
        if key in global_thread and key in queues_dict and key2 in global_thread:
            stop_event_detect = global_thread[key]  # Retrieve the stop event from the dictionary
            stop_event_detect.set()  # Signal the thread to stop
            del global_thread[key]  # Delete the entry from the dictionary after setting the stop event
            stop_event = global_thread[key2]  # Retrieve the stop event from the dictionary
            stop_event.set()  # Signal the thread to stop
            del global_thread[key2]  # Delete the entry from the dictionary after setting the stop event
            stopped_tasks.append(camera_id)
            logger.info(f"Stopped {typ} and removed key for camera {camera_id} of type {typ}.")
        else:
            not_found_tasks.append(camera_id)
            logger.warning(f"No active detection found for {camera_id} of type {typ}.")
    except Exception as e:
        logger.error(f"Error during stopping detection for {camera_id}: {str(e)}", exc_info=True)

    return {
        "success": len(stopped_tasks) > 0,
        "stopped": stopped_tasks,
        "not_found": not_found_tasks
    }