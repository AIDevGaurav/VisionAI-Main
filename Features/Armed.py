import queue
import threading
import time
import cv2
import numpy as np
from ultralytics import YOLO
from app.exceptions import PCError
from app.config import logger, global_thread, queues_dict, executor
from app.mqtt_handler import publish_message_mqtt as pub
from app.utils import capture_image, start_feature_processing
from app.exceptions import ArmError


# Function to adjust ROI points based on provided coordinates
def set_roi_based_on_points(points, coordinates):
    x_offset, y_offset = coordinates["x"], coordinates["y"]
    return [(int(x + x_offset), int(y + y_offset)) for x, y in points]

def capture_and_publish(frame, c_id, s_id, typ):
    try:
        image_path = capture_image(frame)  # Assuming this function saves the image and returns the path
        message = {
            "cameraId": c_id,
            "siteId": s_id,
            "type": typ,
            "image": image_path,
        }
        pub("arm/detection", message)
        logger.info(f"Published arm message for camera {c_id}.")
    except Exception as e:
        logger.error(f"Error capturing image or publishing MQTT for camera {c_id}: {str(e)}")


def detect_armed_person(camera_id, s_id, typ, coordinates, width, height, stop_event):
    """
    :tasks: Detect armed persons in the stream using YOLOv8.
    :return: Capture Image, Video and Publish Mqtt message
    """
    try:
        model = YOLO('Model/armed.pt')
        last_detection_time = 0

        if coordinates and "points" in coordinates and coordinates["points"]:
            roi_points = np.array(set_roi_based_on_points(coordinates["points"], coordinates), dtype=np.int32)
            roi_mask = np.zeros((height, width), dtype=np.uint8)
            cv2.fillPoly(roi_mask, [roi_points], 255)  # Fill mask for the static ROI
        else:
            roi_mask = None

        while not stop_event.is_set():
            # start_time = time.time()
            frame = queues_dict[f"{camera_id}_{typ}"].get(timeout=10)  # Handle timeouts if frame retrieval takes too long
            if frame is None:
                continue

            # # Log the queue size
            # queue_size = queues_dict[f"{camera_id}_{typ}"].qsize()
            # logger.info(f"zipline---: {queue_size}")

            masked_frame = cv2.bitwise_and(frame, frame, mask=roi_mask) if roi_mask is not None else frame

            # Run YOLOv8 inference on the masked frame
            results = model(masked_frame, stream=True, verbose=False, classes = [0, 1])

            for result in results:
                if hasattr(result, 'boxes') and result.boxes.data.shape[0] > 0:
                    if time.time() - last_detection_time > 10:
                        executor.submit(capture_and_publish, frame, camera_id, s_id, typ)
                        last_detection_time = time.time()

            queues_dict[f"{camera_id}_{typ}"].task_done()
            # logger.info(f"people----- {(time.time() - start_time) * 1000:.2f} milliseconds.")

    except Exception as e:
        logger.error(f"Error During Armed detection:{str(e)}")
        return PCError(f"Armed Detection Failed for camera : {camera_id}")


def armed_start(c_id, s_id, typ, co, width, height, rtsp):
    """
    Start the motion detection process in a separate thread for the given camera task.
    """
    try:
        if f"{c_id}_{typ}_detect" in global_thread:
            armed_stop(c_id, typ)
            time.sleep(2)
        executor.submit(start_feature_processing, c_id, typ, rtsp, width, height)
        stop_event = threading.Event()  # Create a stop event for each feature
        global_thread[f"{c_id}_{typ}_detect"] = stop_event
        executor.submit(detect_armed_person, c_id, s_id, typ, co, width, height, stop_event)

        logger.info(f"Started Armed detection for camera {c_id}.")

    except Exception as e:
        logger.error(f"Failed to start Armed detection process for camera {c_id}: {str(e)}", exc_info=True)
        return False
    return True

def armed_stop(camera_id, typ):
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