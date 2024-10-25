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
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2

# Function to adjust ROI points based on provided coordinates
def set_roi_based_on_points(points, coordinates):
    x_offset, y_offset = coordinates["x"], coordinates["y"]
    return [(int(x + x_offset), int(y + y_offset)) for x, y in points]


def capture_and_publish(frame, c_id, s_id, typ, count):
    try:
        image_path = capture_image(frame)  # Assuming this function saves the image and returns the path
        message = {
            "cameraId": c_id,
            "siteId": s_id,
            "type": typ,
            "image": image_path,
            "people_count": count
        }
        pub("people/detection", message)
        logger.info(f"Published people message for camera {c_id}.")
    except Exception as e:
        logger.error(f"Error capturing image or publishing MQTT for camera {c_id}: {str(e)}")


def people_count(camera_id, s_id, typ, coordinates, width, height, stop_event):
    try:
        model = YOLO("Model/yolov8l.pt")

        if coordinates and "points" in coordinates and coordinates["points"]:
            roi_points = np.array(set_roi_based_on_points(coordinates["points"], coordinates), dtype=np.int32)
            roi_mask = np.zeros((height, width), dtype=np.uint8)
            cv2.fillPoly(roi_mask, [roi_points], 255)  # Fill mask for the static ROI
            logger.info(f"ROI set for people detection on camera {camera_id}")
        else:
            roi_mask = None

        previous_people_count = 0  # To track the previous count

        while not stop_event.is_set():
            start_time = time.time()
            frame = queues_dict[f"{camera_id}_{typ}"].get(timeout=10)  # Handle timeouts if frame retrieval takes too long
            if frame is None:
                continue

            # # Log the queue size
            # queue_size = queues_dict[f"{camera_id}_{typ}"].qsize()
            # logger.info(f"people: {queue_size}")

            masked_frame = cv2.bitwise_and(frame, frame, mask=roi_mask) if roi_mask is not None else frame

            # Run YOLOv8 inference on the masked frame
            results = model(masked_frame, conf=0.3, iou=0.4, verbose=False)

            # Initialize people count
            count = 0

            # Iterate through detected objects
            for box in results[0].boxes.data:
                class_id = int(box[5])
                if class_id == 0:  # Class ID 0 corresponds to 'person' in COCO
                    count += 1

            # Publish the count to MQTT and capture the frame only if the count has changed
            if previous_people_count != count:
                executor.submit(capture_and_publish, frame, camera_id, s_id, typ, count)
                previous_people_count = count  # Update the previous count

            # logger.info(f"People_count {(time.time() - start_time) * 1000:.2f} milliseconds.")

    except Exception as e:
        logger.error(f"Error During People Count:{str(e)}")
        return PCError(f"People Count Failed for camera : {camera_id}")



def start_pc(c_id, s_id, typ, co, width, height, rtsp):
    """
    Start the motion detection process in a separate thread for the given camera task.
    """
    try:
        if f"{c_id}_{typ}_detect" in global_thread:
            stop_pc(c_id, typ)
            time.sleep(2)

        executor.submit(start_feature_processing, c_id, typ, rtsp, width, height)
        stop_event = threading.Event()  # Create a stop event for each feature
        global_thread[f"{c_id}_{typ}_detect"] = stop_event
        executor.submit(people_count, c_id, s_id, typ, co, width, height, stop_event)

        logger.info(f"Started people detection for camera {c_id}.")

    except Exception as e:
        logger.error(f"Failed to start detection process for camera {c_id}: {str(e)}", exc_info=True)
        return False
    return True

def stop_pc(camera_id, typ):
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