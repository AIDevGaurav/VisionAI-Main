import queue
import threading
import cv2
import time
import numpy as np
from app.utils import capture_image, start_feature_processing
from app.mqtt_handler import publish_message_mqtt as pub
from app.config import logger, executor, global_thread, queues_dict
from app.exceptions import MotionDetectionError


def set_roi_based_on_points(points, coordinates):
    """
    Scale and set ROI based on given points and coordinates.
    """
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
            "timestamp": time.time()
        }
        pub("motion/detection", message)
        logger.info(f"Published motion message for camera {c_id}.")
    except Exception as e:
        logger.error(f"Error capturing image or publishing MQTT for camera {c_id}: {str(e)}")


def detect_motion(c_id, s_id, typ, co, width, height, stop_event):
    """
    Motion detection loop with static ROI setup.
    """
    logger.info(f"Motion detection started for camera {c_id}.")
    prev_frame_gray = None
    last_detection_time = 0

    try:
        # Initial check if coordinates are provided and contain necessary data
        if co and "points" in co and co["points"]:  # Check if 'points' key exists and is not empty
            roi_points = np.array(set_roi_based_on_points(co["points"], co), dtype=np.int32)
            roi_mask = np.zeros((height, width), dtype=np.uint8)
            cv2.fillPoly(roi_mask, [roi_points], 255)  # Fill mask for the static ROI
            # logger.info(f"ROI set for motion detection on camera {c_id}")
        else:
            roi_mask = None
            # logger.info(f"No ROI set, full frame will be processed for camera {c_id}")

        while not stop_event.is_set():
            # start_time = time.time()
            frame = queues_dict[f"{c_id}_{typ}"].get(timeout=10)  # Handle timeouts if frame retrieval takes too long
            if frame is None:
                continue

            # # Log the queue size
            # queue_size = queues_dict[f"{c_id}_{typ}"].qsize()
            # logger.info(f"motion---: {queue_size}")

            masked_frame = cv2.bitwise_and(frame, frame, mask=roi_mask) if roi_mask is not None else frame

            gray_frame = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY)
            gray_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0)

            if prev_frame_gray is None:
                prev_frame_gray = gray_frame
                continue

            frame_diff = cv2.absdiff(prev_frame_gray, gray_frame)
            _, thresh_frame = cv2.threshold(frame_diff, 16, 255, cv2.THRESH_BINARY)
            thresh_frame = cv2.dilate(thresh_frame, None, iterations=2)

            contours, _ = cv2.findContours(thresh_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                if cv2.contourArea(contour) > 1000:
                    if (time.time() - last_detection_time > 10):
                        logger.info(f"Motion detected for camera {c_id}.")
                        executor.submit(capture_and_publish, frame, c_id, s_id, typ)
                        last_detection_time = time.time()

            prev_frame_gray = gray_frame
            queues_dict[f"{c_id}_{typ}"].task_done()
            # logger.info(f"zipline----- {(time.time() - start_time) * 1000:.2f} milliseconds.")

    except Exception as e:
        logger.error(f"Error during motion detection for camera {c_id}: {str(e)}", exc_info=True)
        raise MotionDetectionError(f"Motion detection failed for camera {c_id}: {str(e)}")


def motion_start(c_id, s_id, typ, co, width, height, rtsp):
    """
    Start the motion detection process in a separate thread for the given camera task.
    """
    try:
        if f"{c_id}_{typ}_detect" in global_thread:
            motion_stop(c_id, typ)
            time.sleep(2)

        executor.submit(start_feature_processing, c_id, typ, rtsp, width, height)
        stop_event = threading.Event()  # Create a stop event for each feature
        global_thread[f"{c_id}_{typ}_detect"] = stop_event
        executor.submit(detect_motion, c_id, s_id, typ, co, width, height, stop_event)

        logger.info(f"Started motion detection for camera {c_id}.")

    except Exception as e:
        logger.error(f"Failed to start detection process for camera {c_id}: {str(e)}", exc_info=True)
        return False
    return True

def motion_stop(camera_id, typ):
    stopped_tasks = []
    not_found_tasks = []

    key = f"{camera_id}_{typ}"  # Construct the key as used in the dictionary

    key2 = f"{camera_id}_{typ}_detect"

    try:
        if key in queues_dict and key in global_thread and key2 in global_thread:
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
