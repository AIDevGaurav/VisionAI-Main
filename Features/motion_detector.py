import queue
import threading
import cv2
import time
import numpy as np
from app.utils import capture_image
from app.mqtt_handler import publish_message_mqtt as pub
from app.config import logger, get_executor, global_thread, queues_dict
from app.exceptions import MotionDetectionError

executor = get_executor()

def set_roi_based_on_points(points, coordinates):
    """
    Scale and set ROI based on given points and coordinates.
    """
    x_offset = coordinates["x"]
    y_offset = coordinates["y"]

    scaled_points = []
    for point in points:
        scaled_x = int(point[0] + x_offset)
        scaled_y = int(point[1] + y_offset)
        scaled_points.append((scaled_x, scaled_y))
    return scaled_points

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

            if roi_mask is not None:
                masked_frame = cv2.bitwise_and(frame, frame, mask=roi_mask)
                cv2.polylines(frame, [roi_points], isClosed=True, color=(255, 0, 0), thickness=2)
            else:
                masked_frame = frame

            gray_frame = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY)
            gray_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0)

            if prev_frame_gray is None:
                prev_frame_gray = gray_frame
                continue

            frame_diff = cv2.absdiff(prev_frame_gray, gray_frame)
            _, thresh_frame = cv2.threshold(frame_diff, 16, 255, cv2.THRESH_BINARY)
            thresh_frame = cv2.dilate(thresh_frame, None, iterations=2)

            contours, _ = cv2.findContours(thresh_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            motion_detected = False

            for contour in contours:
                if cv2.contourArea(contour) > 1000:
                    (x, y, w, h) = cv2.boundingRect(contour)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    motion_detected = True

            if motion_detected and (time.time() - last_detection_time > 10):
                # logger.info(f"Motion detected for camera {c_id}.")
                executor.submit(capture_and_publish, frame, c_id, s_id, typ)
                last_detection_time = time.time()

            cv2.imshow(f"Motion Detection - Camera {c_id}", frame)
            prev_frame_gray = gray_frame
            queues_dict[f"{c_id}_{typ}"].task_done()

            # frame_end_time = time.time()
            # frame_processing_time_ms = (frame_end_time - start_time) * 1000
            # logger.info(f"motion----- {frame_processing_time_ms:.2f} milliseconds.")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        logger.error(f"Error during motion detection for camera {c_id}: {str(e)}", exc_info=True)
        raise MotionDetectionError(f"Motion detection failed for camera {c_id}: {str(e)}")
    finally:
        cv2.destroyWindow(f"Motion Detection - Camera {c_id}")


def motion_start(c_id, s_id, typ, co, width, height):
    """
    Start the motion detection process in a separate thread for the given camera task.
    """
    try:
        stop_event = threading.Event()  # Create a stop event for each feature
        global_thread[f"{c_id}_{typ}"] = stop_event
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

    try:
        if key in global_thread and key in queues_dict:
            stop_event = global_thread[key]  # Retrieve the stop event from the dictionary
            stop_event.set()  # Signal the thread to stop
            del global_thread[key]  # Delete the entry from the dictionary after setting the stop event
            queues_dict[key] = queue.ShutDown
            stopped_tasks.append(camera_id)
            logger.info(f"Stopped motion detection and removed key for camera {camera_id} of type {typ}.")
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
