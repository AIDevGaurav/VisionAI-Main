import queue
import threading
import numpy as np
from ultralytics import YOLO
from app.exceptions import FallError
from app.utils import capture_image, \
    start_feature_processing  # Assuming capture_image and capture_video are defined in utils
from app.mqtt_handler import publish_message_mqtt as pub  # Assuming you have an MQTT handler setup
from app.config import logger, global_thread, executor, queues_dict
import cv2
import time


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
        pub("fall/detection", message)
        logger.info(f"Published fall message for camera {c_id}.")
    except Exception as e:
        logger.error(f"Error capturing image or publishing MQTT for camera {c_id}: {str(e)}")


fall_detected_time = None  # To track when to remove the text

# Function to extract keypoints for fall detection
def extract_keypoints(data, conf):
    keypoints = {
        'left_ankle': data[15][:2] if conf[15] > 0.5 else None,
        'right_ankle': data[16][:2] if conf[16] > 0.5 else None,
        'left_shoulder': data[5][:2] if conf[5] > 0.5 else None,
        'right_shoulder': data[2][:2] if conf[2] > 0.5 else None,
        'left_hip': data[11][:2] if conf[11] > 0.5 else None,
        'right_hip': data[8][:2] if conf[8] > 0.5 else None,
    }
    return keypoints

# Example of handling missing keypoints and running fall detection logic
def process_frame(data, conf):
    global fall_detected_time

    keypoints = extract_keypoints(data, conf)

    # Check if all the necessary keypoints are available
    if (keypoints['left_shoulder'] is not None and
            keypoints['right_shoulder'] is not None and
            keypoints['left_hip'] is not None and
            keypoints['right_hip'] is not None and
            keypoints['left_ankle'] is not None and
            keypoints['right_ankle'] is not None):

        # Get the y-coordinate of the ankles as the reference
        ankle_y = (keypoints['left_ankle'][1] + keypoints['right_ankle'][1]) / 2

        # Calculate the average height of shoulders and hips
        shoulder_height = (keypoints['left_shoulder'][1] + keypoints['right_shoulder'][1]) / 2
        hip_height = (keypoints['left_hip'][1] + keypoints['right_hip'][1]) / 2

        # Check for sudden changes in position
        if (abs(ankle_y - shoulder_height) < 80) and \
           (abs(ankle_y - hip_height) < 80):
            fall_detected_time = time.time()  # Start timer for displaying fall detection text

def fall_detect(camera_id, s_id, typ, coordinates, width, height, stop_event):
    try:
        model = YOLO('Model/yolov8l-pose.pt')

        if coordinates and "points" in coordinates and coordinates["points"]:
            roi_points = np.array(set_roi_based_on_points(coordinates["points"], coordinates), dtype=np.int32)
            roi_mask = np.zeros((height, width), dtype=np.uint8)
            cv2.fillPoly(roi_mask, [roi_points], 255)  # Fill mask for the static ROI
            # logger.info(f"ROI set for motion detection on camera {camera_id}")
        else:
            roi_mask = None

        frame_counter = 0  # Frame counter to track the number of frames processed

        while not stop_event.is_set():
            # start_time=time.time()
            frame = queues_dict[f"{camera_id}_{typ}"].get(timeout=10)  # Handle timeouts if frame retrieval takes too long
            if frame is None:
                continue

            # Increment the frame counter
            frame_counter += 1

            # Skip processing for every 5th frame
            if frame_counter % 5 == 0:
                queues_dict[f"{camera_id}_{typ}"].task_done()
                continue  # Skip this frame and continue to the next iteration

            
            # # Log the queue size
            # queue_size = queues_dict[f"{camera_id}_{typ}"].qsize()
            # logger.info(f"fall---: {queue_size}")

            masked_frame = cv2.bitwise_and(frame, frame, mask=roi_mask) if roi_mask is not None else frame

            # Run YOLOv8 inference on the masked frame
            results = model(masked_frame, stream=True, verbose=False)

            for result in results:
                if result.keypoints.conf is not None:
                    xy_array = result.keypoints.xy.cpu().numpy()  # Convert to NumPy array
                    conf_array = result.keypoints.conf.cpu().numpy()  # Get confidence values

                    # Access individual keypoints from the xy array
                    if xy_array.size > 0:
                        for i in range(xy_array.shape[0]):  # Loop through each person's keypoints
                            # Process for fall detection
                            process_frame(xy_array[i], conf_array[i])

            # Check if fall detection text should be displayed
            if fall_detected_time and (time.time() - fall_detected_time) > 10:
                executor.submit(capture_and_publish, camera_id, s_id, typ)
            queues_dict[f"{camera_id}_{typ}"].task_done()
            # logger.info(f"fall----- {(time.time() - start_time) * 1000= :.2f} milliseconds.")

    except Exception as e:
        logger.error(f"Error During Fall Detection:{str(e)}")
        return FallError(f"Fall Detection Failed for camera : {camera_id}")

def fall_start(c_id, s_id, typ, co, width, height, rtsp):
    """
    Start the motion detection process in a separate thread for the given camera task.
    """
    try:
        if f"{c_id}_{typ}_detect" in global_thread:
            fall_stop(c_id, typ)
            time.sleep(2)
        executor.submit(start_feature_processing, c_id, typ, rtsp, width, height)
        stop_event = threading.Event()  # Create a stop event for each feature
        global_thread[f"{c_id}_{typ}_detect"] = stop_event
        executor.submit(fall_detect, c_id, s_id, typ, co, width, height, stop_event)

        logger.info(f"Started fall detection for camera {c_id}.")

    except Exception as e:
        logger.error(f"Failed to start fall detection process for camera {c_id}: {str(e)}", exc_info=True)
        return False
    return True

def fall_stop(camera_id, typ):
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
