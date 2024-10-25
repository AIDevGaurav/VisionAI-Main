import queue
import threading
import time
from collections import deque
import cv2
import numpy as np
from app.exceptions import PCError
from app.config import logger, global_thread, queues_dict, executor
from app.mqtt_handler import publish_message_mqtt as pub
from app.utils import capture_image, start_feature_processing
from ultralytics import YOLO


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
        pub("fire/detection", message)
        logger.info(f"Published fire message for camera {c_id}.")
    except Exception as e:
        logger.error(f"Error capturing image or publishing MQTT for camera {c_id}: {str(e)}")

def check_color(frame, box):
    x1, y1, x2, y2 = map(int, box)
    roi = frame[y1:y2, x1:x2]
    return np.mean(roi[:, :, 2]) > 200  # Check if average red intensity is high


def detect_fire(camera_id, s_id, typ, coordinates, width, height, stop_event):
    """
    Fire detection function that captures video frames, performs inference,
    and publishes a message if fire is detected.
    """
    try:
        # Define parameters for fire detection
        FIRE_CONFIDENCE_THRESHOLD = 0.5
        MIN_FIRE_SIZE = 100  # Minimum area of fire bounding box
        PERSISTENCE_THRESHOLD = 5  # Number of consecutive frames to confirm fire

        fire_model = YOLO('Model/fire.pt')

        if coordinates and "points" in coordinates and coordinates["points"]:
            roi_points = np.array(set_roi_based_on_points(coordinates["points"], coordinates), dtype=np.int32)
            roi_mask = np.zeros((height, width), dtype=np.uint8)
            cv2.fillPoly(roi_mask, [roi_points], 255)  # Fill mask for the static ROI
        else:
            roi_mask = None

        fire_persistence = deque(maxlen=PERSISTENCE_THRESHOLD)

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
            # logger.info(f"fire---: {queue_size}")

            masked_frame = cv2.bitwise_and(frame, frame, mask=roi_mask) if roi_mask is not None else frame

            # Detect fires in the masked frame
            fire_results = fire_model(masked_frame, stream=True, verbose=False)

            current_fire_detected = False
            for result in fire_results:
                for box in result.boxes:
                    confidence = float(box.conf[0])
                    if confidence >= FIRE_CONFIDENCE_THRESHOLD:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])

                        # Check fire size
                        fire_area = (x2 - x1) * (y2 - y1)
                        if fire_area < MIN_FIRE_SIZE:
                            continue

                        # Check color characteristics
                        if not check_color(masked_frame, (x1, y1, x2, y2)):
                            continue

                        current_fire_detected = True

            # Update fire persistence
            fire_persistence.append(current_fire_detected)

            # Check if fire has been consistently detected
            fire_detected = all(fire_persistence) if len(fire_persistence) == PERSISTENCE_THRESHOLD else False

            # Print detection status
            if fire_detected:
                # print("Fire detected!")
                executor.submit(capture_and_publish, frame, camera_id, s_id, typ)

            queues_dict[f"{camera_id}_{typ}"].task_done()
            # logger.info(f"fire----- {(time.time() - start_time) * 1000:.2f} milliseconds.")

    except Exception as e:
        logger.error(f"Error During Fire detection:{str(e)}")
        return PCError(f"Fire detection Failed for camera : {camera_id}")


def fire_start(c_id, s_id, typ, co, width, height, rtsp):
    """
    Start the fire detection process in a separate thread for the given camera task.
    """
    try:
        if f"{c_id}_{typ}_detect" in global_thread:
            fire_stop(c_id, typ)
            time.sleep(2)
        executor.submit(start_feature_processing, c_id, typ, rtsp, width, height)
        stop_event = threading.Event()  # Create a stop event for each feature
        global_thread[f"{c_id}_{typ}_detect"] = stop_event
        executor.submit(detect_fire, c_id, s_id, typ, co, width, height, stop_event)

        logger.info(f"Started Fire detection for camera {c_id}.")

    except Exception as e:
        logger.error(f"Failed to start detection process for camera {c_id}: {str(e)}", exc_info=True)
        return False
    return True

def fire_stop(camera_id, typ):
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




# import queue
# import threading
# import time
# from collections import deque
# import cv2
# import numpy as np
# from app.exceptions import PCError
# from app.config import logger, global_thread, queues_dict, executor
# from app.mqtt_handler import publish_message_mqtt as pub
# from app.utils import capture_image, start_feature_processing
# from ultralytics import YOLO


# # Function to adjust ROI points based on provided coordinates
# def set_roi_based_on_points(points, coordinates):
#     x_offset, y_offset = coordinates["x"], coordinates["y"]
#     return [(int(x + x_offset), int(y + y_offset)) for x, y in points]


# def capture_and_publish(frame, c_id, s_id, typ):
#     try:
#         image_path = capture_image(frame)  # Assuming this function saves the image and returns the path
#         message = {
#             "cameraId": c_id,
#             "siteId": s_id,
#             "type": typ,
#             "image": image_path,

#         }
#         pub("fire/detection", message)
#         logger.info(f"Published fire message for camera {c_id}.")
#     except Exception as e:
#         logger.error(f"Error capturing image or publishing MQTT for camera {c_id}: {str(e)}")

# def create_mask(frame, boxes, padding=10):
#     mask = np.ones(frame.shape[:2], dtype=np.uint8) * 255
#     for box in boxes:
#         x1, y1, x2, y2 = map(int, box[:4])
#         cv2.rectangle(mask, (max(0, x1 - padding), max(0, y1 - padding)),
#                       (min(frame.shape[1], x2 + padding), min(frame.shape[0], y2 + padding)), 0, -1)
#     return mask

# def check_color(frame, box):
#     x1, y1, x2, y2 = map(int, box)
#     roi = frame[y1:y2, x1:x2]
#     return np.mean(roi[:, :, 2]) > 200  # Check if average red intensity is high


# def detect_fire(camera_id, s_id, typ, coordinates, width, height, stop_event):
#     """
#     Fire detection function that captures video frames, performs inference,
#     and publishes a message if fire is detected.
#     """
#     try:
#         # Define parameters for fire detection
#         FIRE_CONFIDENCE_THRESHOLD = 0.5
#         MIN_FIRE_SIZE = 100  # Minimum area of fire bounding box
#         PERSISTENCE_THRESHOLD = 5  # Number of consecutive frames to confirm fire

#         fire_model = YOLO('Model/fire.pt')
#         device_model = YOLO('Model/yolov8l.pt')


#         # Define the devices we want to detect
#         DEVICES = ["cell phone", "laptop", "tv"]

#         if coordinates and "points" in coordinates and coordinates["points"]:
#             roi_points = np.array(set_roi_based_on_points(coordinates["points"], coordinates), dtype=np.int32)
#             roi_mask = np.zeros((height, width), dtype=np.uint8)
#             cv2.fillPoly(roi_mask, [roi_points], 255)  # Fill mask for the static ROI
#         else:
#             roi_mask = None

#         fire_persistence = deque(maxlen=PERSISTENCE_THRESHOLD)

#         frame_counter = 0  # Frame counter to track the number of frames processed

#         while not stop_event.is_set():
#             # start_time=time.time()
#             frame = queues_dict[f"{camera_id}_{typ}"].get(timeout=10)  # Handle timeouts if frame retrieval takes too long
#             if frame is None:
#                 continue

#             # Increment the frame counter
#             frame_counter += 1

#             # Skip processing for every 5th frame
#             if frame_counter % 5 == 0:
#                 queues_dict[f"{camera_id}_{typ}"].task_done()
#                 continue  # Skip this frame and continue to the next iteration
            
#             # Log the queue size
#             queue_size = queues_dict[f"{camera_id}_{typ}"].qsize()
#             logger.info(f"fire---: {queue_size}")

#             masked_frame = cv2.bitwise_and(frame, frame, mask=roi_mask) if roi_mask is not None else frame

#             # Detect devices first
#             device_results = device_model(masked_frame, stream=True, verbose=False)
#             device_boxes = []
#             for result in device_results:
#                 for box in result.boxes:
#                     class_name = device_model.names[int(box.cls[0])]
#                     if class_name in DEVICES:
#                         device_boxes.append((box.xyxy[0].cpu().numpy(), class_name))

#             # Create a mask to exclude device areas
#             mask = create_mask(masked_frame, [box for box, _ in device_boxes])

#             # Apply the mask to the frame
#             fire_frame = cv2.bitwise_and(masked_frame, masked_frame, mask=mask)

#             # Detect fires in the masked frame
#             fire_results = fire_model(fire_frame, stream=True, verbose=False)

#             current_fire_detected = False
#             for result in fire_results:
#                 for box in result.boxes:
#                     confidence = float(box.conf[0])
#                     if confidence >= FIRE_CONFIDENCE_THRESHOLD:
#                         x1, y1, x2, y2 = map(int, box.xyxy[0])

#                         # Check fire size
#                         fire_area = (x2 - x1) * (y2 - y1)
#                         if fire_area < MIN_FIRE_SIZE:
#                             continue

#                         # Check color characteristics
#                         if not check_color(fire_frame, (x1, y1, x2, y2)):
#                             continue

#                         current_fire_detected = True

#             # Update fire persistence
#             fire_persistence.append(current_fire_detected)

#             # Check if fire has been consistently detected
#             fire_detected = all(fire_persistence) if len(fire_persistence) == PERSISTENCE_THRESHOLD else False

#             # Print detection status
#             if fire_detected:
#                 # print("Fire detected!")
#                 executor.submit(capture_and_publish, fire_frame, camera_id, s_id, typ)

#             queues_dict[f"{camera_id}_{typ}"].task_done()
#             # logger.info(f"fire----- {(time.time() - start_time) * 1000:.2f} milliseconds.")

#     except Exception as e:
#         logger.error(f"Error During Fire detection:{str(e)}")
#         return PCError(f"Fire detection Failed for camera : {camera_id}")


# def fire_start(c_id, s_id, typ, co, width, height, rtsp):
#     """
#     Start the fire detection process in a separate thread for the given camera task.
#     """
#     try:
#         if f"{c_id}_{typ}_detect" in global_thread:
#             fire_stop(c_id, typ)
#             time.sleep(2)
#         executor.submit(start_feature_processing, c_id, typ, rtsp, width, height)
#         stop_event = threading.Event()  # Create a stop event for each feature
#         global_thread[f"{c_id}_{typ}_detect"] = stop_event
#         executor.submit(detect_fire, c_id, s_id, typ, co, width, height, stop_event)

#         logger.info(f"Started Fire detection for camera {c_id}.")

#     except Exception as e:
#         logger.error(f"Failed to start detection process for camera {c_id}: {str(e)}", exc_info=True)
#         return False
#     return True

# def fire_stop(camera_id, typ):
#     stopped_tasks = []
#     not_found_tasks = []

#     key = f"{camera_id}_{typ}"  # Construct the key as used in the dictionary
#     key2 = f"{camera_id}_{typ}_detect"

#     try:
#         if key in global_thread and key in queues_dict and key2 in global_thread:
#             stop_event_detect = global_thread[key]  # Retrieve the stop event from the dictionary
#             stop_event_detect.set()  # Signal the thread to stop
#             del global_thread[key]  # Delete the entry from the dictionary after setting the stop event
#             stop_event = global_thread[key2]  # Retrieve the stop event from the dictionary
#             stop_event.set()  # Signal the thread to stop
#             del global_thread[key2]  # Delete the entry from the dictionary after setting the stop event
#             stopped_tasks.append(camera_id)
#             logger.info(f"Stopped {typ} and removed key for camera {camera_id} of type {typ}.")
#         else:
#             not_found_tasks.append(camera_id)
#             logger.warning(f"No active detection found for {camera_id} of type {typ}.")
#     except Exception as e:
#         logger.error(f"Error during stopping detection for {camera_id}: {str(e)}", exc_info=True)

#     return {
#         "success": len(stopped_tasks) > 0,
#         "stopped": stopped_tasks,
#         "not_found": not_found_tasks
#     }

