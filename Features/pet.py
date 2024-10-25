import threading
import time
import cv2
import numpy as np
from app.exceptions import PCError
from app.config import logger, global_thread, queues_dict, executor
from app.mqtt_handler import publish_message_mqtt as pub
from app.utils import capture_image, start_feature_processing
from ultralytics import YOLO

def capture_and_publish(frame, c_id, s_id, typ):
    try:
        image_path = capture_image(frame)
        message = {
            "cameraId": c_id,
            "siteId": s_id,
            "type": typ,
            "image": image_path,
        }
        pub("pet/detection", message)
        logger.info(f"Published pet message for camera {c_id}.")
    except Exception as e:
        logger.error(f"Error capturing image or publishing MQTT for camera {c_id}: {str(e)}")

def set_roi_based_on_points(points, coordinates):
    x_offset, y_offset = coordinates["x"], coordinates["y"]
    return [(int(p[0] + x_offset), int(p[1] + y_offset)) for p in points]

def detect_pet(camera_id, s_id, typ, coordinates, width, height, stop_event):
    try:
        model = YOLO("Model/yolov8l.pt")

        if coordinates and "points" in coordinates:
            roi_points = np.array(set_roi_based_on_points(coordinates["points"], coordinates), dtype=np.int32)
            roi_mask = np.zeros((height, width), dtype=np.uint8)
            cv2.fillPoly(roi_mask, [roi_points], 255)
            # logger.info(f"ROI set for pet detection on camera {camera_id}")
        else:
            roi_mask = None

        last_detection_time = 0
        
        frame_counter = 0  # Frame counter to track the number of frames processed

        while not stop_event.is_set():
            # start_time = time.time()
            frame = queues_dict[f"{camera_id}_{typ}"].get(timeout=10)  # Handle timeouts if frame retrieval takes too long
            if frame is None:
                continue

            # Increment the frame counter
            frame_counter += 1

            # Skip processing for every 5th frame
            if frame_counter % 5 == 0:
                queues_dict[f"{camera_id}_{typ}"].task_done()
                continue  # Skip this frame and continue to the next iteration

            #    # Log the queue size
            # queue_size = queues_dict[f"{camera_id}_{typ}"].qsize()
            # logger.info(f"pet---: {queue_size}")

            masked_frame = cv2.bitwise_and(frame, frame, mask=roi_mask) if roi_mask is not None else frame
        
            results = model(masked_frame, verbose=False, stream=True, classes = [14, 15, 16, 17, 18, 19, 20, 21, 22, 23])

            for result in results:
                if hasattr(result, 'boxes') and result.boxes.data.shape[0] > 0:  # Check if there are any detected boxes
                    if time.time() - last_detection_time > 10:
                        executor.submit(capture_and_publish, frame, camera_id, s_id, typ)
                        last_detection_time = time.time()

            queues_dict[f"{camera_id}_{typ}"].task_done()
            # logger.info(f"Pet----- {(time.time() - start_time) * 1000:.2f} milliseconds.")


    except Exception as e:
        logger.error(f"Error During Pet detection: {str(e)}")
        return PCError(f"Pet detection Failed for camera: {camera_id}")

def pet_start(c_id, s_id, typ, co, width, height, rtsp):
    """
    Start the fire detection process in a separate thread for the given camera task.
    """
    try:
        if f"{c_id}_{typ}_detect" in global_thread:
            pet_stop(c_id, typ)
            time.sleep(2)
        executor.submit(start_feature_processing, c_id, typ, rtsp, width, height)
        stop_event = threading.Event()  # Create a stop event for each feature
        global_thread[f"{c_id}_{typ}_detect"] = stop_event
        executor.submit(detect_pet, c_id, s_id, typ, co, width, height, stop_event)

        logger.info(f"Started Fire detection for camera {c_id}.")

    except Exception as e:
        logger.error(f"Failed to start detection process for camera {c_id}: {str(e)}", exc_info=True)
        return False
    return True

def pet_stop(camera_id, typ):
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
# import cv2
# import numpy as np
# from app.exceptions import PCError
# from app.config import logger, global_thread, queues_dict, executor
# from app.mqtt_handler import publish_message_mqtt as pub
# from app.utils import capture_image, start_feature_processing
# from ultralytics import YOLO

# def create_mask(frame, boxes, padding=10):
#     mask = np.ones(frame.shape[:2], dtype=np.uint8) * 255
#     for box in boxes:
#         x1, y1, x2, y2 = map(int, box[:4])
#         cv2.rectangle(mask, (max(0, x1 - padding), max(0, y1 - padding)),
#                       (min(frame.shape[1], x2 + padding), min(frame.shape[0], y2 + padding)), 0, -1)
#     return mask

# def capture_and_publish(frame, c_id, s_id, typ):
#     try:
#         image_path = capture_image(frame)
#         message = {
#             "cameraId": c_id,
#             "siteId": s_id,
#             "type": typ,
#             "image": image_path,
#         }
#         pub("pet/detection", message)
#         logger.info(f"Published pet message for camera {c_id}.")
#     except Exception as e:
#         logger.error(f"Error capturing image or publishing MQTT for camera {c_id}: {str(e)}")

# def set_roi_based_on_points(points, coordinates):
#     x_offset, y_offset = coordinates["x"], coordinates["y"]
#     return [(int(p[0] + x_offset), int(p[1] + y_offset)) for p in points]

# def detect_pet(camera_id, s_id, typ, coordinates, width, height, stop_event):
#     try:
#         model = YOLO("Model/yolov8l.pt")
#         pet_classes = [14, 15, 16, 17, 18, 19, 20, 21, 22, 23]  # Pet classes from COCO dataset
#         DEVICES = ["cell phone", "laptop", "tv"]

#         if coordinates and "points" in coordinates:
#             roi_points = np.array(set_roi_based_on_points(coordinates["points"], coordinates), dtype=np.int32)
#             roi_mask = np.zeros((height, width), dtype=np.uint8)
#             cv2.fillPoly(roi_mask, [roi_points], 255)
#             # logger.info(f"ROI set for pet detection on camera {camera_id}")
#         else:
#             roi_mask = None

#         last_detection_time = 0

#         while not stop_event.is_set():
#             # start_time = time.time()
#             frame = queues_dict[f"{camera_id}_{typ}"].get(timeout=10)
#             if frame is None:
#                 continue

#             #    # Log the queue size
#             # queue_size = queues_dict[f"{camera_id}_{typ}"].qsize()
#             # logger.info(f"pet---: {queue_size}")

#             masked_frame = cv2.bitwise_and(frame, frame, mask=roi_mask) if roi_mask is not None else frame
        
#             device_results = model(masked_frame, verbose = False, stream=True)
#             device_boxes = [(box.xyxy[0].cpu().numpy(), model.names[int(box.cls[0])])
#                             for result in device_results for box in result.boxes if model.names[int(box.cls[0])] in DEVICES]

#             mask = create_mask(masked_frame, [box for box, _ in device_boxes])
#             pet_frame = cv2.bitwise_and(masked_frame, masked_frame, mask=mask)
#             pet_results = model(pet_frame, verbose=False, stream=True)

#             pet_detected = False
#             for info in pet_results:
#                 for box in info.boxes:
#                     if int(box.cls[0]) in pet_classes:
#                         pet_detected = True

#             if pet_detected and time.time() - last_detection_time > 10:
#                 executor.submit(capture_and_publish, frame, camera_id, s_id, typ)
#                 last_detection_time = time.time()

#             queues_dict[f"{camera_id}_{typ}"].task_done()
#             # logger.info(f"people----- {(time.time() - start_time) * 1000:.2f} milliseconds.")


#     except Exception as e:
#         logger.error(f"Error During Pet detection: {str(e)}")
#         return PCError(f"Pet detection Failed for camera: {camera_id}")
#     finally:
#         cv2.destroyWindow(f'PET {camera_id}_{typ}')

# def pet_start(c_id, s_id, typ, co, width, height, rtsp):
#     """
#     Start the fire detection process in a separate thread for the given camera task.
#     """
#     try:
#         if f"{c_id}_{typ}_detect" in global_thread:
#             pet_stop(c_id, typ)
#             time.sleep(2)
#         executor.submit(start_feature_processing, c_id, typ, rtsp, width, height)
#         stop_event = threading.Event()  # Create a stop event for each feature
#         global_thread[f"{c_id}_{typ}_detect"] = stop_event
#         executor.submit(detect_pet, c_id, s_id, typ, co, width, height, stop_event)

#         logger.info(f"Started Fire detection for camera {c_id}.")

#     except Exception as e:
#         logger.error(f"Failed to start detection process for camera {c_id}: {str(e)}", exc_info=True)
#         return False
#     return True

# def pet_stop(camera_id, typ):
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

