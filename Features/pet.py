import queue
import threading
import time
import cv2
import numpy as np
from ultralytics import YOLO

from app.exceptions import PCError
from app.config import logger, global_thread, queues_dict, get_executor
from app.mqtt_handler import publish_message_mqtt as pub
from app.utils import capture_image, start_feature_processing

executor = get_executor()

def create_mask(frame, boxes, padding=10):
    mask = np.ones(frame.shape[:2], dtype=np.uint8) * 255
    for box in boxes:
        x1, y1, x2, y2 = map(int, box[:4])
        cv2.rectangle(mask, (max(0, x1 - padding), max(0, y1 - padding)),
                      (min(frame.shape[1], x2 + padding), min(frame.shape[0], y2 + padding)), 0, -1)
    return mask


def capture_and_publish(frame, c_id, s_id, typ):
    try:
        image_path = capture_image(frame)  # Assuming this function saves the image and returns the path
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
    x_offset = coordinates["x"]
    y_offset = coordinates["y"]

    scaled_points = []
    for point in points:
        scaled_x = int(point[0] + x_offset)
        scaled_y = int(point[1] + y_offset)
        scaled_points.append((scaled_x, scaled_y))

    return scaled_points

def detect_pet(camera_id, s_id, typ, coordinates, width, height, stop_event):
    try:
        model = YOLO('Model/yolov8l.pt')

        # Pet class indices in COCO dataset
        pet_classes = [14, 15, 16, 17, 18, 19, 20, 21, 22, 23]  # Bird, Cat, Dog, etc.

        # Define the devices we want to detect
        DEVICES = ["cell phone", "laptop", "tv"]

        if coordinates and "points" in coordinates and coordinates["points"]:
            roi_points = np.array(set_roi_based_on_points(coordinates["points"], coordinates), dtype=np.int32)
            roi_mask = np.zeros((height, width), dtype=np.uint8)
            cv2.fillPoly(roi_mask, [roi_points], 255)  # Fill mask for the static ROI
            logger.info(f"ROI set for pet detection on camera {camera_id}")
        else:
            roi_mask = None

        last_detection_time = 0

        while not stop_event.is_set():
            frame = queues_dict[f"{camera_id}_{typ}"].get(timeout=10)
            if frame is None:
                continue

            if roi_mask is not None:
                masked_frame = cv2.bitwise_and(frame, frame, mask=roi_mask)
                cv2.polylines(frame, [roi_points], isClosed=True, color=(255, 0, 0), thickness=2)
            else:
                masked_frame = frame

            # Detect devices first
            device_results = model(masked_frame)
            device_boxes = []
            for result in device_results:
                for box in result.boxes:
                    class_name = model.names[int(box.cls[0])]
                    if class_name in DEVICES:
                        device_boxes.append((box.xyxy[0].cpu().numpy(), class_name))

                    # Draw device boxes
                    for box, class_name in device_boxes:
                        x1, y1, x2, y2 = map(int, box)
                        # cv2.rectangle(masked_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        # cv2.putText(masked_frame, class_name, (x1, y1 - 10),
                        #             cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Create a mask to exclude device areas
            mask = create_mask(masked_frame, [box for box, _ in device_boxes])

            # Apply the mask to the frame
            pet_frame = cv2.bitwise_and(masked_frame, masked_frame, mask=mask)

            # Detect fires in the masked frame
            pet_results = model(pet_frame)

            pet_detected = False

            for info in pet_results:
                parameters = info.boxes
                for box in parameters:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    class_detect = int(box.cls[0])

                    if class_detect in pet_classes:
                        pet_detected = True

                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                        cv2.putText(frame, "Pet", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            if pet_detected:
                current_time = time.time()
                if current_time - last_detection_time > 10:
                    executor.submit(capture_and_publish, frame, camera_id, s_id, typ)
                    last_detection_time = current_time

            cv2.imshow(f"PET {camera_id}_{typ}", frame)
            queues_dict[f"{camera_id}_{typ}"].task_done()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        logger.error(f"Error During Pet detection:{str(e)}")
        return PCError(f"Pet detection Failed for camera : {camera_id}")
    finally:
        cv2.destroyWindow(f'PET {camera_id}_{typ}')


def pet_start(c_id, s_id, typ, co, width, height, rtsp):
    try:
        if f"{c_id}_{typ}_detect" in global_thread:
            pet_stop(c_id, typ)
        executor.submit(start_feature_processing, c_id, typ, rtsp, width, height)
        stop_event = threading.Event()
        global_thread[f"{c_id}_{typ}_detect"] = stop_event
        executor.submit(detect_pet, c_id, s_id, typ, co, width, height, stop_event)
        logger.info(f"Started pet detection for camera {c_id}.")
    except Exception as e:
        logger.error(f"Failed to start pet detection process for camera {c_id}: {str(e)}", exc_info=True)
        return False
    return True

def pet_stop(camera_id, typ):
    stopped_tasks = []
    not_found_tasks = []

    key = f"{camera_id}_{typ}"
    key2 = f"{camera_id}_{typ}_detect"

    try:
        if key in global_thread and key in queues_dict and key2 in global_thread:
            stop_event = global_thread[key]
            stop_event.set()
            del global_thread[key]
            stop_event = global_thread[key2]
            stop_event.set()
            del global_thread[key2]
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
# from ultralytics import YOLO
#
# from app.exceptions import PCError
# from app.config import logger, global_thread, queues_dict, get_executor
# from app.mqtt_handler import publish_message_mqtt as pub
# from app.utils import capture_image, start_feature_processing
#
# executor = get_executor()
#
# def capture_and_publish(frame, c_id, s_id, typ):
#     try:
#         image_path = capture_image(frame)  # Assuming this function saves the image and returns the path
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
#
# def set_roi_based_on_points(points, coordinates):
#     x_offset = coordinates["x"]
#     y_offset = coordinates["y"]
#
#     scaled_points = []
#     for point in points:
#         scaled_x = int(point[0] + x_offset)
#         scaled_y = int(point[1] + y_offset)
#         scaled_points.append((scaled_x, scaled_y))
#
#     return scaled_points
#
# def detect_pet(camera_id, s_id, typ, coordinates, width, height, stop_event):
#     try:
#         model = YOLO('Model/yolov8n.pt')
#
#         if coordinates and "points" in coordinates and coordinates["points"]:
#             roi_points = np.array(set_roi_based_on_points(coordinates["points"], coordinates), dtype=np.int32)
#             roi_mask = np.zeros((height, width), dtype=np.uint8)
#             cv2.fillPoly(roi_mask, [roi_points], 255)  # Fill mask for the static ROI
#         else:
#             roi_mask = None
#
#         last_detection_time = 0
#
#         while not stop_event.is_set():
#             start_time = time.time()
#             frame = queues_dict[f"{camera_id}_{typ}"].get(timeout=10)
#             if frame is None:
#                 continue
#             # Log the queue size
#             queue_size = queues_dict[f"{camera_id}_{typ}"].qsize()
#             logger.info(f"pet---: {queue_size}")
#
#             if roi_mask is not None:
#                 masked_frame = cv2.bitwise_and(frame, frame, mask=roi_mask)
#             else:
#                 masked_frame = frame
#
#             results = model(masked_frame, stream=True, verbose=False)
#
#             pet_detected = False
#
#             for info in results:
#                 parameters = info.boxes
#                 for box in parameters:
#                     class_detect = int(box.cls[0])
#
#                     if class_detect in [14, 15, 16, 17, 18, 19, 20, 21, 22, 23]:
#                         pet_detected = True
#
#             if pet_detected and ((time.time() - last_detection_time) > 10):
#                 executor.submit(capture_and_publish, frame, camera_id, s_id, typ)
#                 last_detection_time = time.time()
#
#             queues_dict[f"{camera_id}_{typ}"].task_done()
#             frame_end_time = time.time()
#             frame_processing_time_ms = (frame_end_time - start_time) * 1000
#             logger.info(f"pet----- {frame_processing_time_ms:.2f} milliseconds.")
#
#     except Exception as e:
#         logger.error(f"Error During Pet detection:{str(e)}")
#         return PCError(f"Pet detection Failed for camera : {camera_id}")
#
# def pet_start(c_id, s_id, typ, co, width, height, rtsp):
#     try:
#         if f"{c_id}_{typ}_detect" in global_thread:
#             pet_stop(c_id, typ)
#             time.sleep(2)
#         executor.submit(start_feature_processing, c_id, typ, rtsp, width, height)
#         stop_event = threading.Event()
#         global_thread[f"{c_id}_{typ}_detect"] = stop_event
#         executor.submit(detect_pet, c_id, s_id, typ, co, width, height, stop_event)
#         logger.info(f"Started pet detection for camera {c_id}.")
#     except Exception as e:
#         logger.error(f"Failed to start pet detection process for camera {c_id}: {str(e)}", exc_info=True)
#         return False
#     return True
#
# def pet_stop(camera_id, typ):
#     stopped_tasks = []
#     not_found_tasks = []
#
#     key = f"{camera_id}_{typ}"
#     key2 = f"{camera_id}_{typ}_detect"
#
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
#
#     return {
#         "success": len(stopped_tasks) > 0,
#         "stopped": stopped_tasks,
#         "not_found": not_found_tasks
#     }





