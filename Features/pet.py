import queue
import threading
import time
import cv2
import numpy as np
from app.exceptions import PCError
from app.config import logger, global_thread, queues_dict, YOLOv8Single, get_executor
from app.mqtt_handler import publish_message_mqtt as pub
from app.utils import capture_image, start_feature_processing

executor = get_executor()

# COCO class names (only including pet-related classes)
classnames = [
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe'
]

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
        model = YOLOv8Single()

        # Pet class indices in COCO dataset
        pet_classes = [14, 15, 16, 17, 18, 19, 20, 21, 22, 23]  # Bird, Cat, Dog, etc.

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

            results = model(masked_frame, stream=True, verbose=False)

            pet_detected = False

            for info in results:
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

            cv2.imshow(f"Camera {camera_id}_{typ}", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        logger.error(f"Error During Pet detection:{str(e)}")
        return PCError(f"Pet detection Failed for camera : {camera_id}")

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
# from app.exceptions import PCError
# from app.config import logger, global_thread, queues_dict, YOLOv8Single, get_executor
# from app.mqtt_handler import publish_message_mqtt as pub
# from app.utils import capture_image
# import math
#
#
# executor = get_executor()
#
# # COCO class names
# classnames = [
#     'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
#     'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
#     'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
#     'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
#     'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
#     'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
#     'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
#     'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
#     'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
#     'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
# ]
# def capture_and_publish(frame, c_id, s_id, typ, ):
#     try:
#         image_path = capture_image(frame)  # Assuming this function saves the image and returns the path
#         message = {
#             "cameraId": c_id,
#             "siteId": s_id,
#             "type": typ,
#             "image": image_path,
#
#         }
#         pub("pet/detection", message)
#         logger.info(f"Published pet message for camera {c_id}.")
#     except Exception as e:
#         logger.error(f"Error capturing image or publishing MQTT for camera {c_id}: {str(e)}")
#
#
# # Function to calculate Euclidean distance between two bounding boxes
# def calculate_distance(box1, box2):
#     center_x1, center_y1 = (box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2
#     center_x2, center_y2 = (box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2
#
#     distance = math.sqrt((center_x2 - center_x1) ** 2 + (center_y2 - center_y1) ** 2)
#     return distance
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
#
# # Function to detect animals and phones, and determine proximity
# def detect_pet(camera_id, s_id, typ, coordinates, width, height, stop_event):
#     try:
#         # YOLO model (pre-trained on COCO dataset)
#         model = YOLOv8Single()
#
#         # Animal class indices in COCO dataset
#         animal_classes = [14, 15, 16, 17, 18, 19, 20, 21, 22, 23]  # Bird, Cat, Dog, etc.
#         # Adding cell phone to the detection list (ID 67 in COCO dataset)
#         animal_and_phone_classes = animal_classes + [67]  # Adding cell phone class ID to the list
#
#
#         if coordinates["points"]:
#             roi_points = np.array(set_roi_based_on_points(coordinates["points"], coordinates), dtype=np.int32)
#             roi_mask = np.zeros((height, width), dtype=np.uint8)
#             cv2.fillPoly(roi_mask, [roi_points], 255)  # Fill mask for the static ROI
#             logger.info(f"ROI set for pet detection on camera {camera_id}")
#         else:
#             roi_mask = None
#
#         last_detection_time = 0
#
#         while not stop_event.is_set():
#             # start_time = time.time()
#             frame = queues_dict[f"{camera_id}_{typ}"].get(
#                 timeout=10)  # Handle timeouts if frame retrieval takes too long
#             if frame is None:
#                 continue
#
#             # # Log the queue size
#             # queue_size = queues_dict[f"{camera_id}_{typ}"].qsize()
#             # logger.info(f"pet: {queue_size}")
#
#             if roi_mask is not None:
#                 masked_frame = cv2.bitwise_and(frame, frame, mask=roi_mask)
#                 cv2.polylines(frame, [roi_points], isClosed=True, color=(255, 0, 0), thickness=2)
#             else:
#                 masked_frame = frame
#
#             # Run YOLOv8 inference on the masked frame
#             results = model(masked_frame, conf=0.3, iou=0.4, verbose=False)
#
#             pet_box = None
#             phone_box = None
#
#             for info in results:
#                 parameters = info.boxes
#                 for box in parameters:
#                     x1, y1, x2, y2 = box.xyxy[0]
#                     x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
#                     confidence = box.conf[0]
#                     class_detect = int(box.cls[0])
#
#                     if class_detect in animal_and_phone_classes:
#                         class_name = classnames[class_detect]
#                         conf = confidence * 100
#
#                         # Draw bounding box and label
#                         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                         label = f'{class_name} {conf:.2f}%'
#                         cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
#
#                         if class_detect in animal_classes:
#                             pet_box = (x1, y1, x2, y2)
#                         elif class_detect == 67:  # Cell phone class
#                             phone_box = (x1, y1, x2, y2)
#
#                 # Check if both a pet and a cell phone were detected
#                 if pet_box and phone_box:
#                     distance = calculate_distance(pet_box, phone_box)
#                     # print(f"Distance between pet and cell phone: {distance}")
#
#                     # Define proximity threshold (you can adjust this based on your needs)
#                     proximity_threshold = 10  # Example threshold, adjust as needed
#
#                     if distance < proximity_threshold:
#                         current_time = time.time()
#                         if current_time - last_detection_time > 10:
#                             executor.submit(capture_and_publish, frame, camera_id, s_id, typ)
#
#                         if cv2.waitKey(1) & 0xFF == ord('q'):
#                             break
#
#     except Exception as e:
#         logger.error(f"Error During Pet detection:{str(e)}")
#         return PCError(f"Pet detection Failed for camera : {camera_id}")
#
#     # finally:
#     #     cv2.destroyWindow(f'Pet detection - Camera {camera_id}')
#
#
# # Function to start detection task
# def pet_start(c_id, s_id, typ, co, width, height):
#     """
#     Start the motion detection process in a separate thread for the given camera task.
#     """
#     try:
#         stop_event = threading.Event()  # Create a stop event for each feature
#         global_thread[f"{c_id}_{typ}_detect"] = stop_event
#         executor.submit(detect_pet, c_id, s_id, typ, co, width, height, stop_event)
#
#         logger.info(f"Started motion detection for camera {c_id}.")
#
#     except Exception as e:
#         logger.error(f"Failed to start detection process for camera {c_id}: {str(e)}", exc_info=True)
#         return False
#     return True
#
# def pet_stop(camera_id, typ):
#     stopped_tasks = []
#     not_found_tasks = []
#
#     key = f"{camera_id}_{typ}"  # Construct the key as used in the dictionary
#     key2 = f"{camera_id}_{typ}_detect"
#
#     try:
#         if key in global_thread and key in queues_dict and key2 in global_thread:
#             stop_event = global_thread[key]  # Retrieve the stop event from the dictionary
#             stop_event.set()  # Signal the thread to stop
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
#
#
#
