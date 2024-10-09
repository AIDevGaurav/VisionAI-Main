# import cv2
# import multiprocessing
# import time
# import json
# from ultralytics import YOLO
# from app.utils import capture_image, capture_video
# from app.mqtt_handler import publish_message_mqtt as pub
# from app.config import logger
# from app.exceptions import PetError
#
# # Animal class indices in COCO dataset
# animal_classes = [15, 16, 17, 18, 19, 20, 21, 22, 23]  # Cat, Dog, etc.
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
#
# # Global dictionary to keep track of processes
# tasks_processes = {}
#
#
# def detect_animal(rtsp_url, camera_id, site_id, display_width, display_height, type, co_ordinate, stop_event):
#     """
#     :tasks: take the rtsp url and start stream to detect animals inside stream through YOLO v8
#     :return: Capture Image, Video and Publish Mqtt message
#     """
#     try:
#         model = YOLO('Model/yolov8l.pt')
#         cap = cv2.VideoCapture(rtsp_url)
#         if not cap.isOpened():
#             raise PetError(f"Failed to open Camera: {camera_id}")
#
#         window_name = f'Animal Detection - Camera {camera_id}'
#         cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)  # Create a window that can be resized
#
#         last_detection_time = 0
#
#         while not stop_event.is_set():
#             ret, frame = cap.read()
#             if not ret:
#                 logger.warning(f"Camera failed id: {camera_id}")
#                 break
#
#             frame = cv2.resize(frame, (display_width, display_height))
#
#             # Run YOLO detection
#             results = model(frame)
#
#             for info in results:
#                 parameters = info.boxes
#                 for box in parameters:
#                     x1, y1, x2, y2 = box.xyxy[0]
#                     x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
#                     confidence = box.conf[0]
#                     class_detect = int(box.cls[0])
#
#                     current_time = time.time()
#                     if class_detect in animal_classes and (current_time - last_detection_time > 10):
#                         class_name = classnames[class_detect]
#                         conf = confidence * 100
#
#                         # Draw bounding box and label
#                         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                         cv2.putText(frame, "Animal-Detected",(x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
#
#                         frame_copy = frame.copy()
#                         image_filename = capture_image(frame_copy)
#                         video_filename = "testing" # capture_video(rtsp_url)
#
#                         # Publish MQTT message
#                         message = {
#                             "cameraId": camera_id,
#                             "class": class_name,
#                             "siteId": site_id,
#                             "type": type,
#                             "image": image_filename,
#                             "video": video_filename
#                         }
#                         pub("pet/detection", message)
#                         # print(f"Published message: {json.dumps(message)}")
#                         last_detection_time = current_time
#
#             # Display the frame
#             cv2.imshow(window_name, frame)
#
#             # Break loop on 'q' key press
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break
#
#         cap.release()
#         cv2.destroyWindow(window_name)
#
#     except Exception as e:
#         logger.error(f"Error for camera:{camera_id} in Pet Detection")
#         raise PetError(f"Error in Pet Detection for camera id {camera_id}")
#
# def pet_start(task):
#     """
#     :param task: Json Array
#     tasks: Format the input data and start the pet detection with multi_processing for multiple cameras
#     :return: True or false
#     """
#     try:
#         camera_id = task["cameraId"]
#         site_id = task["siteId"]
#         display_width = task["display_width"]
#         display_height = task["display_height"]
#         types = task["type"]
#         rtsp_url = task["rtsp_link"]
#         co_ordinate = task["co_ordinates"]
#         if camera_id not in tasks_processes:
#             stop_event = multiprocessing.Event()
#             tasks_processes[camera_id] = stop_event
#
#             # Start PC detection in a new process
#             process = multiprocessing.Process(target=detect_animal, args=(
#                 rtsp_url, camera_id, site_id, display_width, display_height, types, co_ordinate, stop_event))
#             tasks_processes[camera_id] = process
#             process.start()
#             logger.info(f"Started PC detection for camera {camera_id}.")
#         else:
#             logger.warning(f"PC detection already running for camera {camera_id}.")
#             return False
#     except Exception as e:
#         logger.error(f"Failed to start detection process for camera {camera_id}: {str(e)}", exc_info=True)
#         return False
#     return True
#
# def pet_stop(camera_ids):
#     """
#     Stop-pet detection for the given camera IDs.
#     """
#     stopped_tasks = []
#     not_found_tasks = []
#
#     for camera_id in camera_ids:
#         if camera_id in tasks_processes:
#             try:
#                 tasks_processes[camera_id].terminate()  # Stop the process
#                 tasks_processes[camera_id].join()  # Wait for the process to stop
#                 del tasks_processes[camera_id]  # Remove from the dictionary
#                 stopped_tasks.append(camera_id)
#                 logger.info(f"Stopped Pet detection for camera {camera_id}.")
#             except Exception as e:
#                 logger.error(f"Failed to stop Pet detection for camera {camera_id}: {str(e)}", exc_info=True)
#         else:
#             not_found_tasks.append(camera_id)
#
#     return {
#         "success": len(stopped_tasks) > 0,
#         "stopped": stopped_tasks,
#         "not_found": not_found_tasks
#     }
#

# import threading
# import time
# import cv2
# import numpy as np
# from app.config import logger, global_thread, queues_dict, YOLOv8Single, get_executor
# from app.mqtt_handler import publish_message_mqtt as pub
# from app.utils import capture_image
#
#
# executor = get_executor()
#
# def set_roi_based_on_points(points, coordinates):
#     """
#     Scale and set ROI based on given points and coordinates.
#     """
#     x_offset = coordinates["x"]
#     y_offset = coordinates["y"]
#
#     scaled_points = []
#     for point in points:
#         scaled_x = int(point[0] + x_offset)
#         scaled_y = int(point[1] + y_offset)
#         scaled_points.append((scaled_x, scaled_y))
#     return scaled_points
#
# def capture_and_publish(frame, c_id, s_id, typ):
#     try:
#         logger.info(f"Capturing image for motion detected on camera {c_id}.")
#         image_path = capture_image(frame)  # Assuming this function saves the image and returns the path
#         message = {
#             "cameraId": c_id,
#             "siteId": s_id,
#             "type": typ,
#             "image": image_path,
#             "timestamp": time.time()
#         }
#         pub("pet/detection", message)
#         logger.info(f"Published motion detection message for camera {c_id}.")
#     except Exception as e:
#         logger.error(f"Error capturing image or publishing MQTT for camera {c_id}: {str(e)}")
#
#
# def detect_pet(c_id, s_id, typ, co, width, height, stop_event):
#     """
#     pet detection loop with static ROI setup.
#     """
#
#     # COCO class names
#     classnames = [
#         'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
#         'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
#         'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
#         'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
#         'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
#         'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
#         'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
#         'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
#         'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
#         'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
#     ]
#
#     # YOLO model (pre-trained on COCO dataset)
#     model = YOLOv8Single()
#
#     # Animal class indices in COCO dataset
#     animal_classes = [14, 15, 16, 17, 18, 19, 20, 21, 22, 23]  # Bird, Cat, Dog, etc.
#     # Adding electronic device classes (cell phone: 67, tv: 62, laptop: 63)
#     # electronic_classes = [67, 62, 63]  # Cell phone, TV, Laptop
#
#     # prev_frame_gray = None
#     last_detection_time = 0
#
#     try:
#         # Initial check if coordinates are provided and contain necessary data
#         if co["points"]:  # Check if 'points' key exists and is not empty
#             roi_points = np.array(set_roi_based_on_points(co["points"], co), dtype=np.int32)
#             roi_mask = np.zeros((height, width), dtype=np.uint8)
#             cv2.fillPoly(roi_mask, [roi_points], 255)  # Fill mask for the static ROI
#             logger.info(f"ROI set for motion detection on camera {c_id}")
#         else:
#             roi_mask = None
#             logger.info(f"No ROI set, full frame will be processed for camera {c_id}")
#
#         while not stop_event.is_set():
#             # start_time = time.time()
#             frame = queues_dict[f"{c_id}_{typ}"].get(timeout=10)  # Handle timeouts if frame retrieval takes too long
#             if frame is None:
#                 continue
#
#             # queue_size = queues_dict.qsize()
#             # logger.info(f"Current queue size: {queue_size}")
#
#             # First pass: Apply ROI mask if exists and detect electronic devices
#             if roi_mask is not None:
#                 initial_masked_frame = cv2.bitwise_and(frame, frame, mask=roi_mask)
#                 cv2.polylines(frame, [roi_points], isClosed=True, color=(0, 0, 255), thickness=2)
#             else:
#                 initial_masked_frame = frame
#
#             # Detect electronic devices only in ROI or full frame
#             electronic_results = model(initial_masked_frame, classes=[67, 62, 63])
#
#             # Process electronic device detections and create device mask
#             device_detections = {}
#             device_mask = np.zeros((height, width), dtype=np.uint8)
#
#             for info in electronic_results:
#                 parameters = info.boxes
#                 for box in parameters:
#                     x1, y1, x2, y2 = box.xyxy[0]
#                     x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
#                     class_detect = int(box.cls[0])
#                     confidence = float(box.conf[0])
#
#                     device_type = {67: "cellphone", 62: "tv", 63: "laptop"}[class_detect]
#                     device_detections[device_type] = (x1, y1, x2, y2, confidence)
#
#                     # Add device area to device mask
#                     cv2.rectangle(device_mask, (x1, y1), (x2, y2), 255, -1)
#
#             # Create final mask combining ROI and device exclusions
#             if roi_mask is not None:
#                 final_mask = cv2.bitwise_and(roi_mask, cv2.bitwise_not(device_mask))
#             else:
#                 final_mask = cv2.bitwise_not(device_mask)
#
#             # Apply final mask to frame for pet detection
#             final_masked_frame = cv2.bitwise_and(frame, frame, mask=final_mask)
#
#             # Detect pets in remaining area
#             pet_results = model(final_masked_frame, classes=animal_classes)
#
#             pet_detections = False
#             for info in pet_results:
#                 parameters = info.boxes
#                 for box in parameters:
#                     x1, y1, x2, y2 = box.xyxy[0]
#                     x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
#                     class_detect = int(box.cls[0])
#                     confidence = float(box.conf[0])
#
#                     pet_type = classnames[class_detect]
#                     pet_detections=True
#
#             # Draw detections on visualization frame
#             # Draw electronic device boxes
#             for device_type, (x1, y1, x2, y2, conf) in device_detections.items():
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
#                 cv2.putText(frame, f"{device_type} {conf:.2f}", (x1, y1 - 10),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
#
#             # # Draw pet boxes
#             # for pet_type, x1, y1, x2, y2, conf in pet_detections:
#             #     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#             #     cv2.putText(frame, f"{pet_type} {conf:.2f}", (x1, y1 - 10),
#             #                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
#
#             if pet_detections and (time.time() - last_detection_time > 10):
#                 # logger.info(f"Motion detected for camera {c_id}.")
#                 executor.submit(capture_and_publish, frame, c_id, s_id, typ)
#                 last_detection_time = time.time()
#
#             cv2.imshow(f"pet Detection - Camera {c_id}", frame)
#             # prev_frame_gray = gray_frame
#             queues_dict[f"{c_id}_{typ}"].task_done()
#             # frame_end_time = time.time()
#             # frame_processing_time_ms = (frame_end_time - start_time) * 1000
#             # logger.info(f"Frame processed in {frame_processing_time_ms:.2f} milliseconds.")
#
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break
#
#     except Exception as e:
#         logger.error(f"Error during motion detection for camera {c_id}: {str(e)}", exc_info=True)
#         # raise PetError(f"Motion detection failed for camera {c_id}: {str(e)}")
#     finally:
#         cv2.destroyWindow(f"Motion Detection - Camera {c_id}")
#
# def pet_start(c_id, s_id, typ, co, width, height):
#     """
#     Start the motion detection process in a separate thread for the given camera task.
#     """
#     try:
#         stop_event = threading.Event()  # Create a stop event for each feature
#         global_thread[f"{c_id}_{typ}"] = stop_event
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
#
#     try:
#         if key in global_thread:
#             stop_event = global_thread[key]  # Retrieve the stop event from the dictionary
#             stop_event.set()  # Signal the thread to stop
#             del global_thread[key]  # Delete the entry from the dictionary after setting the stop event
#             stopped_tasks.append(camera_id)
#             logger.info(f"Stopped pet detection and removed key for camera {camera_id} of type {typ}.")
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


# flask vala
# detecting
import cv2
import multiprocessing
import time
import json
import os
import math
from ultralytics import YOLO
from flask import Flask, request, jsonify
from flask_cors import CORS
import paho.mqtt.client as mqtt

# Flask app setup
app = Flask(__name__)
CORS(app)

# MQTT configuration
broker = "broker.hivemq.com"  # Replace with your MQTT broker address
port = 1883  # MQTT port
topic = "pet/detection"
mqtt_client = mqtt.Client(client_id="AnimalDetection")

mqtt_client.connect(broker, port, keepalive=300)
mqtt_client.loop_start()

# YOLO model (pre-trained on COCO dataset)
model = YOLO('Model/yolov8l.pt')

# Animal class indices in COCO dataset
animal_classes = [14, 15, 16, 17, 18, 19, 20, 21, 22, 23]  # Bird, Cat, Dog, etc.
# Adding cell phone to the detection list (ID 67 in COCO dataset)
animal_and_phone_classes = animal_classes + [67]  # Adding cell phone class ID to the list

# COCO class names
classnames = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
    'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Global dictionary to keep track of processes
tasks_processes = {}
process_lock = multiprocessing.Lock()

# Ensure directories exist
image_dir = "images"
video_dir = "videos"
os.makedirs(image_dir, exist_ok=True)
os.makedirs(video_dir, exist_ok=True)

# Function to capture and save an image
def capture_image(frame):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    image_filename = os.path.join(image_dir, f"Animal_{timestamp}.jpg")
    cv2.imwrite(image_filename, frame)
    absolute_image_path = os.path.abspath(image_filename)
    return absolute_image_path

# Function to capture and save a 5-second video
def capture_video(rtsp_url):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    video_filename = os.path.join(video_dir, f"Animal_{timestamp}.mp4")

    # Use the MP4V codec for MP4 format
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    cap_video = cv2.VideoCapture(rtsp_url)
    width = int(cap_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create a VideoWriter object with MP4 format
    out = cv2.VideoWriter(video_filename, fourcc, 20.0, (width, height))

    start_time = time.time()
    while int(time.time() - start_time) < 5:  # Capture for 5 seconds
        ret, frame = cap_video.read()
        if not ret:
            break
        out.write(frame)

    cap_video.release()
    out.release()
    absolute_video_path = os.path.abspath(video_filename)
    return absolute_video_path

# Function to calculate Euclidean distance between two bounding boxes
def calculate_distance(box1, box2):
    center_x1, center_y1 = (box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2
    center_x2, center_y2 = (box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2

    distance = math.sqrt((center_x2 - center_x1) ** 2 + (center_y2 - center_y1) ** 2)
    return distance

# Function to detect animals and phones, and determine proximity
def detect_animal(rtsp_url, camera_id, site_id, display_width, display_height, type, stop_event):
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        print(f"Error: Unable to open video stream for camera {camera_id}")
        return

    window_name = f'Animal Detection - Camera {camera_id}'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)  # Create a window that can be resized

    last_detection_time = 0

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (display_width, display_height))

        # Run YOLO detection
        results = model(frame)

        pet_box = None
        phone_box = None

        for info in results:
            parameters = info.boxes
            for box in parameters:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                confidence = box.conf[0]
                class_detect = int(box.cls[0])

                if class_detect in animal_and_phone_classes:
                    class_name = classnames[class_detect]
                    conf = confidence * 100

                    # Draw bounding box and label
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f'{class_name} {conf:.2f}%'
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                    if class_detect in animal_classes:
                        pet_box = (x1, y1, x2, y2)
                    elif class_detect == 67:  # Cell phone class
                        phone_box = (x1, y1, x2, y2)

        # Check if both a pet and a cell phone were detected
        if pet_box and phone_box:
            distance = calculate_distance(pet_box, phone_box)
            print(f"Distance between pet and cell phone: {distance}")

            # Define proximity threshold (you can adjust this based on your needs)
            proximity_threshold = 5  # Example threshold, adjust as needed

            if distance < proximity_threshold:
                current_time = time.time()
                if current_time - last_detection_time > 10:
                    frame_copy = frame.copy()
                    image_filename = capture_image(frame_copy)
                    video_filename = capture_video(rtsp_url)

                    # Publish MQTT message
                    message = {
                        "cameraId": camera_id,
                        "class": "pet-near-cellphone",
                        "siteId": site_id,
                        "type": type,
                        "image": image_filename,
                        "video": video_filename
                    }
                    mqtt_client.publish(topic, json.dumps(message))
                    print(f"Published proximity message: {json.dumps(message)}")
                    last_detection_time = current_time

        # Display the frame
        cv2.imshow(window_name, frame)

        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyWindow(window_name)

# Function to start detection task
def start_detection(task):
    camera_id = task["cameraId"]
    site_id = task["siteId"]
    display_width = task["display_width"]
    display_height = task["display_height"]
    type = task["type"]
    rtsp_url = task["rtsp_link"]
    # co_ordinate = task["co_ordinate"]

    stop_event = multiprocessing.Event()
    process = multiprocessing.Process(target=detect_animal, args=(
        rtsp_url, camera_id, site_id, display_width, display_height, type,  stop_event))

    with process_lock:
        # Stop any existing task for the same camera ID
        if camera_id in tasks_processes:
            print(f"Stopping existing detection for camera {camera_id}")
            tasks_processes[camera_id]['stop_event'].set()  # Signal the existing process to stop
            tasks_processes[camera_id]['process'].join()  # Wait for the existing process to finish
            del tasks_processes[camera_id]  # Remove from the active tasks

        tasks_processes[camera_id] = {'process': process, 'stop_event': stop_event}
        process.start()

# API endpoint to start detection
@app.route('/start', methods=['POST'])
def start_detection_endpoint():
    tasks = request.json
    for task in tasks:
        start_detection(task)
    return jsonify({"message": 'Animal detection tasks started'}), 200

# API endpoint to stop detection
@app.route('/stop', methods=['POST'])
def stop_detection():
    camera_ids = request.json.get('camera_ids', [])
    if not isinstance(camera_ids, list):
        return jsonify({"error": "camera_ids should be an array"}), 400

    stopped_tasks = []
    not_found_tasks = []

    with process_lock:
        for camera_id in camera_ids:
            if camera_id in tasks_processes:
                print(f"Stopping detection for camera {camera_id}")  # Debugging print
                tasks_processes[camera_id]['stop_event'].set()  # Signal to stop the detection process
                tasks_processes[camera_id]['process'].join()  # Wait for the process to finish
                del tasks_processes[camera_id]  # Remove from the active tasks
                stopped_tasks.append(camera_id)
            else:
                not_found_tasks.append(camera_id)

    success = len(stopped_tasks) > 0
    response = {
        "success": success,
        "stopped": stopped_tasks,
        "not_found": not_found_tasks
    }

    print(f"Stop API Response: {response}")  # Debugging print
    return jsonify(response), 200

# Main entry point
if __name__ == '__main__':
    from waitress import serve

    serve(app, host='0.0.0.0', port=5000)