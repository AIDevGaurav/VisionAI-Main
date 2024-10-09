import cv2
import cvzone  # Make sure you have cvzone installed
import math
import time
from ultralytics import YOLO
import multiprocessing
from app.utils import capture_image, capture_video  # Assuming capture_image and capture_video are defined in utils
from app.mqtt_handler import publish_message_mqtt as pub  # Assuming you have an MQTT handler setup
from app.config import logger  # Assuming you have a logger setup in config
from app.exceptions import FallError

# Global dictionary to keep track of processes
tasks_processes = {}

def fall_detect(rtsp_url, camera_id, site_id, display_width, display_height, types, stop_event):
    # YOLO model (pre-trained on COCO dataset)
    model = YOLO('Model/yolov8l.pt')

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

    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        print(f"Error: Unable to open video stream for camera {camera_id}")
        return

    window_name = f'Fall Detection - Camera {camera_id}'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)  # Create a window that can be resized

    last_detection_time = 0
    threshold_confidence = 50  # Adjust the threshold confidence value as needed

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (display_width, display_height))

        # Run YOLO detection
        results = model(frame)

        for info in results:
            parameters = info.boxes
            for box in parameters:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                confidence = box.conf[0]
                class_detect = int(box.cls[0])
                class_detect = classnames[class_detect]
                conf = math.ceil(confidence * 100)

                height = y2 - y1
                width = x2 - x1
                threshold = height - width

                if conf > threshold_confidence and class_detect == 'person':
                    cvzone.cornerRect(frame, [x1, y1, width, height], l=30, rt=6)
                    cvzone.putTextRect(frame, f'{class_detect}', [x1 + 8, y1 - 12], thickness=2, scale=2)

                    current_time = time.time()
                    # Fall detection condition (if height < width)
                    if threshold < 0 and (current_time - last_detection_time > 10):
                        cvzone.putTextRect(frame, 'Fall Detected', [x1, y1], thickness=2, scale=2)

                        frame_copy = frame.copy()
                        image_filename = capture_image(frame_copy)
                        video_filename = "testing" # capture_video(rtsp_url)

                        message = {
                            "cameraId": camera_id,
                            "siteId": site_id,
                            "type": types,
                            "image": image_filename,
                            "video": video_filename
                        }
                        # Publish MQTT message
                        pub("fall/detection", message)
                        last_detection_time = current_time

        # Display the frame
        cv2.imshow(window_name, frame)

        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyWindow(window_name)

def fall_start(task, id1, typ, co):
    camera_id = task["cameraId"]
    try:
        site_id = task["siteId"]
        display_width = task["display_width"]
        display_height = task["display_height"]
        types = task["type"]
        rtsp_url = task["rtsp_link"]
        if camera_id not in tasks_processes:
            stop_event = multiprocessing.Event()
            tasks_processes[camera_id] = stop_event

            # Start motion detection in a new process
            process = multiprocessing.Process(
                target=fall_detect,
                args=(rtsp_url, camera_id, site_id, display_width, display_height, types, stop_event)
            )
            tasks_processes[camera_id] = process
            process.start()
            logger.info(f"Started fall detection for camera {camera_id}.")
        else:
            logger.warning(f"fall detection already running for camera {camera_id}.")
            return False

    except Exception as e:
        logger.error(f"Failed to start detection process for camera {camera_id}: {str(e)}", exc_info=True)
        return False
    return True


def fall_stop(camera_ids):
    """
    fire motion detection for the given camera IDs.
    """
    stopped_tasks = []
    not_found_tasks = []

    for camera_id in camera_ids:
        if camera_id in tasks_processes:
            try:
                tasks_processes[camera_id].terminate()  # Stop the process
                tasks_processes[camera_id].join()  # Wait for the process to stop
                del tasks_processes[camera_id]  # Remove from the dictionary
                stopped_tasks.append(camera_id)
                logger.info(f"Stopped fall detection for camera {camera_id}.")
            except Exception as e:
                logger.error(f"Failed to fall detection for camera {camera_id}: {str(e)}", exc_info=True)
        else:
            not_found_tasks.append(camera_id)

    return {
        "success": len(stopped_tasks) > 0,
        "stopped": stopped_tasks,
        "not_found": not_found_tasks
    }
