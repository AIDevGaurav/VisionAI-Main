import queue
import threading
import time
from collections import deque
import cv2
import numpy as np
from app.exceptions import PCError
from app.config import logger, global_thread, queues_dict, get_executor, YOLOv8fire, YOLOv8Single
from app.mqtt_handler import publish_message_mqtt as pub
from app.utils import capture_image, start_feature_processing
from ultralytics import YOLO


executor = get_executor()

# Define parameters for fire detection
FIRE_CONFIDENCE_THRESHOLD = 0.5
MIN_FIRE_SIZE = 100  # Minimum area of fire bounding box
PERSISTENCE_THRESHOLD = 5  # Number of consecutive frames to confirm fire
COLOR_THRESHOLD = 200  # Threshold for red channel intensity


# Function to adjust ROI points based on provided coordinates
def set_roi_based_on_points(points, coordinates):
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

        }
        pub("fire/detection", message)
        logger.info(f"Published fire message for camera {c_id}.")
    except Exception as e:
        logger.error(f"Error capturing image or publishing MQTT for camera {c_id}: {str(e)}")

def create_mask(frame, boxes, padding=10):
    mask = np.ones(frame.shape[:2], dtype=np.uint8) * 255
    for box in boxes:
        x1, y1, x2, y2 = map(int, box[:4])
        cv2.rectangle(mask, (max(0, x1 - padding), max(0, y1 - padding)),
                      (min(frame.shape[1], x2 + padding), min(frame.shape[0], y2 + padding)), 0, -1)
    return mask

def check_color(frame, box):
    x1, y1, x2, y2 = map(int, box)
    roi = frame[y1:y2, x1:x2]
    return np.mean(roi[:, :, 2]) > COLOR_THRESHOLD  # Check if average red intensity is high


def detect_fire(camera_id, s_id, typ, coordinates, width, height, stop_event):
    """
    Fire detection function that captures video frames, performs inference,
    and publishes a message if fire is detected.
    """
    try:

        fire_model = YOLOv8fire()
        device_model = YOLOv8Single()


        # Define the devices we want to detect
        DEVICES = ["cell phone", "laptop", "tv"]

        if coordinates and "points" in coordinates and coordinates["points"]:
            roi_points = np.array(set_roi_based_on_points(coordinates["points"], coordinates), dtype=np.int32)
            roi_mask = np.zeros((height, width), dtype=np.uint8)
            cv2.fillPoly(roi_mask, [roi_points], 255)  # Fill mask for the static ROI
            logger.info(f"ROI set for fire detection on camera {camera_id}")
        else:
            roi_mask = None

        fire_persistence = deque(maxlen=PERSISTENCE_THRESHOLD)

        while not stop_event.is_set():
            # start_time = time.time()
            frame = queues_dict[f"{camera_id}_{typ}"].get(
                timeout=10)  # Handle timeouts if frame retrieval takes too long
            if frame is None:
                continue

            # # Log the queue size
            # queue_size = queues_dict[f"{camera_id}_{typ}"].qsize()
            # logger.info(f"people: {queue_size}")

            if roi_mask is not None:
                masked_frame = cv2.bitwise_and(frame, frame, mask=roi_mask)
                cv2.polylines(frame, [roi_points], isClosed=True, color=(255, 0, 0), thickness=2)
            else:
                masked_frame = frame

            # Detect devices first
            device_results = device_model(masked_frame)
            device_boxes = []
            for result in device_results:
                for box in result.boxes:
                    class_name = device_model.names[int(box.cls[0])]
                    if class_name in DEVICES:
                        device_boxes.append((box.xyxy[0].cpu().numpy(), class_name))

                    # Draw device boxes
                    for box, class_name in device_boxes:
                        x1, y1, x2, y2 = map(int, box)
                        cv2.rectangle(masked_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(masked_frame, class_name, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Create a mask to exclude device areas
            mask = create_mask(masked_frame, [box for box, _ in device_boxes])

            # Apply the mask to the frame
            fire_frame = cv2.bitwise_and(masked_frame, masked_frame, mask=mask)

            # Detect fires in the masked frame
            fire_results = fire_model(fire_frame)

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
                        if not check_color(fire_frame, (x1, y1, x2, y2)):
                            continue

                        current_fire_detected = True
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(frame, f'Fire ', (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            # Update fire persistence
            fire_persistence.append(current_fire_detected)

            # Check if fire has been consistently detected
            fire_detected = all(fire_persistence) if len(fire_persistence) == PERSISTENCE_THRESHOLD else False

            # Print detection status
            if fire_detected:
                print("Fire detected!")
                executor.submit(capture_and_publish, fire_frame, camera_id, s_id, typ)
            for _, class_name in device_boxes:
                print(f"{class_name} detected")

            cv2.imshow(f"Camera {camera_id}_{typ}", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        logger.error(f"Error During Fire detection:{str(e)}")
        return PCError(f"Fire detection Failed for camera : {camera_id}")

    # finally:
    #     cv2.destroyWindow(f'People Count - Camera {camera_id}')


def fire_start(c_id, s_id, typ, co, width, height, rtsp):
    """
    Start the fire detection process in a separate thread for the given camera task.
    """
    try:
        if f"{c_id}_{typ}_detect" in global_thread:
            fire_stop(c_id, typ)
        executor.submit(start_feature_processing, c_id, typ, rtsp, width, height)
        stop_event = threading.Event()  # Create a stop event for each feature
        global_thread[f"{c_id}_{typ}_detect"] = stop_event
        executor.submit(detect_fire, c_id, s_id, typ, co, width, height, stop_event)

        logger.info(f"Started motion detection for camera {c_id}.")

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
            stop_event = global_thread[key]  # Retrieve the stop event from the dictionary
            stop_event.set()  # Signal the thread to stop
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

