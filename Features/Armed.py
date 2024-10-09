import cv2
import multiprocessing
import time
import json
from ultralytics import YOLO
from app.utils import capture_image, capture_video
from app.mqtt_handler import publish_message_mqtt as pub
from app.config import logger
from app.exceptions import ArmError



# Global dictionary to keep track of processes
tasks_processes = {}

def detect_armed_person(rtsp_url, camera_id, site_id, display_width, display_height, type, co_ordinate, stop_event):
    """
    :tasks: Detect armed persons in the stream using YOLOv8.
    :return: Capture Image, Video and Publish Mqtt message
    """
    try:
        # Load your trained YOLOv8 model
        model = YOLO('Model/armed.pt')

        # Armed person class indices (ID 0 for 'gun' and ID 1 for 'person with a gun')
        armed_classes = [0, 1]  # Class IDs for armed persons (adjust based on model)

        # Class names in your custom dataset
        classnames = ['gun', 'person with a gun', 'person']  # Adjust as per your dataset

        cap = cv2.VideoCapture(rtsp_url)
        if not cap.isOpened():
            raise ArmError(f"Failed to open Camera: {camera_id}")

        last_detection_time = 0
        detection_delay = 10  # Time (in seconds) between consecutive detections

        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                logger.warning(f"Camera failed id: {camera_id}")
                raise ArmError("Armed Cap Error")

            frame = cv2.resize(frame, (display_width, display_height))

            # Run YOLO detection with confidence and NMS thresholds
            results = model(frame, conf=0.5, iou=0.5)

            for info in results:
                parameters = info.boxes
                for box in parameters:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    class_detect = int(box.cls[0])

                    current_time = time.time()

                    if class_detect in armed_classes:
                        if current_time - last_detection_time > detection_delay:
                            class_name = classnames[class_detect]

                            # Draw bounding box and label
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                            label = f'{class_name}'  # Include confidence score in the label
                            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

                            frame_copy = frame.copy()
                            image_filename = capture_image(frame_copy)
                            video_filename = "testing"  # Replace with actual video capture function

                            # Publish MQTT message
                            message = {
                                "cameraId": camera_id,
                                "class": class_name,
                                "siteId": site_id,
                                "type": type,
                                "image": image_filename,
                                "video": video_filename
                            }
                            pub("arm/detection", message)
                            last_detection_time = current_time

            # Display the frame
            cv2.imshow(f'Armed Person Detection - Camera {camera_id}', frame)

            # Break loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyWindow(f'Armed Person Detection - Camera {camera_id}')

    except Exception as e:
        logger.error(f"Error for camera:{camera_id} in Armed Detection: {str(e)}")
        raise ArmError(f"Error in Armed Detection for camera id {camera_id}")


def armed_detection_start(task, id1, typ, co):
    """
    :param task: Json Array
    tasks: Format the input data and start armed detection with multiprocessing for multiple cameras
    :return: True or false
    """
    try:
        camera_id = task["cameraId"]
        site_id = task["siteId"]
        display_width = task["display_width"]
        display_height = task["display_height"]
        types = task["type"]
        rtsp_url = task["rtsp_link"]
        co_ordinate = task["co_ordinates"]
        if camera_id not in tasks_processes:
            stop_event = multiprocessing.Event()
            tasks_processes[camera_id] = stop_event

            # Start detection in a new process
            process = multiprocessing.Process(target=detect_armed_person, args=(
                rtsp_url, camera_id, site_id, display_width, display_height, types, co_ordinate, stop_event))
            tasks_processes[camera_id] = process
            process.start()
            logger.info(f"Started Armed Detection for camera {camera_id}.")
        else:
            logger.warning(f"Armed Detection already running for camera {camera_id}.")
            return False
    except Exception as e:
        logger.error(f"Failed to start detection process for camera {camera_id}: {str(e)}", exc_info=True)
        return False
    return True

def armed_detection_stop(camera_ids):
    """
    Stop armed detection for the given camera IDs.
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
                logger.info(f"Stopped Armed Detection for camera {camera_id}.")
            except Exception as e:
                logger.error(f"Failed to stop Armed Detection for camera {camera_id}: {str(e)}", exc_info=True)
        else:
            not_found_tasks.append(camera_id)

    return {
        "success": len(stopped_tasks) > 0,
        "stopped": stopped_tasks,
        "not_found": not_found_tasks
    }
