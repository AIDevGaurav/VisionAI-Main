import cv2
from ultralytics import YOLO
import multiprocessing
import time
from app.utils import capture_image, capture_video  # Assuming capture_image and capture_video are defined in utils
from app.mqtt_handler import publish_message_mqtt as pub  # Assuming you have an MQTT handler setup
from app.config import logger  # Assuming you have a logger setup in config
from app.exceptions import FireError


tasks_processes = {}  # Dictionary to keep track of running processes

def detect_fire(rtsp_url, camera_id, site_id, coordinates, type ,display_width, display_height, stop_event):
    """
    Fire detection function that captures video frames, performs inference,
    and publishes a message if fire is detected.
    """
    try:
        cap = cv2.VideoCapture(rtsp_url)

        if not cap.isOpened():
            raise FireError(f"Error: Unable to open RTSP stream at {rtsp_url}")

        # Load the YOLO model for fire detection
        model_path = 'Model/fire.pt'  # Replace with the path to your YOLOv8 fire.pt model
        model = YOLO(model_path)

        last_detection_time = 0

        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                raise FireError(f"Error: Unable to open RTSP stream at {rtsp_url}")
                break

            # Get display width and height from coordinates
            display_width = display_width
            display_height = display_height

            # Resize the frame to match display size
            frame = cv2.resize(frame, (display_width, display_height))

            # Perform inference on the frame
            results = model(frame)

            # YOLO returns a list of results, extract the first one
            result = results[0]

            # Annotate detected objects with class names and bounding boxes
            fire_detected = False
            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence = box.conf[0]
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Get the class name from the model's class names
                class_name = model.names[class_id]
                logger.info(f"Detected class: {class_name}")

                # Only trigger for "fire"
                if class_name.lower() == "fire":
                    fire_detected = True
                    # Draw bounding box on the frame
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(frame, f'{class_name}',
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            # If fire is detected, publish a message
            current_time = time.time()
            if fire_detected and (current_time - last_detection_time > 60):
                logger.info(f"Fire detected for camera {camera_id}. Capturing image and video.")

                frame_copy = frame.copy()
                image_path = capture_image(frame_copy)  # Save an image when fire is detected
                video_path = "testing" # capture_video(rtsp_url)  # Capture video from the stream

                try:
                    # Publish to MQTT
                    pub_message = {
                        "rtsp_link": rtsp_url,
                        "cameraId": camera_id,
                        "siteId": site_id,
                        "type": type,
                        "image": image_path,
                        "video": video_path
                    }
                    pub("fire/detection", pub_message)
                    logger.info(f"Published fire detection message: {pub_message}")
                except Exception as e:
                    logger.error(f"Error publishing MQTT message: {e}", exc_info=True)
                    raise

                last_detection_time = current_time

            # Display the frame
            cv2.imshow('Fire Detection (RTSP)', frame)

            # Break if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    except Exception as e:
        logger.error(f"Error during Fire detection for camera {camera_id}: {str(e)}", exc_info=True)
        raise FireError(f"Fire detection failed for camera {camera_id}: {str(e)}")

def fire_start(task):
    """
    Start the fire detection process.
    """
    try:
        camera_id=task["cameraId"]
        display_width = task["display_width"]
        display_height = task["display_height"]
        if camera_id not in tasks_processes:
            stop_event = multiprocessing.Event()
            tasks_processes[camera_id]=stop_event
            # start fire detection
            process = multiprocessing.Process(
                target=detect_fire,
                args=(task["rtsp_link"], camera_id, task["siteId"], task["co_ordinates"], task["type"], display_width, display_height, stop_event)
            )
            tasks_processes[camera_id] = process
            process.start()
            logger.info(f"Started Fire detection for camera {camera_id}.")
        else:
            logger.warning(f"Fire detection already running for camera {camera_id}.")
            return False
    except Exception as e:
        logger.error(f"Failed to start detection process for camera {camera_id}: {str(e)}", exc_info=True)
        return False
    return True

def fire_stop(camera_ids):
    """
    Stop Fire detection for the given camera IDs.
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
                logger.info(f"Stopped Fire detection for camera {camera_id}.")
            except Exception as e:
                logger.error(f"Failed to stop Fire detection for camera {camera_id}: {str(e)}", exc_info=True)
        else:
            not_found_tasks.append(camera_id)

    return {
        "success": len(stopped_tasks) > 0,
        "stopped": stopped_tasks,
        "not_found": not_found_tasks
    }
