import queue
import threading
from app.mqtt_handler import publish_message_mqtt as pub
import cv2
import time
import os
from app.config import logger, queues_dict, global_thread, executor
from app.exceptions import FrameError

image_dir = "images"
video_dir = "videos"
os.makedirs(image_dir, exist_ok=True)
os.makedirs(video_dir, exist_ok=True)

def capture_image(frame):
    try:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        image_filename = os.path.join(image_dir, f"Images_{timestamp}.jpg")
        cv2.imwrite(image_filename, frame)
        logger.info(f"Image captured and saved to {image_filename}")
        return os.path.abspath(image_filename)

    except Exception as e:
        logger.error(f"Error capturing image: {e}", exc_info=True)
        raise

def capture_video(rtsp_url):
    try:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        video_filename = os.path.join(video_dir, f"Videos_{timestamp}.mp4")

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
        logger.info(f"Video captured and saved to {video_filename}")
        return os.path.abspath(video_filename)

    except Exception as e:
        logger.error(f"Error capturing video: {e}", exc_info=True)
        raise


def capture_frame(rtsp, c_id, typ, w, h, stop_event):
    """Capture frames from the RTSP stream and put them into the buffer."""
    cap = cv2.VideoCapture(rtsp)

    if not cap.isOpened():
        logger.error(f"Error opening video stream: {rtsp}")
        return

    # logger.info(f"Started capturing from {rtsp} for {c_id}_{typ}")

    try:
        while not stop_event.is_set():
            ret, frame = cap.read()

            if not ret:
                logger.error(f"Failed to capture frame from {rtsp}")
                break

            # Resize the frame to the required width and height
            frame = cv2.resize(frame, (w, h))

            # Check if the queue is full before adding
            key = f"{c_id}_{typ}"
            if key not in queues_dict:
                logger.error(f"Queue for {key} does not exist.")
                break

            if not queues_dict[key].full():
                queues_dict[key].put(frame)
                # logger.info(f"Frame added to queue: {key}. Queue size: {queues_dict[key].qsize()}")
            else:
                logger.warning(f"Frame queue for {key} is full, skipping frame.")
    except Exception as e:
        logger.exception(f"Error occurred while capturing frames: {e}")
    finally:
        cap.release()
        queues_dict.pop(f"{c_id}_{typ}")
        logger.info(f"Released video capture for {rtsp}")


def start_feature_processing(c_id, typ, rtsp, width, height):
    """
    Start processing for a specific camera and feature type.
    """
    key = f"{c_id}_{typ}"

    # Initialize the queue if it doesn't exist
    if key not in queues_dict:
        queues_dict[key] = queue.Queue(maxsize=500)  # You can adjust maxsize as needed
        logger.info(f"Queue created for {key} with maxsize {queues_dict[key].maxsize}")

    # Initialize the stop event for controlling the thread
    if key not in global_thread:
        stop_event = threading.Event()
        global_thread[key] = stop_event

    # Submit the frame capture task to the executor
    executor.submit(capture_frame, rtsp, c_id, typ, width, height, global_thread[key])
    logger.info(f"Task submitted to executor for {key}")

