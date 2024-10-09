import queue
import threading
from app.mqtt_handler import publish_message_mqtt as pub
import cv2
import time
import os
import psutil
from app.config import logger, queues_dict, global_thread, get_executor
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


# def adjust_queue_size():
#     while True:
#         memory = psutil.virtual_memory()
#         # Example: Adjust queue size based on available memory
#         if memory.available < 500 * 1024 * 1024:  # Less than 500MB available
#             frame_queue.maxsize = max(2, frame_queue.qsize() // 2)  # Reduce size, but ensure it is not less than 2
#         else:
#             frame_queue.maxsize = min(100, frame_queue.qsize() * 2)  # Increase size, max out at 100
#
#         time.sleep(10)  # Check every 10 seconds


def capture_frame(rtsp, c_id, typ, w, h, stop_event):
    """Capture frames from the RTSP stream and put them into the buffer."""
    cap = cv2.VideoCapture(rtsp)
    if not cap.isOpened():
        print("Error opening video stream")
        return

    try:
        while not stop_event.is_set():
            ret, frame = cap.read()
            frame = cv2.resize(frame, (w, h))
            if not ret:
                logger.error("Failed to capture frame")
                break
            if not queues_dict[f"{c_id}_{typ}"].full():
                queues_dict[f"{c_id}_{typ}"].put(frame)
            else:
                logger.warning("Frame queue is full; adjusting capture settings.")
                # Additional logic to handle full queue
    finally:
        cap.release()


def start_feature_processing(c_id, typ, rtsp, width, height):
    """
    Start processing for a specific camera and feature type.

    :param c_id: Camera ID
    :param typ: Type of feature to process (e.g., 'motion_detection')
    :param rtsp: RTSP stream URL
    :param width: Width of the frames to process
    :param height: Height of the frames to process
    """
    stop_event = None
    key = f"{c_id}_{typ}"
    executor = get_executor()
    # Check if a queue already exists for this camera and type
    if key not in queues_dict:
        queues_dict[key] = queue.Queue()
        # logger.info(f"Queue created for {key}")

    if key not in global_thread:
        stop_event = threading.Event()
        global_thread[key] = stop_event

    # Submit the task to the executor
    executor.submit(capture_frame, rtsp, c_id, typ, width, height, stop_event)
    # logger.info(f"Task submitted to executor for {key}")

