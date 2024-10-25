import os
import logging
import psutil
from concurrent.futures import ThreadPoolExecutor

#Dictionary to hold executor thread
global_thread = {}

# Dictionary to hold the queues for each camera and feature
queues_dict = {}

# Logging Configuration
os.makedirs('logs', exist_ok=True)

LOG_FILE = os.path.join('logs', 'app.log')

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.CRITICAL,
    format='%(asctime)s %(levelname)s %(name)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger('app')

class Executor:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            max_workers = len([x for x in psutil.cpu_percent(percpu=True) if x < 75])  # For available cores
            logger.info(f"Max Worker Available are {max_workers}")
            cls._instance = ThreadPoolExecutor(max_workers= max_workers)
        return cls._instance

executor = Executor()

# class YOLOv8Single:
#     def __init__(self):
#         logging.info("Loading YOLOv8 model...")
#         self.model = YOLO('Model/yolov8l.pt')
#         logging.info("Model loaded successfully.")
#
#
# class YOLOv8fire:
#     def __init__(self):
#         logging.info("Loading YOLOv8 model...")
#         self.model = YOLO('Model/fire.pt')
#         logging.info("Model loaded successfully.")
#
# class YOLOv8armed:
#     def __init__(self):
#         logging.info("Loading YOLOv8 model...")
#         self.model = YOLO('Model/armed.pt')
#         logging.info("Model loaded successfully.")
#
# class YOLOv8pose:
#     def __init__(self):
#         logging.info("Loading YOLOv8 model...")
#         self.model = YOLO('Model/yolov8l-pose.pt')
#         logging.info("Model loaded successfully.")

