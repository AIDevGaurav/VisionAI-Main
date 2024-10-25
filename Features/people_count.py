import queue
import threading
import time
import cv2
import numpy as np
from ultralytics import YOLO
from app.exceptions import PCError
from app.config import logger, global_thread, queues_dict, executor
from app.mqtt_handler import publish_message_mqtt as pub
from app.utils import capture_image, start_feature_processing
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2

# Function to adjust ROI points based on provided coordinates
def set_roi_based_on_points(points, coordinates):
    x_offset, y_offset = coordinates["x"], coordinates["y"]
    return [(int(x + x_offset), int(y + y_offset)) for x, y in points]


def capture_and_publish(frame, c_id, s_id, typ, count):
    try:
        image_path = capture_image(frame)  # Assuming this function saves the image and returns the path
        message = {
            "cameraId": c_id,
            "siteId": s_id,
            "type": typ,
            "image": image_path,
            "people_count": count
        }
        pub("people/detection", message)
        logger.info(f"Published people message for camera {c_id}.")
    except Exception as e:
        logger.error(f"Error capturing image or publishing MQTT for camera {c_id}: {str(e)}")

# Define function to load TensorRT engine
def load_engine(engine_path):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

# Initialize the engine
engine = load_engine('Model/yolov8l.engine')

# Create context and allocate buffers
context = engine.create_execution_context()

def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()

    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))

        # Allocate device memory
        buffer = cuda.mem_alloc(size * dtype.itemsize)

        # Append the buffer to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(buffer)
        else:
            outputs.append(buffer)

        bindings.append(int(buffer))
    return inputs, outputs, bindings, stream

# Allocate buffers once outside of the function to avoid reallocation on each frame
inputs, outputs, bindings, stream = allocate_buffers(engine)

def people_count(camera_id, s_id, typ, coordinates, width, height, stop_event):
    try:
        if coordinates and "points" in coordinates and coordinates["points"]:
            roi_points = np.array(set_roi_based_on_points(coordinates["points"], coordinates), dtype=np.int32)
            roi_mask = np.zeros((height, width), dtype=np.uint8)
            cv2.fillPoly(roi_mask, [roi_points], 255)  # Fill mask for the static ROI
        else:
            roi_mask = None

        previous_people_count = 0
        frame_counter = 0

        while not stop_event.is_set():
            frame = queues_dict[f"{camera_id}_{typ}"].get(timeout=10)
            if frame is None:
                continue

            frame_counter += 1
            if frame_counter % 5 == 0:
                queues_dict[f"{camera_id}_{typ}"].task_done()
                continue

            masked_frame = cv2.bitwise_and(frame, frame, mask=roi_mask) if roi_mask is not None else frame

            # Preprocess frame (resize, normalization, etc.)
            input_frame = cv2.resize(masked_frame, (640, 640))  # Resize to model input size
            input_frame = input_frame.transpose(2, 0, 1).astype(np.float32)  # HWC to CHW
            input_frame /= 255.0  # Normalize to [0, 1]

            # Transfer the input data to the GPU
            cuda.memcpy_htod_async(inputs[0], input_frame, stream)

            # Run inference
            context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)

            # Transfer the output data from the GPU
            output = np.empty((1, 25200, 85), dtype=np.float32)  # Adjust the output shape as per your model
            cuda.memcpy_dtoh_async(output, outputs[0], stream)
            stream.synchronize()

            # Postprocess output (detect people based on class 0)
            count = 0
            for detection in output[0]:
                class_id = int(detection[5])
                if class_id == 0:  # Class 0 is typically 'person'
                    count += 1

            if previous_people_count != count:
                executor.submit(capture_and_publish, frame, camera_id, s_id, typ, count)
                previous_people_count = count

            queues_dict[f"{camera_id}_{typ}"].task_done()

    except Exception as e:
        logger.error(f"Error During People Count: {str(e)}")
        return PCError(f"People Count Failed for camera: {camera_id}")



def start_pc(c_id, s_id, typ, co, width, height, rtsp):
    """
    Start the motion detection process in a separate thread for the given camera task.
    """
    try:
        if f"{c_id}_{typ}_detect" in global_thread:
            stop_pc(c_id, typ)
            time.sleep(2)

        executor.submit(start_feature_processing, c_id, typ, rtsp, width, height)
        stop_event = threading.Event()  # Create a stop event for each feature
        global_thread[f"{c_id}_{typ}_detect"] = stop_event
        executor.submit(people_count, c_id, s_id, typ, co, width, height, stop_event)

        logger.info(f"Started people detection for camera {c_id}.")

    except Exception as e:
        logger.error(f"Failed to start detection process for camera {c_id}: {str(e)}", exc_info=True)
        return False
    return True

def stop_pc(camera_id, typ):
    stopped_tasks = []
    not_found_tasks = []

    key = f"{camera_id}_{typ}"  # Construct the key as used in the dictionary
    key2 = f"{camera_id}_{typ}_detect"

    try:
        if key in global_thread and key in queues_dict and key2 in global_thread:
            stop_event_detect = global_thread[key]  # Retrieve the stop event from the dictionary
            stop_event_detect.set()  # Signal the thread to stop
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