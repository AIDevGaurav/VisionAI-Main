import multiprocessing
import cv2
import math
import numpy as np
from ultralytics import YOLO
from Features.sort import Sort
from app.utils import capture_image
from app.mqtt_handler import publish_message_mqtt as pub
from app.config import logger
from app.exceptions import ZipError


tasks_processes = {}  # Dictionary to keep track of running processes


# Apply the given coordinates to ROI points
def use_coordinates(points, coordinates):
    x_offset = coordinates["x"]
    y_offset = coordinates["y"]

    placed_points = []
    for point in points:
        placed_x = int(point[0] + x_offset)
        placed_y = int(point[1] + y_offset)
        placed_points.append((placed_x, placed_y))

    return placed_points


# Check if the centroid of a person is inside the ROI polygon
def is_inside_roi(cx, cy, roi_points):
    result = cv2.pointPolygonTest(np.array(roi_points, np.int32), (cx, cy), False)
    return result >= 0


def detect_people_count(rtsp_url, site_id, camera_id, alarm_id, roi_coords, zipline_coords, arrow_coords, display_size, type, stop_event):
    # Initialize YOLO model and SORT tracker
    model = YOLO("Model/yolov8l.pt")
    tracker = Sort(max_age=20, min_hits=4, iou_threshold=0.4)

    try:
        cap = cv2.VideoCapture(rtsp_url)
        count = 0  # Single count variable for both up and down

        # Dictionary to store the previous Y position (cy) of each person to detect crossings
        previous_positions = {}

        display_width, display_height = display_size

        # Define a buffer zone around the zipline
        buffer_vertical = 60  # Vertical buffer around the line
        buffer_horizontal = 60  # Horizontal buffer around the line

        # Arrow start and end points based on arrow coordinates
        arrow_start = (int(arrow_coords['x']), int(arrow_coords['y']))
        arrow_end = (
        arrow_start[0] + int(arrow_coords['points'][1][0]), arrow_start[1] + int(arrow_coords['points'][1][1]))

        # Determine the direction of the arrow (up or down)
        arrow_direction = arrow_coords['points'][1][1]  # Positive for down, negative for up

        # Create a window for each camera stream
        window_name = f"Camera {camera_id}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        # Set the window size to the display size provided
        cv2.resizeWindow(window_name, display_width, display_height)

        # Error handling for the camera stream
        if not cap.isOpened():
            print(f"Error: Unable to open camera stream for camera {camera_id}")
            return

        while not stop_event.is_set():
            success, frame = cap.read()
            if not success:
                print(f"Error: Failed to read from camera {camera_id}")
                break

            # Resize the frame to match the provided display size
            frame = cv2.resize(frame, (display_width, display_height))

            # Draw zipline and ROI on the frame using the coordinates directly from the JSON
            cv2.line(frame, zipline_coords[0], zipline_coords[1], (0, 0, 255), 5)  # Draw the zipline
            cv2.polylines(frame, [np.array(roi_coords, np.int32)], True, (255, 0, 0), 2)  # Draw the ROI

            # Draw the arrow on the frame
            cv2.arrowedLine(frame, arrow_start, arrow_end, (0, 255, 255), 3, tipLength=0.5)  # Draw the arrow

            # Display the frame with the visual indicators (arrow, zipline, ROI)
            cv2.imshow(window_name, frame)

            # Create an ROI mask
            roi_mask = np.zeros_like(frame[:, :, 0])  # Same size as one channel of the frame
            cv2.fillPoly(roi_mask, [np.array(roi_coords, np.int32)], 255)

            # Apply the ROI mask to the frame
            roi_frame = cv2.bitwise_and(frame, frame, mask=roi_mask)

            # Run YOLO model on the cropped frame (inside the ROI)
            results = model(roi_frame, stream=True)
            detections = np.empty((0, 5))

            # Start tracking only when people are detected inside the ROI
            tracking_started = False

            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1b, y1b, x2b, y2b = box.xyxy[0]
                    x1b, y1b, x2b, y2b = int(x1b), int(y1b), int(x2b), int(y2b)

                    # Confidence and class filtering (person class)
                    conf = math.ceil((box.conf[0] * 100)) / 100
                    cls = int(box.cls[0])
                    if cls == 0 and conf > 0.3:  # Class 0 corresponds to 'person'
                        detections = np.vstack((detections, [x1b, y1b, x2b, y2b, conf]))
                        tracking_started = True  # Begin tracking when a person is detected inside the ROI

            # Proceed to tracking only if a person is detected inside the ROI
            if tracking_started:
                tracked_people = tracker.update(detections)

                # Get the slope (m) and intercept (b) of the zipline
                m = (zipline_coords[1][1] - zipline_coords[0][1]) / (zipline_coords[1][0] - zipline_coords[0][0])
                b = zipline_coords[0][1] - m * zipline_coords[0][0]

                for person in tracked_people:
                    x1, y1, x2, y2, person_id = person
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    # Calculate the centroid of the bounding box
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2  # Center point of the bounding box

                    # Calculate the y-coordinate of the zipline at the person's x-position (cx)
                    zipline_y = m * cx + b

                    # Get the previous y-coordinate of the person (if any)
                    prev_cy = previous_positions.get(person_id, cy)

                    # Restrict detection to objects within the buffer zone of the zipline
                    if (abs(cy - zipline_y) <= buffer_vertical) and (
                            zipline_coords[0][0] - buffer_horizontal <= cx <= zipline_coords[1][0] + buffer_horizontal):
                        # If the arrow points up (negative value), only count upward crossings
                        if arrow_direction < 0 and cy < zipline_y and prev_cy >= zipline_y:
                            count += 1
                            frame_copy = frame.copy()
                            image_filename = capture_image(frame_copy)
                            message = {
                                "rtsp_link": rtsp_url,
                                "siteId": site_id,
                                "cameraId": camera_id,
                                "alarmId": alarm_id,
                                "type": type,
                                "people_count": count,
                                "image": image_filename
                            }
                            pub("zipline/detection", message)
                            cv2.line(frame, zipline_coords[0], zipline_coords[1], (0, 255, 0),
                                     5)  # Change line color to green for up

                        # If the arrow points down (positive value), only count downward crossings
                        elif arrow_direction > 0 and cy > zipline_y and prev_cy <= zipline_y:
                            count += 1
                            frame_copy = frame.copy()
                            image_filename = capture_image(frame_copy)
                            message = {
                                "rtsp_link": rtsp_url,
                                "siteId": site_id,
                                "cameraId": camera_id,
                                "alarmId": alarm_id,
                                "type": type,
                                "people_count": count,
                                "image": image_filename
                            }
                            pub("zipline/detection", message)
                            cv2.line(frame, zipline_coords[0], zipline_coords[1], (0, 0, 255),
                                     5)  # Keep line color red for down

                    # Update the previous position of the person
                    previous_positions[person_id] = cy

                    # Draw bounding box and centroid on the frame
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(frame, (cx, cy), 5, (0, 0, 255), cv2.FILLED)

                # Show people count on the frame
                cv2.putText(frame, f"Count: {count}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Show the updated frame in the window for the camera
                cv2.imshow(window_name, frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_event.set()
                break

        cap.release()
        cv2.destroyAllWindows()

        return count
        # cap = cv2.VideoCapture(rtsp_url)
        # count = 0  # Count variable for both up and down
        #
        # previous_positions = {}
        # debounce_time = 1.5  # seconds
        # last_count_time = {}
        #
        # display_width, display_height = display_size
        # buffer_vertical = 20  # Vertical buffer around the line
        # buffer_horizontal = 20  # Horizontal buffer around the line
        #
        # arrow_start = (int(arrow_coords['x']), int(arrow_coords['y']))
        # arrow_end = (arrow_start[0] + int(arrow_coords['points'][1][0]), arrow_start[1] + int(arrow_coords['points'][1][1]))
        # arrow_direction = arrow_coords['points'][1][1]  # Positive for down, negative for up
        #
        # window_name = f"Camera {camera_id}"
        # cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        # cv2.resizeWindow(window_name, display_width, display_height)
        #
        # if not cap.isOpened():
        #     print(f"Error: Unable to open camera stream for camera {camera_id}")
        #     return
        #
        # while not stop_event.is_set():
        #     success, frame = cap.read()
        #     if not success:
        #         print(f"Error: Failed to read from camera {camera_id}")
        #         break
        #
        #     frame = cv2.resize(frame, (display_width, display_height))
        #     cv2.line(frame, zipline_coords[0], zipline_coords[1], (0, 0, 255), 5)  # Draw the zipline
        #     cv2.polylines(frame, [np.array(roi_coords, np.int32)], True, (255, 0, 0), 2)  # Draw the ROI
        #     cv2.arrowedLine(frame, arrow_start, arrow_end, (0, 255, 255), 3, tipLength=0.5)  # Draw the arrow
        #     cv2.imshow(window_name, frame)
        #
        #     # Process every frame without skipping
        #     roi_mask = np.zeros_like(frame[:, :, 0])
        #     cv2.fillPoly(roi_mask, [np.array(roi_coords, np.int32)], 255)
        #     roi_frame = cv2.bitwise_and(frame, frame, mask=roi_mask)
        #
        #     results = model(roi_frame, stream=True)
        #     detections = np.empty((0, 5))
        #     tracking_started = False
        #
        #     for r in results:
        #         boxes = r.boxes
        #         for box in boxes:
        #             x1b, y1b, x2b, y2b = map(int, box.xyxy[0])
        #             conf = math.ceil((box.conf[0] * 100)) / 100
        #             cls = int(box.cls[0])
        #             if cls == 0 and conf > 0.3:  # Class 0 corresponds to 'person'
        #                 detections = np.vstack((detections, [x1b, y1b, x2b, y2b, conf]))
        #                 tracking_started = True
        #
        #     if tracking_started:
        #         tracked_people = tracker.update(detections)
        #         m = (zipline_coords[1][1] - zipline_coords[0][1]) / (zipline_coords[1][0] - zipline_coords[0][0])
        #         b = zipline_coords[0][1] - m * zipline_coords[0][0]
        #
        #         current_time = cv2.getTickCount() / cv2.getTickFrequency()
        #
        #         for person in tracked_people:
        #             x1, y1, x2, y2, person_id = map(int, person)
        #             cx, cy = (x1 + x2) // 2, (y1 + y2) // 2  # Center point of the bounding box
        #
        #             if not is_inside_roi(cx, cy, roi_coords):
        #                 continue
        #
        #             zipline_y = m * cx + b
        #             prev_cy = previous_positions.get(person_id, cy)
        #
        #             if (abs(cy - zipline_y) <= buffer_vertical) and (zipline_coords[0][0] - buffer_horizontal <= cx <= zipline_coords[1][0] + buffer_horizontal):
        #                 if person_id not in last_count_time or (current_time - last_count_time[person_id]) > debounce_time:
        #                     if arrow_direction < 0 and cy < zipline_y and prev_cy >= zipline_y:
        #                         count += 1
        #                         last_count_time[person_id] = current_time
        #                         image_filename = capture_image(frame)
        #                         message = {
        #                             "rtsp_link": rtsp_url,
        #                             "siteId": site_id,
        #                             "cameraId": camera_id,
        #                             "alarmId": alarm_id,
        #                             "type": type,
        #                             "people_count": count,
        #                             "image": image_filename
        #                         }
        #                         pub("zipline/detection", message)
        #                         cv2.line(frame, zipline_coords[0], zipline_coords[1], (0, 255, 0), 5)
        #                     elif arrow_direction > 0 and cy > zipline_y and prev_cy <= zipline_y:
        #                         count += 1
        #                         last_count_time[person_id] = current_time
        #                         image_filename = capture_image(frame)
        #                         message = {
        #                             "rtsp_link": rtsp_url,
        #                             "siteId": site_id,
        #                             "cameraId": camera_id,
        #                             "alarmId": alarm_id,
        #                             "type": type,
        #                             "people_count": count,
        #                             "image": image_filename
        #                         }
        #                         pub("zipline/detection", message)
        #                         cv2.line(frame, zipline_coords[0], zipline_coords[1], (0, 0, 255), 5)
        #
        #             previous_positions[person_id] = cy
        #             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        #             cv2.circle(frame, (cx, cy), 5, (0, 0, 255), cv2.FILLED)
        #
        #         cv2.putText(frame, f"Count: {count}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        #         cv2.imshow(window_name, frame)
        #
        #     if cv2.waitKey(1) & 0xFF == ord('q'):
        #         stop_event.set()
        #         break
        #
        # cap.release()
        # cv2.destroyAllWindows()
        #
        # return count

    except Exception as e:
        logger.error(f"Error during Zipline crossing detection for camera {camera_id}: {str(e)}", exc_info=True)
        raise ZipError(f"zipline detection failed for camera {camera_id}: {str(e)}")


# Function to start each camera in its own process
def zipline_start(task):
    """
        Start the zipline detection process in a separate thread for the given camera task.
    """
    camera_id = task["cameraId"]
    try:
        line_coords = use_coordinates(task['line']['points'], task['line'])
        roi_coords = use_coordinates(task['roi']['points'], task['roi'])
        if camera_id not in tasks_processes:
            stop_event = multiprocessing.Event()
            tasks_processes[camera_id] = stop_event

            # Start zipline detection in a new process
            process = multiprocessing.Process(
                target=detect_people_count,
                args=(task["rtsp_link"], task["siteId"], camera_id, task["alarmId"], roi_coords, line_coords, task["arrow"],(task["display_width"], task["display_height"]), task["type"], stop_event) #rtsp_url, site_id, camera_id, alarm_id, roi_coords, zipline_coords, arrow_coords, display_size, type, stop_event
            )
            tasks_processes[camera_id] = process
            process.start()
            logger.info(f"Started zipline detection for camera {camera_id}.")
        else:
            logger.warning(f"zipline detection already running for camera {camera_id}.")
            return False

    except Exception as e:
        logger.error(f"Failed to start detection process for camera {camera_id}: {str(e)}", exc_info=True)
        return False
    return True

def zipline_stop(camera_ids):
    """
    Stop zipline detection for the given camera IDs.
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
                logger.info(f"Stopped zipline detection for camera {camera_id}.")
            except Exception as e:
                logger.error(f"Failed to stop zipline detection for camera {camera_id}: {str(e)}", exc_info=True)
        else:
            not_found_tasks.append(camera_id)

    return {
        "success": len(stopped_tasks) > 0,
        "stopped": stopped_tasks,
        "not_found": not_found_tasks
    }