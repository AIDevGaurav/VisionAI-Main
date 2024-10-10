import threading
from flask import Blueprint, request, jsonify
from app.utils import start_feature_processing
from Features.Armed import armed_detection_stop, armed_detection_start
# from Features.Pet_detect import pet_start, pet_stop
from Features.Zipline import zipline_start, zipline_stop
from Features.fall import fall_stop, fall_start
from Features.motion_detector import motion_start
from app.exceptions import CustomError, handle_exception
from app.config import logger, get_executor
from Features.people_count import start_pc, stop_pc
from Features.fire import fire_stop, fire_start


api_blueprint = Blueprint('api', __name__)
executor = get_executor()

@api_blueprint.route('/start', methods=['POST'])
def start():
    try:
        tasks = request.json
        if not tasks or not isinstance(tasks, list):
            raise CustomError("Invalid input data. 'tasks' should be a list.")

        for task in tasks:
            c_id = task["cameraId"]
            s_id = task['siteId']
            rtsp = task['rtsp_link']
            width = task['display_width']
            height = task['display_height']

            for feature in task["features"]:
                typ, co = feature['type'], feature["co_ordinates"]
                if typ == "MOTION_DETECTION":
                    executor.submit(start_feature_processing, c_id, typ, rtsp, width, height)
                    motion_start(c_id, s_id, typ, co, width, height)
                elif typ == "PEOPLE_COUNT":
                    executor.submit(start_feature_processing, c_id, typ, rtsp, width, height)
                    start_pc(c_id, s_id, typ, co, width, height)
                elif typ == "PET_DETECTION":
                    executor.submit(start_feature_processing, c_id, typ, rtsp, width, height)
                    # pet_start(c_id, s_id, typ, co, width, height)
                elif typ == "FIRE_DETECTION":
                    executor.submit(start_feature_processing, c_id, typ, rtsp, width, height)
                    fire_start(c_id, s_id, typ, co)
                elif typ == "FALL_DETECTION":
                    executor.submit(start_feature_processing, c_id, typ, rtsp, width, height)
                    fall_start(c_id, s_id, typ, co, width, height)
                elif typ == "ZIP_LINE_CROSSING":
                    executor.submit(start_feature_processing, c_id, typ, rtsp, width, height)
                    zipline_start(c_id, s_id, typ, co)
                elif typ == "ARM_DETECTION":
                    executor.submit(start_feature_processing, c_id, typ, rtsp, width, height)
                    armed_detection_start(c_id, s_id, typ, co)
                else:
                    logger.error("Incorrect type provided check start if-else loop")
                    return jsonify({"success": False, "error": "Incorrect type"})

        logger.info("Detection tasks started successfully from api file.")
        return jsonify({"success": True, "message": "Detection started"}), 200
    except CustomError as e:
        return jsonify({"success": False, "error": str(e), "message": "Failed to start detection tasks."}), 400
    except Exception as e:
        return handle_exception(e)


@api_blueprint.route('/stop', methods=['POST'])
def stop_motion_detection():
    try:
        camera_ids = request.json.get('camera_ids', [])
        typ = request.json.get('type')
        if not isinstance(camera_ids, list):
            raise CustomError("'cameraIds' should be an array.")

        if typ == "MOTION_DETECTION":
            pass
            # response = motion_stop(camera_ids)
        elif typ == "PET_DETECTION":
            response = pet_stop(camera_ids)
        elif typ == "PEOPLE_COUNT":
            response = stop_pc(camera_ids)
        elif typ == "FIRE_DETECTION":
            response = fire_stop(camera_ids)
        elif typ == "FALL_DETECTION":
            response = fall_stop(camera_ids)
        elif typ == "ZIP_LINE_CROSSING":
            response = zipline_stop(camera_ids)
        elif typ == "ARM_DETECTION":
            response = armed_detection_stop(camera_ids)
        else:
            return jsonify({"success": False, "message": "Invaild Type"}), 400

        if response["success"]:
            logger.info(f"Detection stopped for cameras: {response['stopped']}")
        else:
            logger.warning(f"No active detection found for cameras: {response['not_found']}")

        return jsonify(response), 200

    except CustomError as e:
        return jsonify({"success": False, "error": str(e), "message": "Failed to stop detection tasks."}), 400
    except Exception as e:
        return handle_exception(e)
