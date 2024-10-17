import threading
from flask import Blueprint, request, jsonify
from app.utils import start_feature_processing
from Features.Armed import armed_stop, armed_start
from Features.Zipline import zipline_start, zipline_stop
from Features.fall import fall_stop, fall_start
from Features.motion_detector import motion_start, motion_stop
from app.exceptions import CustomError, handle_exception
from app.config import logger, get_executor
from Features.people_count import start_pc, stop_pc
from Features.fire import fire_stop, fire_start
from Features.pet import pet_stop, pet_start

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
            enable = task["enabled_features"]
            disabled = task['disabled_features']

            if enable:
                for feature in enable:
                    typ, co = feature['type'], feature["co_ordinates"]
                    if typ == "MOTION_DETECTION":
                        motion_start(c_id, s_id, typ, co, width, height, rtsp)
                    elif typ == "PEOPLE_COUNT":
                        start_pc(c_id, s_id, typ, co, width, height, rtsp)
                    elif typ == "PET_DETECTION":
                        pet_start(c_id, s_id, typ, co, width, height, rtsp)
                    elif typ == "FIRE_DETECTION":
                        fire_start(c_id, s_id, typ, co,width,height, rtsp)
                    elif typ == "FALL_DETECTION":
                        fall_start(c_id, s_id, typ, co, width, height, rtsp)
                    elif typ == "ZIP_LINE_CROSSING":
                        zipline_start(c_id, s_id, typ, co,width,height, rtsp)
                    elif typ == "ARM_DETECTION":
                        armed_start(c_id, s_id, typ, co, width, height, rtsp)
                    else:
                        logger.error("Incorrect type provided check start if-else loop")
                        return jsonify({"success": False, "error": "Incorrect type"})

            if disabled:
                for disable in disabled:
                    if disable == "MOTION_DETECTION":
                        executor.submit(motion_stop, c_id, disable)
                    elif disable == "PEOPLE_COUNT":
                        executor.submit(stop_pc, c_id, disable)
                    elif disable == "PET_DETECTION":
                        executor.submit(pet_stop, c_id, disable)
                    elif disable == "FIRE_DETECTION":
                        executor.submit(fire_stop, c_id, disable)
                    elif disable == "FALL_DETECTION":
                        executor.submit(fall_stop, c_id, disable)
                    elif disable == "ZIP_LINE_CROSSING":
                        executor.submit(zipline_stop, c_id, disable)
                    elif disable == "ARM_DETECTION":
                        executor.submit(armed_stop, c_id, disable)
                    else:
                        logger.error("Incorrect type provided check start if-else loop")
                        return jsonify({"success": False, "error": "Incorrect type"})


        logger.info("Detection tasks started successfully from api file.")
        return jsonify({"success": True, "message": "Detection started"}), 200
    except CustomError as e:
        logger.error("error", e)
        return jsonify({"success": False, "message": "Failed to start detection tasks."}), 400
    except Exception as e:
        return handle_exception(e)

