import os
from app.config import logger
from app.exceptions import handle_exception
from flask_cors import CORS
from flask import Flask
from app.mqtt_handler import start_mqtt_client

def create_app():
    app = Flask(__name__)
    CORS(app)

    # Register blueprints (API routes)
    from app.api import api_blueprint
    app.register_blueprint(api_blueprint)

    # Global error handler
    app.register_error_handler(Exception, handle_exception)

    # Start the MQTT client loop once when the app starts
    start_mqtt_client()

    # Log process ID to track app restarts
    logger.info(f"App Started Successfully with PID: {os.getpid()}")

    print("Started......")
    return app
