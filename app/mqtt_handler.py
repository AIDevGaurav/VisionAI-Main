import os
import paho.mqtt.client as mqtt
import json
from app.config import logger

# Initialize the MQTT client once, globally
mqtt_client = mqtt.Client(client_id=f"Gaurav_{os.getpid()}", clean_session=True)

# Enable detailed logging
mqtt_client.enable_logger()


# Define MQTT callbacks
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        logger.info(f"Connected to MQTT broker with result code {rc}")
        # client.subscribe(topic)
    else:
        logger.error(f"Failed to connect to MQTT broker, result code {rc}")


def on_disconnect(client, userdata, rc):
    logger.info(f"Disconnected from MQTT broker with result code {rc}")
    if rc != 0:
        logger.warning("Unexpected disconnection, reconnecting...")
        try:
            client.reconnect()
        except Exception as e:
            logger.error(f"Reconnection failed: {e}")


# Assign the callbacks to the client
mqtt_client.on_connect = on_connect
mqtt_client.on_disconnect = on_disconnect


# Connect and start the loop once when the app starts
def start_mqtt_client():
    logger.info(f"Starting MQTT client with client_id={mqtt_client._client_id.decode()} and PID: {os.getpid()}")
    mqtt_client.connect("192.168.1.23", 1883, keepalive=600)
    mqtt_client.loop_start()


# Function to publish a message (no re-initialization or reconnection)
def publish_message_mqtt(topic, message):
    mqtt_client.publish(topic, json.dumps(message))
    logger.info(f"Published message: {json.dumps(message)}")
