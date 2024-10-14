from waitress import serve
from app import create_app
from flask_cors import CORS

app = create_app()
CORS(app, resources={r"/*": {"origins": "*"}})  # Allow all origins


if __name__ == "__main__":
    # Run the app using Waitress
    serve(app, host="192.168.1.18", port=5000)
