"""
Configuration settings and constants for the Road Damage CV application.
"""

# Default configuration values
DEFAULT_RTMP = "rtmp://192.168.1.102/live"
DEFAULT_COM_PORT = "COM8"
DEFAULT_BAUD = "115200"
DEFAULT_MODEL = "best_road_damage.onnx"
DEFAULT_ROBOFLOW_API_KEY = "Pwr60R16IPozPzElpd1Q"
DEFAULT_WORKSPACE = "edys-flow"
DEFAULT_WORKFLOW_ID = "road-damage-cv"
DEFAULT_LOCAL_INFERENCE_URL = "http://localhost:9001"

# Performance and stability settings
FRAME_SKIP_RATIO = 2  # Process every 2nd frame
MAX_QUEUE_SIZE = 3
INFERENCE_TIMEOUT = 0.1
CLEANUP_TIMEOUT = 2.0

# Model types
MODEL_TYPES = {
    "LOCAL": "Local Model",
    "ROBOFLOW_CLOUD": "Roboflow Cloud API", 
    "ROBOFLOW_LOCAL": "Roboflow Local Inference"
}

# Supported baud rates
SUPPORTED_BAUD_RATES = ["9600", "19200", "38400", "57600", "115200"]

# File paths
MODELS_DIR = "models"
LOGS_DIR = "logs"
