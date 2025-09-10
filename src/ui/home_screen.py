"""
Home screen UI component for configuration and connection setup.
"""

import os
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QWidget,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QFormLayout,
    QComboBox,
    QMessageBox,
)

from ..config import (
    DEFAULT_RTMP,
    DEFAULT_COM_PORT,
    DEFAULT_BAUD,
    DEFAULT_MODEL,
    DEFAULT_ROBOFLOW_API_KEY,
    DEFAULT_WORKSPACE,
    DEFAULT_WORKFLOW_ID,
    DEFAULT_LOCAL_INFERENCE_URL,
    GPS_UI_UPDATE_INTERVAL,
    GPS_HISTORY_SIZE,
    MODEL_TYPES,
    SUPPORTED_BAUD_RATES,
    MODELS_DIR
)

try:
    from inference import InferencePipeline
    from inference_sdk import InferenceHTTPClient
    ROBOFLOW_AVAILABLE = True
except Exception:
    ROBOFLOW_AVAILABLE = False


class HomeScreen(QWidget):
    """Home screen for configuration and connection setup"""

    def __init__(self, on_start_callback):
        super().__init__()
        self.on_start_callback = on_start_callback
        self._setup_ui()

    def _setup_ui(self):
        """Setup the UI components"""
        title = QLabel("Drone YOLO - Connect Stream")
        title.setAlignment(Qt.AlignCenter)
        title.setFont(QFont("Segoe UI", 16, QFont.Bold))

        form = QFormLayout()

        # RTMP URL input
        self.rtmp_input = QLineEdit(DEFAULT_RTMP)
        form.addRow("RTMP URL:", self.rtmp_input)

        # Model type selection
        self.model_type_combo = QComboBox()
        self.model_type_combo.addItems(list(MODEL_TYPES.values()))
        self.model_type_combo.currentTextChanged.connect(self._on_model_type_changed)
        form.addRow("Model Type:", self.model_type_combo)

        # Local model selection
        self.model_combo = QComboBox()
        available_models = self._find_available_models()
        if available_models:
            self.model_combo.addItems(available_models)
            if DEFAULT_MODEL in available_models:
                self.model_combo.setCurrentText(DEFAULT_MODEL)
        else:
            self.model_combo.addItems(["best_road_damage.onnx", "best_road_damage.pt", "crack4.pt", "yolov5s.pt", "yolov5m.pt"])
            self.model_combo.setCurrentText(DEFAULT_MODEL)
        form.addRow("Local Model:", self.model_combo)

        # Roboflow API configuration
        self.roboflow_api_key = QLineEdit(DEFAULT_ROBOFLOW_API_KEY)
        form.addRow("Roboflow API Key:", self.roboflow_api_key)

        self.roboflow_workspace = QLineEdit(DEFAULT_WORKSPACE)
        form.addRow("Workspace:", self.roboflow_workspace)

        self.roboflow_workflow_id = QLineEdit(DEFAULT_WORKFLOW_ID)
        form.addRow("Workflow ID:", self.roboflow_workflow_id)

        # Local inference configuration
        self.local_inference_url = QLineEdit(DEFAULT_LOCAL_INFERENCE_URL)
        form.addRow("Local Inference URL:", self.local_inference_url)

        # GPS configuration
        self.com_input = QLineEdit(DEFAULT_COM_PORT)
        form.addRow("GPS COM Port:", self.com_input)

        self.baud_combo = QComboBox()
        self.baud_combo.addItems(SUPPORTED_BAUD_RATES)
        self.baud_combo.setCurrentText(DEFAULT_BAUD)
        form.addRow("GPS Baud:", self.baud_combo)

        # Start button
        self.start_btn = QPushButton("Start Detection")
        self.start_btn.clicked.connect(self._on_start_clicked)

        # Layout
        layout = QVBoxLayout()
        layout.addWidget(title)
        layout.addLayout(form)
        layout.addWidget(self.start_btn)
        layout.addStretch(1)
        self.setLayout(layout)

        # Initialize UI state
        self._on_model_type_changed(MODEL_TYPES["LOCAL"])

    def _on_model_type_changed(self, model_type: str):
        """Enable/disable model configuration fields based on selected type"""
        is_local = model_type == MODEL_TYPES["LOCAL"]
        is_roboflow_cloud = model_type == MODEL_TYPES["ROBOFLOW_CLOUD"]
        is_roboflow_local = model_type == MODEL_TYPES["ROBOFLOW_LOCAL"]
        
        self.model_combo.setEnabled(is_local)
        self.roboflow_api_key.setEnabled(is_roboflow_cloud or is_roboflow_local)
        self.roboflow_workspace.setEnabled(is_roboflow_cloud or is_roboflow_local)
        self.roboflow_workflow_id.setEnabled(is_roboflow_cloud or is_roboflow_local)
        self.local_inference_url.setEnabled(is_roboflow_local)

    def _is_valid_rtmp(self, url: str) -> bool:
        """Validate RTMP URL format"""
        return url.startswith("rtmp://") and len(url.split("/")) >= 3

    def _on_start_clicked(self):
        """Handle start button click"""
        url = self.rtmp_input.text().strip()
        model_type = self.model_type_combo.currentText()
        com_port = self.com_input.text().strip()
        baud = int(self.baud_combo.currentText())

        if not self._is_valid_rtmp(url):
            QMessageBox.warning(
                self,
                "Invalid RTMP",
                "Please enter a valid RTMP URL (e.g., rtmp://host/live)",
            )
            return

        if not com_port:
            QMessageBox.warning(
                self, "Invalid COM Port", "Please enter a COM port (e.g., COM8)"
            )
            return

        # Validate based on model type
        if model_type == MODEL_TYPES["LOCAL"]:
            config = self._get_local_model_config(url, com_port, baud)
        elif model_type == MODEL_TYPES["ROBOFLOW_CLOUD"]:
            config = self._get_roboflow_cloud_config(url, com_port, baud)
        elif model_type == MODEL_TYPES["ROBOFLOW_LOCAL"]:
            config = self._get_roboflow_local_config(url, com_port, baud)
        else:
            return

        if config:
            self.on_start_callback(config)

    def _get_local_model_config(self, url: str, com_port: str, baud: int) -> dict:
        """Get configuration for local model"""
        model = self.model_combo.currentText().strip()
        if not self._validate_model_file(model):
            QMessageBox.warning(
                self,
                "Model File Not Found",
                f"The selected model file '{model}' could not be found.\n"
                f"Please ensure the model file is in the {MODELS_DIR} directory.",
            )
            return None

        return {
            "rtmp_url": url,
            "model_type": "local",
            "model": model,
            "com_port": com_port,
            "baud": baud,
            "gps_ui_update_interval": GPS_UI_UPDATE_INTERVAL,
            "gps_history_size": GPS_HISTORY_SIZE,
        }

    def _get_roboflow_cloud_config(self, url: str, com_port: str, baud: int) -> dict:
        """Get configuration for Roboflow cloud API"""
        if not ROBOFLOW_AVAILABLE:
            QMessageBox.warning(
                self,
                "Roboflow Not Available",
                "Roboflow inference package is not installed.\n"
                "Please install it with: pip install inference",
            )
            return None
        
        api_key = self.roboflow_api_key.text().strip()
        workspace = self.roboflow_workspace.text().strip()
        workflow_id = self.roboflow_workflow_id.text().strip()
        
        if not api_key or not workspace or not workflow_id:
            QMessageBox.warning(
                self,
                "Missing Roboflow Configuration",
                "Please fill in all Roboflow API fields (API Key, Workspace, Workflow ID).",
            )
            return None
        
        return {
            "rtmp_url": url,
            "model_type": "roboflow",
            "inference_type": "cloud",
            "api_key": api_key,
            "workspace": workspace,
            "workflow_id": workflow_id,
            "com_port": com_port,
            "baud": baud,
            "gps_ui_update_interval": GPS_UI_UPDATE_INTERVAL,
            "gps_history_size": GPS_HISTORY_SIZE,
        }

    def _get_roboflow_local_config(self, url: str, com_port: str, baud: int) -> dict:
        """Get configuration for Roboflow local inference"""
        inference_url = self.local_inference_url.text().strip()
        workflow_id = self.roboflow_workflow_id.text().strip()
        workspace = self.roboflow_workspace.text().strip()
        api_key = self.roboflow_api_key.text().strip()
        
        if not inference_url or not workflow_id or not workspace or not api_key:
            QMessageBox.warning(
                self,
                "Missing Local Inference Configuration",
                "Please fill in Local Inference URL, Workspace, Workflow ID, and API Key fields.",
            )
            return None
        
        if not inference_url.startswith("http://") and not inference_url.startswith("https://"):
            QMessageBox.warning(
                self,
                "Invalid URL",
                "Please enter a valid URL starting with http:// or https://",
            )
            return None
        
        return {
            "rtmp_url": url,
            "model_type": "roboflow",
            "inference_type": "local",
            "inference_url": inference_url,
            "api_key": api_key,
            "workspace": workspace,
            "workflow_id": workflow_id,
            "com_port": com_port,
            "baud": baud,
            "gps_ui_update_interval": GPS_UI_UPDATE_INTERVAL,
            "gps_history_size": GPS_HISTORY_SIZE,
        }

    def _find_available_models(self) -> list:
        """Find all .onnx and .pt model files in the models directory"""
        try:
            model_files = []
            
            if os.path.exists(MODELS_DIR):
                for file in os.listdir(MODELS_DIR):
                    if file.endswith('.onnx') or file.endswith('.pt'):
                        model_files.append(file)
            
            # Ensure DEFAULT_MODEL is prioritized if it exists on disk
            default_path = os.path.join(MODELS_DIR, DEFAULT_MODEL)
            model_files = sorted(set(model_files))
            if os.path.exists(default_path) and DEFAULT_MODEL in model_files:
                model_files.remove(DEFAULT_MODEL)
                model_files.insert(0, DEFAULT_MODEL)
                
            return model_files
        except Exception:
            return []

    def _validate_model_file(self, model_name: str) -> bool:
        """Check if the model file exists in the models directory"""
        if not model_name:
            return False
        
        model_path = os.path.join(MODELS_DIR, model_name)
        return os.path.exists(model_path)
