"""
Worker thread for handling video processing and inference in a separate thread.
"""

import sys
import time
import json
import threading
import queue
import os
import signal
import tracemalloc
from datetime import datetime
from typing import Optional, Dict, Any
from pathlib import Path
import gc
import requests
import base64

import numpy as np
import cv2
import psutil
import torch

# Add yolov5 to path for imports
import sys
from pathlib import Path
yolov5_path = Path(__file__).parent.parent.parent / "yolov5"
if str(yolov5_path) not in sys.path:
    sys.path.insert(0, str(yolov5_path))  # Insert at beginning to prioritize yolov5 imports

# Windows compatibility fix for PosixPath (based on Stack Overflow solution)
import platform
if platform.system() == "Windows":
    # Store original PosixPath for restoration
    import pathlib
    _original_posix_path = pathlib.PosixPath
    # Monkey patch pathlib to use WindowsPath instead of PosixPath
    pathlib.PosixPath = pathlib.WindowsPath

from models.common import DetectMultiBackend
from utils.general import check_img_size, non_max_suppression, scale_boxes
from utils.torch_utils import select_device

try:
    import serial  # pyserial
except Exception:
    serial = None

try:
    from inference import InferencePipeline
    from inference_sdk import InferenceHTTPClient
    ROBOFLOW_AVAILABLE = True
except Exception:
    ROBOFLOW_AVAILABLE = False

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except Exception:
    ONNX_AVAILABLE = False

from PySide6.QtCore import QThread, Signal

from .class_counter import ClassCounter
from .logger import Logger
from ..config import (
    FRAME_SKIP_RATIO,
    MAX_QUEUE_SIZE,
    INFERENCE_TIMEOUT,
    CLEANUP_TIMEOUT,
    LOGS_DIR
)


class WorkerThread(QThread):
    """Separate thread for heavy processing to prevent GUI freezing"""
    frame_ready = Signal(np.ndarray, dict)  # frame, counts
    status_update = Signal(str)
    
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.stop_event = threading.Event()
        self.frame_queue = queue.Queue(maxsize=MAX_QUEUE_SIZE)
        self.log_queue = queue.Queue(maxsize=MAX_QUEUE_SIZE)
        
        # Performance tracking
        self.fps = 0.0
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.frame_skip_counter = 0
        
        # Model and processing
        self.model = None
        self.onnx_session = None
        self.roboflow_pipeline = None
        self.class_counter = ClassCounter(accumulate=True)
        
        # GPS data
        self.latest_gps = {}
        
        # Setup logging
        self._setup_logging()
        
    def _setup_logging(self):
        """Setup CSV logging with timestamped filename"""
        os.makedirs(LOGS_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = os.path.join(LOGS_DIR, f"drone_yolo_{timestamp}.csv")
        
        fieldnames = [
            "timestamp", "system_uptime", "frame_id", "fps", "counts",
            "gps_utc", "gps_lat", "gps_lon", "gps_alt", "detection_count", "extra"
        ]
        
        self.logger = Logger(log_filename, fieldnames)
        self.frame_id = 0
        
        # Precompute a distinct color palette for classes (BGR)
        # 20 visually distinct colors; will cycle if classes > len(palette)
        self._palette = [
            (255, 56, 56), (255, 157, 151), (255, 112, 31), (255, 178, 29),
            (207, 210, 49), (72, 249, 10), (146, 204, 23), (61, 219, 134),
            (26, 147, 52), (0, 212, 187), (44, 153, 168), (0, 194, 255),
            (52, 69, 147), (100, 115, 255), (142, 140, 255), (204, 173, 255),
            (255, 186, 255), (255, 140, 157), (237, 149, 100), (255, 198, 108)
        ]

    def _color_for_class(self, class_id: int) -> tuple:
        """Return a distinct BGR color for a given class id."""
        if class_id is None:
            return (0, 255, 0)
        try:
            return self._palette[class_id % len(self._palette)]
        except Exception:
            return (0, 255, 0)

    def _draw_box_with_label(self, image, x1, y1, x2, y2, label: str, class_id: int = None):
        """Draw a rectangle with a thicker outline and a readable label box.
        - Uses per-class color if class_id is provided
        - Larger font and boxed text for visibility
        """
        color = self._color_for_class(class_id)
        # Box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        # Label background
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7  # larger than before
        thickness = 2
        label = str(label)
        (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
        th_box = th + 6
        y_text = max(y1 - 5, th_box + 5)
        # Filled rectangle for text background
        cv2.rectangle(image, (x1, y_text - th_box), (x1 + tw + 6, y_text), color, -1)
        # Text in contrasting color (black or white depending on brightness)
        brightness = (0.299 * color[2] + 0.587 * color[1] + 0.114 * color[0])
        text_color = (0, 0, 0) if brightness > 128 else (255, 255, 255)
        cv2.putText(image, label, (x1 + 3, y_text - 4), font, font_scale, text_color, thickness, cv2.LINE_AA)

    def run(self):
        """Main worker thread loop"""
        try:
            self.status_update.emit("Starting worker thread...")
            
            # Reset class counter for new session
            self.class_counter.reset()
            print("Class counter reset for new session")
            
            if self.config["model_type"] == "local":
                self._run_local_model()
            else:
                self._run_roboflow_model()
                
        except Exception as e:
            self.status_update.emit(f"Worker thread error: {e}")
        finally:
            self._cleanup()

    def _run_local_model(self):
        """Run local YOLO model with optimizations - using YOLOv5 DetectMultiBackend"""
        try:
            # Load model using YOLOv5 DetectMultiBackend (same as detect.py)
            self.status_update.emit("Loading custom model...")
            model_path = self.config["model"]
            if not os.path.isabs(model_path):
                model_path = os.path.join("models", model_path)
            
            # Check if model file exists
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model not found: {model_path}")
            
            # Check if it's an ONNX model
            if model_path.endswith('.onnx'):
                self._load_onnx_model(model_path)
                return
            
            # Debug: Inspect model file before loading
            self.status_update.emit("Debug: Inspecting model file...")
            print(f"Debug: Loading model from: {model_path}")
            try:
                ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
                print("Checkpoint keys:", ckpt.keys())
                print("Type:", type(ckpt))
                
                if 'model' in ckpt:
                    print("Model type:", type(ckpt['model']))
                    print("Model attributes:", [attr for attr in dir(ckpt['model']) if not attr.startswith('_')])
                    if hasattr(ckpt['model'], 'yaml'):
                        print("Model yaml:", ckpt['model'].yaml)
                    if hasattr(ckpt['model'], 'names'):
                        print("Model names:", ckpt['model'].names)
                    if hasattr(ckpt['model'], 'stride'):
                        print("Model stride:", ckpt['model'].stride)
                
                # Check for other common keys
                for key in ['epoch', 'best_fitness', 'optimizer', 'ema', 'updates']:
                    if key in ckpt:
                        print(f"{key}: {ckpt[key]}")
                        
            except Exception as debug_e:
                print(f"Debug error loading checkpoint: {debug_e}")
                self.status_update.emit(f"Debug error: {debug_e}")
            
            # Load model using YOLOv5 DetectMultiBackend (handles all model types)
            device = select_device("")
            
            # Memory optimization: Use smaller image size if memory is low
            try:
                self.model = DetectMultiBackend(model_path, device=device, dnn=False, data=None, fp16=True)  # Use fp16 to save memory
                self.stride = getattr(self.model, 'stride', 32)
                self.names = getattr(self.model, 'names', {})
                self.pt = getattr(self.model, 'pt', False)
                self.imgsz = check_img_size((640, 640), s=self.stride)  # check image size
                
                # Debug: Print what DetectMultiBackend extracted
                print(f"Debug: DetectMultiBackend loaded successfully")
                print(f"  - Stride: {self.stride}")
                print(f"  - Names: {self.names}")
                print(f"  - PT (PyTorch): {self.pt}")
                print(f"  - Image size: {self.imgsz}")
                print(f"  - Model device: {getattr(self.model, 'device', 'unknown')}")
                print(f"  - Model fp16: {getattr(self.model, 'fp16', 'unknown')}")
                print(f"  - Available model attributes: {[attr for attr in dir(self.model) if not attr.startswith('_')]}")
                self.status_update.emit(f"Debug: Model loaded - stride: {self.stride}, names: {list(self.names.values()) if self.names else 'None'}")
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    self.status_update.emit("Memory low: Using smaller model size...")
                    # Clear any existing model
                    if hasattr(self, 'model'):
                        del self.model
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
                    # Try with smaller image size
                    self.model = DetectMultiBackend(model_path, device=device, dnn=False, data=None, fp16=True)
                    self.stride = getattr(self.model, 'stride', 32)
                    self.names = getattr(self.model, 'names', {})
                    self.pt = getattr(self.model, 'pt', False)
                    self.imgsz = check_img_size((320, 320), s=self.stride)  # smaller image size
                    
                    # Debug: Print what DetectMultiBackend extracted (fallback case)
                    print(f"Debug: DetectMultiBackend loaded successfully (fallback with smaller size)")
                    print(f"  - Stride: {self.stride}")
                    print(f"  - Names: {self.names}")
                    print(f"  - PT (PyTorch): {self.pt}")
                    print(f"  - Image size: {self.imgsz}")
                    print(f"  - Model device: {getattr(self.model, 'device', 'unknown')}")
                    print(f"  - Model fp16: {getattr(self.model, 'fp16', 'unknown')}")
                    print(f"  - Available model attributes: {[attr for attr in dir(self.model) if not attr.startswith('_')]}")
                    self.status_update.emit(f"Debug: Model loaded (fallback) - stride: {self.stride}, names: {list(self.names.values()) if self.names else 'None'}")
                else:
                    raise e
            
            # Optimize model
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(device)
            
            if device == "cuda":
                torch.backends.cudnn.benchmark = True
                self.model.half()
                
            # Warmup
            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            with torch.no_grad():
                _ = self.model(dummy)
                
            self.status_update.emit(f"Model loaded on {device}")
            
            # Start capture thread
            capture_thread = threading.Thread(target=self._capture_frames, daemon=True)
            capture_thread.start()
            
            # Start GPS thread
            gps_thread = threading.Thread(target=self._gps_loop, daemon=True)
            gps_thread.start()
            
            # Start logging thread
            logging_thread = threading.Thread(target=self._logging_loop, daemon=True)
            logging_thread.start()
            
            # Main processing loop
            self._process_frames()
            
        except Exception as e:
            self.status_update.emit(f"Local model error: {e}")

    def _load_onnx_model(self, model_path):
        """Load ONNX model for inference"""
        try:
            if not ONNX_AVAILABLE:
                raise ImportError("ONNX Runtime not available. Install with: pip install onnxruntime")
            
            self.status_update.emit("Loading ONNX model...")
            print(f"Loading ONNX model from: {model_path}")
            
            # Create ONNX Runtime session
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']
            self.onnx_session = ort.InferenceSession(model_path, providers=providers)
            
            # Get model info
            input_info = self.onnx_session.get_inputs()[0]
            output_info = self.onnx_session.get_outputs()[0]
            
            print(f"ONNX Model loaded successfully")
            print(f"  - Input name: {input_info.name}")
            print(f"  - Input shape: {input_info.shape}")
            print(f"  - Input type: {input_info.type}")
            print(f"  - Output name: {output_info.name}")
            print(f"  - Output shape: {output_info.shape}")
            print(f"  - Output type: {output_info.type}")
            
            # Set model attributes (for compatibility with existing code)
            self.stride = 32  # Default stride for YOLOv5
            self.names = {0: 'alligator cracking', 1: 'edge cracking', 2: 'longitudinal cracking', 
                         3: 'patching', 4: 'pothole', 5: 'rutting', 6: 'transverse cracking'}
            self.pt = False  # Not PyTorch model
            self.imgsz = 640  # Default image size
            
            self.status_update.emit(f"ONNX model loaded - stride: {self.stride}, names: {list(self.names.values())}")
            
            # Start capture thread
            capture_thread = threading.Thread(target=self._capture_frames, daemon=True)
            capture_thread.start()
            
            # Start GPS thread
            gps_thread = threading.Thread(target=self._gps_loop, daemon=True)
            gps_thread.start()
            
            # Start logging thread
            logging_thread = threading.Thread(target=self._logging_loop, daemon=True)
            logging_thread.start()
            
            # Main processing loop
            self._process_onnx_frames()
            
        except Exception as e:
            self.status_update.emit(f"ONNX model error: {e}")
            import traceback
            traceback.print_exc()

    def _run_roboflow_model(self):
        """Run Roboflow model with optimizations"""
        try:
            self.status_update.emit("Initializing Roboflow...")
            
            # Check if using local inference
            if self.config.get("inference_type") == "local":
                self._run_local_roboflow_inference()
            else:
                self._run_cloud_roboflow_inference()
                
        except Exception as e:
            self.status_update.emit(f"Roboflow error: {e}")

    def _run_cloud_roboflow_inference(self):
        """Run cloud-based Roboflow inference"""
        print(f"Initializing Roboflow Cloud with:")
        print(f"  API Key: {self.config['api_key'][:10]}...")
        print(f"  Workspace: {self.config['workspace']}")
        print(f"  Workflow ID: {self.config['workflow_id']}")
        print(f"  Video URL: {self.config['rtmp_url']}")
        
        self.roboflow_pipeline = InferencePipeline.init_with_workflow(
            api_key=self.config["api_key"],
            workspace_name=self.config["workspace"],
            workflow_id=self.config["workflow_id"],
            video_reference=self.config["rtmp_url"],
            max_fps=15,
            on_prediction=self._roboflow_callback,
        )
        
        self.roboflow_pipeline.start()
        self.status_update.emit("Roboflow cloud started")
        print("Roboflow cloud pipeline started successfully")
        
        # Start GPS and logging threads
        gps_thread = threading.Thread(target=self._gps_loop, daemon=True)
        gps_thread.start()
        
        logging_thread = threading.Thread(target=self._logging_loop, daemon=True)
        logging_thread.start()
        
        # Wait for stop event
        while not self.stop_event.is_set():
            time.sleep(0.1)

    def _run_local_roboflow_inference(self):
        """Run local Roboflow inference via SDK"""
        self.status_update.emit("Connecting to local Roboflow Inference...")
        
        try:
            # Initialize the inference client with API key
            self.inference_client = InferenceHTTPClient(
                api_url=self.config['inference_url'],
                api_key=self.config.get('api_key', '')
            )
            
            # Test connection
            try:
                server_info = self.inference_client.get_server_info()
                self.status_update.emit(f"Connected to local Roboflow Inference: {server_info}")
            except Exception as e:
                self.status_update.emit(f"Server info not available, but continuing: {e}")
            
            self.status_update.emit("Connected to local Roboflow Inference")
            
            # Start capture thread
            capture_thread = threading.Thread(target=self._capture_frames, daemon=True)
            capture_thread.start()
            
            # Start GPS thread
            gps_thread = threading.Thread(target=self._gps_loop, daemon=True)
            gps_thread.start()
            
            # Start logging thread
            logging_thread = threading.Thread(target=self._logging_loop, daemon=True)
            logging_thread.start()
            
            # Main processing loop
            self._process_local_inference_frames()
            
        except Exception as e:
            self.status_update.emit(f"Cannot connect to local inference: {e}")
            return

    def _process_local_inference_frames(self):
        """Process frames using local Roboflow inference SDK"""
        print("Starting local inference frame processing loop")
        frame_count = 0
        
        while not self.stop_event.is_set():
            try:
                frame = self.frame_queue.get(timeout=INFERENCE_TIMEOUT)
                frame_count += 1
                print(f"Got frame {frame_count} from queue, shape: {frame.shape}")
            except queue.Empty:
                print("No frames in queue, waiting...")
                continue
                
            # Frame skipping for performance
            self.frame_skip_counter += 1
            if self.frame_skip_counter % FRAME_SKIP_RATIO != 0:
                continue
                
            try:
                # Debug: Print frame info
                if self.frame_count % 10 == 0:
                    self.status_update.emit(f"Processing frame {self.frame_count}, shape: {frame.shape}")
                    print(f"Processing frame {self.frame_count}, shape: {frame.shape}")
                
                # Run inference using the SDK
                print(f"Running inference with workspace='{self.config.get('workspace', 'default')}', workflow_id='{self.config.get('workflow_id', 'default')}'")
                
                result = self.inference_client.run_workflow(
                    workspace_name=self.config.get("workspace", "default"),
                    workflow_id=self.config.get("workflow_id", "default"),
                    images={
                        "image": frame
                    }
                )
                
                # Debug: Print result info
                print(f"Inference completed. Result type: {type(result)}")
                if isinstance(result, dict):
                    print(f"Result keys: {list(result.keys())}")
                else:
                    print(f"Result content: {result}")
                
                if self.frame_count % 10 == 0:
                    self.status_update.emit(f"Inference completed. Result type: {type(result)}")
                
                # Extract counts from result
                counts = self._extract_counts_from_sdk_result(result)
                print(f"Extracted counts: {counts}")
                
                # Create annotated frame
                annotated = frame.copy()
                
                # Try to get annotated image from workflow outputs
                if isinstance(result, dict) and "output_image" in result and result["output_image"] is not None:
                    try:
                        oi = result["output_image"]
                        annotated = oi.numpy_image if hasattr(oi, "numpy_image") else oi
                        print("Using output_image for annotated frame")
                    except Exception:
                        pass
                elif "output2" in result and result["output2"] is not None:
                    print("Using output2 (label visualization) for annotated frame")
                    annotated = result["output2"]
                elif "output3" in result and result["output3"] is not None:
                    print("Using output3 (bounding box visualization) for annotated frame")
                    annotated = result["output3"]
                elif "image" in result and result["image"] is not None:
                    print("Using result image for annotated frame")
                    annotated = result["image"]
                else:
                    # Fallback: draw predictions manually
                    print("Drawing predictions manually")
                    predictions = None
                    if "output" in result and isinstance(result["output"], dict):
                        output = result["output"]
                        if "predictions" in output:
                            predictions = output["predictions"]
                        elif "detections" in output:
                            predictions = output["detections"]
                        elif "results" in output:
                            predictions = output["results"]
                    elif "predictions" in result:
                        predictions = result["predictions"]
                    
                    if predictions:
                        # Draw boxes from common prediction schemas
                        try:
                            for pred in predictions:
                                if not isinstance(pred, dict):
                                    continue
                                label = pred.get("class") or pred.get("class_name") or pred.get("label") or "object"
                                # Support xyxy or center-width-height
                                if all(k in pred for k in ("x1", "y1", "x2", "y2")):
                                    x1, y1, x2, y2 = int(pred["x1"]), int(pred["y1"]), int(pred["x2"]), int(pred["y2"])
                                else:
                                    # Assume center-based with width/height
                                    cx = pred.get("x") or pred.get("cx") or 0
                                    cy = pred.get("y") or pred.get("cy") or 0
                                    w = pred.get("width") or pred.get("w") or 0
                                    h = pred.get("height") or pred.get("h") or 0
                                    x1 = int(cx - w / 2)
                                    y1 = int(cy - h / 2)
                                    x2 = int(cx + w / 2)
                                    y2 = int(cy + h / 2)
                                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                cv2.putText(annotated, label, (x1, max(0, y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                                print(f"Detected {label} at ({x1},{y1})-({x2},{y2})")
                        except Exception as _:
                            pass
                
                # Update class counter using dedicated Roboflow method (similar to local model)
                self.class_counter.update_from_roboflow(result)
                
                # Extract counts for display and logging
                counts = self.class_counter.get_counts()
                print(f"Local Roboflow counts: {counts}")
                
                # Choose counts to display/log: prefer per-frame, fallback to accumulated
                try:
                    display_counts = counts if (isinstance(counts, dict) and counts) else self.class_counter.get_counts()
                except Exception:
                    display_counts = counts

                # Emit results to GUI
                print(f"Emitting frame to GUI with counts: {display_counts}")
                self.frame_ready.emit(annotated, display_counts)
                
                # Log data
                if not self.log_queue.full():
                    self.log_queue.put((display_counts, frame))
                    
            except Exception as e:
                if hasattr(self, '_last_error_time'):
                    if time.time() - self._last_error_time > 5.0:
                        self.status_update.emit(f"Local inference error: {e}")
                        self._last_error_time = time.time()
                else:
                    self._last_error_time = time.time()

    def _extract_counts_from_sdk_result(self, result):
        """Extract class counts from SDK inference result"""
        counts = {}
        try:
            # Handle different result formats from the SDK
            if isinstance(result, dict):
                # Debug: Print the full result structure
                print(f"Full result structure: {list(result.keys())}")
                
                # Try different possible locations for predictions
                predictions = None
                
                # Check for workflow outputs (your specific case)
                if "output" in result and isinstance(result["output"], dict):
                    output = result["output"]
                    print(f"Output structure: {list(output.keys()) if isinstance(output, dict) else type(output)}")
                    
                    if "predictions" in output:
                        predictions = output["predictions"]
                    elif "detections" in output:
                        predictions = output["detections"]
                    elif "results" in output:
                        predictions = output["results"]
                    elif isinstance(output, list):
                        predictions = output
                
                # Check for direct predictions
                elif "predictions" in result:
                    predictions = result["predictions"]
                elif "results" in result and isinstance(result["results"], list):
                    predictions = result["results"]
                elif "detections" in result:
                    predictions = result["detections"]
                
                # Extract counts from predictions
                if predictions:
                    from collections import Counter
                    labels = []
                    
                    if isinstance(predictions, list):
                        for pred in predictions:
                            if isinstance(pred, dict):
                                if "class" in pred:
                                    labels.append(pred["class"])
                                elif "class_name" in pred:
                                    labels.append(pred["class_name"])
                                elif "label" in pred:
                                    labels.append(pred["label"])
                            elif isinstance(pred, str):
                                labels.append(pred)
                    
                    counts = dict(Counter(labels))
                    print(f"Extracted labels: {labels}")
                    print(f"Counts: {counts}")
                else:
                    print("No predictions found in result")
                    
        except Exception as e:
            print(f"Error extracting counts: {e}")
            counts = {}
        return counts

    def _capture_frames(self):
        """Optimized RTMP capture with frame dropping"""
        try:
            # Use FFmpeg backend for better RTMP support
            cap = cv2.VideoCapture(self.config["rtmp_url"], cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer
            
            if not cap.isOpened():
                self.status_update.emit(f"Cannot open stream: {self.config['rtmp_url']}")
                return
                
            self.status_update.emit("Video capture started")
            
            frame_count = 0
            while not self.stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    if frame_count % 100 == 0:
                        print("Failed to read frame from RTMP stream")
                    time.sleep(0.01)
                    continue
                
                frame_count += 1
                if frame_count % 30 == 0:
                    print(f"Captured frame {frame_count}, shape: {frame.shape}")
                
                # Drop frames if queue is full (prevent memory buildup)
                if not self.frame_queue.full():
                    self.frame_queue.put(frame)
                    if frame_count % 30 == 0:
                        print(f"Added frame {frame_count} to queue")
                else:
                    # Drop oldest frame and add new one
                    try:
                        self.frame_queue.get_nowait()
                        self.frame_queue.put(frame)
                        if frame_count % 30 == 0:
                            print(f"Replaced frame in queue (queue full)")
                    except queue.Empty:
                        pass
                
                # Update FPS
                self.frame_count += 1
                now = time.time()
                if now - self.last_fps_time >= 1.0:
                    self.fps = self.frame_count / (now - self.last_fps_time)
                    self.frame_count = 0
                    self.last_fps_time = now
                    
        except Exception as e:
            self.status_update.emit(f"Capture error: {e}")
        finally:
            try:
                cap.release()
            except:
                pass

    def _process_frames(self):
        """Process frames with optimizations"""
        from collections import Counter
        
        while not self.stop_event.is_set():
            try:
                frame = self.frame_queue.get(timeout=INFERENCE_TIMEOUT)
            except queue.Empty:
                continue
                
            # Frame skipping for performance
            self.frame_skip_counter += 1
            if self.frame_skip_counter % FRAME_SKIP_RATIO != 0:
                continue
                
            # Run inference using YOLOv5 DetectMultiBackend (same as detect.py)
            # Preprocess image with memory optimization
            try:
                # Ensure image is in correct format (H, W, C) -> (C, H, W)
                if len(frame.shape) == 3 and frame.shape[2] == 3:
                    # Convert from HWC to CHW format
                    frame = frame.transpose(2, 0, 1)
                
                # Convert to float32 first to avoid memory issues
                frame_float = frame.astype(np.float32)
                im = torch.from_numpy(frame_float).to(self.model.device)
                im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
                im /= 255  # 0 - 255 to 0.0 - 1.0
                
                # Ensure we have 4D tensor: (batch, channels, height, width)
                if len(im.shape) == 3:
                    im = im[None]  # add batch dimension
                elif len(im.shape) == 2:
                    # Handle grayscale images by adding channel dimension
                    im = im.unsqueeze(0).unsqueeze(0)  # (H, W) -> (1, 1, H, W)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    self.status_update.emit("Memory error: Clearing cache and retrying...")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
                    # Try with smaller image
                    frame_small = cv2.resize(frame, (320, 320))
                    
                    # Ensure image is in correct format (H, W, C) -> (C, H, W)
                    if len(frame_small.shape) == 3 and frame_small.shape[2] == 3:
                        frame_small = frame_small.transpose(2, 0, 1)
                    
                    frame_float = frame_small.astype(np.float32)
                    im = torch.from_numpy(frame_float).to(self.model.device)
                    im = im.half() if self.model.fp16 else im.float()
                    im /= 255
                    
                    # Ensure we have 4D tensor: (batch, channels, height, width)
                    if len(im.shape) == 3:
                        im = im[None]  # add batch dimension
                    elif len(im.shape) == 2:
                        im = im.unsqueeze(0).unsqueeze(0)  # (H, W) -> (1, 1, H, W)
                else:
                    raise e
            
            # Run inference
            with torch.no_grad():
                pred = self.model(im, augment=False, visualize=False)
            
            # Apply NMS
            pred = non_max_suppression(pred, 0.25, 0.45, None, False, max_det=1000)
            
            # Process predictions
            annotated = frame.copy()
            counts = {}
            
            for i, det in enumerate(pred):  # per image
                if len(det):
                    # Debug: Print detection tensor shape
                    print(f"Debug: Detection tensor shape: {det.shape}")
                    print(f"Debug: Detection tensor dtype: {det.dtype}")
                    print(f"Debug: First few detections: {det[:min(3, len(det))]}")
                    
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], frame.shape).round()
                    
                    # Extract counts - handle different detection formats
                    try:
                        if det.shape[1] == 6:  # Standard format: x1, y1, x2, y2, conf, cls
                            for c in det[:, 5].unique():
                                n = (det[:, 5] == c).sum()  # detections per class
                                class_name = self.names[int(c)]
                                counts[class_name] = int(n)
                        elif det.shape[1] == 5:  # Alternative format: x1, y1, x2, y2, conf (no class)
                            counts["object"] = len(det)
                        else:
                            counts["unknown"] = len(det)
                    except Exception as e:
                        self.status_update.emit(f"Error extracting counts: {e}")
                        counts["error"] = 1
                    
                    # Draw boxes - handle different detection formats
                    try:
                        if det.shape[1] == 6:  # Standard format: x1, y1, x2, y2, conf, cls
                            for detection in reversed(det):
                                if len(detection) >= 6:
                                    x1, y1, x2, y2, conf, cls = detection[:6]
                                    c = int(cls)  # integer class
                                    label = f"{self.names[c]} {conf:.2f}"
                                    self._draw_box_with_label(
                                        annotated,
                                        int(x1), int(y1), int(x2), int(y2),
                                        label,
                                        class_id=c
                                    )
                        elif det.shape[1] == 5:  # Alternative format: x1, y1, x2, y2, conf (no class)
                            for detection in reversed(det):
                                if len(detection) >= 5:
                                    x1, y1, x2, y2, conf = detection[:5]
                                    label = f"object {conf:.2f}"
                                    self._draw_box_with_label(
                                        annotated,
                                        int(x1), int(y1), int(x2), int(y2),
                                        label,
                                        class_id=None
                                    )
                        else:
                            self.status_update.emit(f"Unexpected detection format: {det.shape}")
                    except Exception as e:
                        self.status_update.emit(f"Error drawing boxes: {e}")
            
            # Emit results to GUI
            self.frame_ready.emit(annotated, counts)
            
            # Log data
            if not self.log_queue.full():
                self.log_queue.put((counts, frame))
            
            # Memory cleanup after each frame
            del im, pred, annotated
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    def _process_onnx_frames(self):
        """Process frames using ONNX model"""
        while not self.stop_event.is_set():
            try:
                frame = self.frame_queue.get(timeout=INFERENCE_TIMEOUT)
            except queue.Empty:
                continue
                
            # Frame skipping for performance
            self.frame_skip_counter += 1
            if self.frame_skip_counter % FRAME_SKIP_RATIO != 0:
                continue
                
            # Preprocess image for ONNX
            try:
                # Resize to model input size
                input_size = 640
                frame_resized = cv2.resize(frame, (input_size, input_size))
                
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                
                # Normalize to [0, 1]
                frame_normalized = frame_rgb.astype(np.float32) / 255.0
                
                # Convert to CHW format and add batch dimension
                frame_chw = np.transpose(frame_normalized, (2, 0, 1))
                frame_batch = np.expand_dims(frame_chw, axis=0)
                
            except Exception as e:
                self.status_update.emit(f"ONNX preprocessing error: {e}")
                continue
            
            # Run ONNX inference
            try:
                input_name = self.onnx_session.get_inputs()[0].name
                outputs = self.onnx_session.run(None, {input_name: frame_batch})
                
                # Process ONNX output (simplified - may need adjustment based on actual output format)
                if outputs and len(outputs) > 0:
                    # The ONNX output should be similar to YOLOv5 output
                    # This is a simplified processing - you may need to adjust based on your model's output format
                    pred = outputs[0]  # Assuming first output contains predictions
                    
                    # Convert to torch tensor for compatibility with existing NMS code
                    if isinstance(pred, np.ndarray):
                        pred_tensor = torch.from_numpy(pred)
                    else:
                        pred_tensor = pred
                    
                    # Apply NMS (same as PyTorch version)
                    pred_nms = non_max_suppression(pred_tensor, 0.25, 0.45, None, False, max_det=1000)
                    
                    # Process predictions
                    annotated = frame.copy()
                    counts = {}
                    
                    for i, det in enumerate(pred_nms):  # per image
                        if len(det):
                            # Rescale boxes from input_size to original frame size
                            det[:, :4] = scale_boxes((input_size, input_size), det[:, :4], frame.shape).round()
                            
                            # Extract counts
                            try:
                                if det.shape[1] == 6:  # Standard format: x1, y1, x2, y2, conf, cls
                                    for c in det[:, 5].unique():
                                        n = (det[:, 5] == c).sum()  # detections per class
                                        class_name = self.names[int(c)]
                                        counts[class_name] = int(n)
                                elif det.shape[1] == 5:  # Alternative format: x1, y1, x2, y2, conf (no class)
                                    counts["object"] = len(det)
                                else:
                                    counts["unknown"] = len(det)
                            except Exception as e:
                                self.status_update.emit(f"Error extracting ONNX counts: {e}")
                                counts["error"] = 1
                            
                            # Draw boxes
                            try:
                                if det.shape[1] == 6:  # Standard format
                                    for detection in reversed(det):
                                        if len(detection) >= 6:
                                            x1, y1, x2, y2, conf, cls = detection[:6]
                                            c = int(cls)
                                            label = f"{self.names[c]} {conf:.2f}"
                                            self._draw_box_with_label(
                                                annotated,
                                                int(x1), int(y1), int(x2), int(y2),
                                                label,
                                                class_id=c
                                            )
                                elif det.shape[1] == 5:  # Alternative format
                                    for detection in reversed(det):
                                        if len(detection) >= 5:
                                            x1, y1, x2, y2, conf = detection[:5]
                                            label = f"object {conf:.2f}"
                                            self._draw_box_with_label(
                                                annotated,
                                                int(x1), int(y1), int(x2), int(y2),
                                                label,
                                                class_id=None
                                            )
                            except Exception as e:
                                self.status_update.emit(f"Error drawing ONNX boxes: {e}")
                else:
                    # No detections
                    annotated = frame.copy()
                    counts = {}
                
                # Emit results to GUI
                self.frame_ready.emit(annotated, counts)
                
                # Log data
                if not self.log_queue.full():
                    self.log_queue.put((counts, frame))
                
            except Exception as e:
                if hasattr(self, '_last_onnx_error_time'):
                    if time.time() - self._last_onnx_error_time > 5.0:
                        self.status_update.emit(f"ONNX inference error: {e}")
                        self._last_onnx_error_time = time.time()
                else:
                    self._last_onnx_error_time = time.time()

    def _roboflow_callback(self, result, video_frame):
        """Roboflow prediction callback"""
        try:
            print(f"Roboflow callback received - result type: {type(result)}, video_frame type: {type(video_frame)}")
            
            # Normalize input: sometimes callbacks return (result, frame) as a tuple
            if isinstance(result, tuple):
                try:
                    if len(result) == 2:
                        result, video_frame = result
                    elif len(result) == 1:
                        result = result[0]
                except Exception:
                    pass

            # Extract frame and counts
            annotated = None
            counts = {}
            
            # Prefer unified output image if present
            if isinstance(result, dict) and result.get("output_image") is not None:
                try:
                    oi = result["output_image"]
                    annotated = oi.numpy_image if hasattr(oi, "numpy_image") else oi
                except Exception:
                    annotated = result.get("output_image")
            # Get annotated frame from legacy field
            elif isinstance(result, dict) and result.get("label_visualization"):
                try:
                    annotated = result["label_visualization"].numpy_image
                except Exception:
                    annotated = result["label_visualization"]
            elif video_frame is not None:
                if hasattr(video_frame, 'image'):
                    annotated = video_frame.image
                elif hasattr(video_frame, 'numpy_image'):
                    annotated = video_frame.numpy_image
                else:
                    annotated = video_frame
            
            # Extract predictions from various schemas
            predictions = None
            if isinstance(result, dict):
                print(f"Roboflow result keys: {list(result.keys())}")
                
                # Try different possible locations for predictions
                if "model_predictions" in result and isinstance(result["model_predictions"], dict) and "predictions" in result["model_predictions"]:
                    predictions = result["model_predictions"]["predictions"]
                    print(f"Found predictions in model_predictions.predictions: {len(predictions) if predictions else 0}")
                elif "output" in result and isinstance(result["output"], dict):
                    out = result["output"]
                    if isinstance(out, dict):
                        predictions = out.get("predictions") or out.get("detections") or out.get("results")
                        print(f"Found predictions in output: {len(predictions) if predictions else 0}")
                elif "predictions" in result:
                    predictions = result["predictions"]
                    print(f"Found predictions in root: {len(predictions) if predictions else 0}")
                elif "detections" in result:
                    predictions = result["detections"]
                    print(f"Found detections in root: {len(predictions) if predictions else 0}")
                else:
                    print("No predictions found in result")
            
            # Update class counter using dedicated Roboflow method (similar to local model)
            self.class_counter.update_from_roboflow(result)
            
            # Extract counts for display and logging
            counts = self.class_counter.get_counts()
            print(f"Roboflow counts: {counts}")
            
            # Draw bounding boxes if we have an annotated frame
            if annotated is not None and predictions:
                try:
                    print(f"Drawing {len(predictions)} predictions on frame")
                    for i, pred in enumerate(predictions):
                        if not isinstance(pred, dict):
                            continue
                        
                        # Get label for drawing
                        label = (pred.get("class") or pred.get("class_name") or 
                                pred.get("label") or pred.get("name") or 
                                pred.get("category") or "object")
                        
                        # Try different bounding box formats
                        if all(k in pred for k in ("x1", "y1", "x2", "y2")):
                            x1, y1, x2, y2 = int(pred["x1"]), int(pred["y1"]), int(pred["x2"]), int(pred["y2"])
                        elif all(k in pred for k in ("x", "y", "width", "height")):
                            x1 = int(pred["x"])
                            y1 = int(pred["y"])
                            x2 = int(pred["x"] + pred["width"])
                            y2 = int(pred["y"] + pred["height"])
                        elif all(k in pred for k in ("cx", "cy", "w", "h")):
                            cx = pred.get("cx") or 0
                            cy = pred.get("cy") or 0
                            w = pred.get("w") or 0
                            h = pred.get("h") or 0
                            x1 = int(cx - w / 2)
                            y1 = int(cy - h / 2)
                            x2 = int(cx + w / 2)
                            y2 = int(cy + h / 2)
                        else:
                            # Skip drawing if we can't determine bounding box
                            continue
                            
                        self._draw_box_with_label(
                            annotated,
                            x1, y1, x2, y2,
                            label,
                            class_id=None
                        )
                        # Try to map label to class id if available
                        class_id = None
                        try:
                            if isinstance(self.names, dict):
                                base_label = str(label).split()[0]
                                for k, v in self.names.items():
                                    if v == base_label:
                                        class_id = int(k)
                                        break
                        except Exception:
                            class_id = None
                        self._draw_box_with_label(
                            annotated,
                            x1, y1, x2, y2,
                            label,
                            class_id=class_id
                        )
                        print(f"Detected {label} at ({x1},{y1})-({x2},{y2})")
                        
                except Exception as e:
                    print(f"Error drawing predictions: {e}")
                    pass
            
            # Ensure we emit some frame even without annotations
            if annotated is None and video_frame is not None:
                if hasattr(video_frame, 'image'):
                    annotated = video_frame.image
                elif hasattr(video_frame, 'numpy_image'):
                    annotated = video_frame.numpy_image
                else:
                    annotated = video_frame
            
            # Determine display counts - prefer per-frame, fallback to accumulated
            try:
                display_counts = counts if (isinstance(counts, dict) and counts) else self.class_counter.get_counts()
            except Exception:
                display_counts = counts if isinstance(counts, dict) else {}
            
            if annotated is not None:
                self.frame_ready.emit(annotated, display_counts)
                
            if not self.log_queue.full():
                self.log_queue.put((display_counts, video_frame))
                
        except Exception as e:
            if hasattr(self, '_last_error_time'):
                if time.time() - self._last_error_time > 5.0:
                    self.status_update.emit(f"Roboflow callback error: {e}")
                    self._last_error_time = time.time()
            else:
                self._last_error_time = time.time()

    def _gps_loop(self):
        """GPS NMEA processing"""
        if serial is None:
            return
            
        try:
            ser = serial.Serial(self.config["com_port"], self.config["baud"], timeout=1)
        except Exception:
            ser = None
            
        last_gga = {}
        last_rmc = {}
        
        try:
            while not self.stop_event.is_set():
                if ser is None:
                    time.sleep(1)
                    continue
                    
                try:
                    line = ser.readline().decode(errors="ignore").strip()
                    if not line.startswith("$"):
                        continue
                        
                    parts = line.split(",")
                    talker = parts[0][3:6]
                    
                    if talker == "GGA" and len(parts) >= 10:
                        lat = self._parse_lat(parts[2], parts[3])
                        lon = self._parse_lon(parts[4], parts[5])
                        alt = self._safe_float(parts[9])
                        last_gga = {"lat": lat, "lon": lon, "alt": alt}
                    elif talker == "RMC" and len(parts) >= 10:
                        utc = self._parse_time(parts[1], parts[9])
                        last_rmc = {"utc": utc}
                    
                    # Merge GPS data
                    merged = {}
                    merged.update(last_rmc)
                    merged.update(last_gga)
                    if merged:
                        self.latest_gps = merged
                        
                except Exception:
                    continue
        finally:
            try:
                if ser is not None:
                    ser.close()
            except:
                pass

    def _logging_loop(self):
        """CSV logging thread"""
        while not self.stop_event.is_set():
            try:
                counts, frame = self.log_queue.get(timeout=0.1)
            except queue.Empty:
                continue
                
            # Log detection data
            try:
                total_detections = int(sum(counts.values())) if isinstance(counts, dict) else 0
            except Exception:
                total_detections = 0
            self.logger.log(
                timestamp=datetime.now().isoformat(),
                system_uptime=time.perf_counter(),
                frame_id=self.frame_id,
                fps=self.fps,
                counts=counts,
                gps_utc=self.latest_gps.get("utc", ""),
                gps_lat=self.latest_gps.get("lat", ""),
                gps_lon=self.latest_gps.get("lon", ""),
                gps_alt=self.latest_gps.get("alt", ""),
                detection_count=total_detections,
                extra={
                    "accumulated_counts": self.class_counter.get_counts(),
                    "total_detections": total_detections
                }
            )
            self.frame_id += 1

    def _cleanup(self):
        """Cleanup resources"""
        try:
            self.status_update.emit("Cleaning up resources...")
            
            # Stop Roboflow
            if self.roboflow_pipeline:
                try:
                    self.roboflow_pipeline.join()
                except:
                    pass
                    
            # Clear model
            if self.model:
                del self.model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            # Clear ONNX session
            if self.onnx_session:
                del self.onnx_session
                    
            # Close logger
            try:
                self.logger.close()
            except:
                pass
                
            # Restore original PosixPath (Windows compatibility fix)
            if platform.system() == "Windows" and '_original_posix_path' in globals():
                import pathlib
                pathlib.PosixPath = _original_posix_path
                
            # Force garbage collection
            gc.collect()
            
            self.status_update.emit("Cleanup complete")
            
        except Exception as e:
            self.status_update.emit(f"Cleanup error: {e}")

    def stop(self):
        """Stop the worker thread"""
        self.stop_event.set()
        self.quit()
        self.wait(CLEANUP_TIMEOUT * 1000)  # Wait up to 2 seconds

    @staticmethod
    def _safe_float(s: str) -> Optional[float]:
        try:
            return float(s)
        except Exception:
            return None

    @staticmethod
    def _parse_lat(lat_str: str, hemi: str) -> Optional[float]:
        if not lat_str:
            return None
        try:
            deg = float(lat_str[:2])
            minutes = float(lat_str[2:])
            val = deg + minutes / 60.0
            if hemi == "S":
                val = -val
            return val
        except Exception:
            return None

    @staticmethod
    def _parse_lon(lon_str: str, hemi: str) -> Optional[float]:
        if not lon_str:
            return None
        try:
            deg = float(lon_str[:3])
            minutes = float(lon_str[3:])
            val = deg + minutes / 60.0
            if hemi == "W":
                val = -val
            return val
        except Exception:
            return None

    @staticmethod
    def _parse_time(time_str: str, date_str: str) -> Optional[str]:
        if not time_str or not date_str or len(date_str) < 6:
            return None
        try:
            hh = int(time_str[0:2])
            mm = int(time_str[2:4])
            ss = float(time_str[4:])
            dd = int(date_str[0:2])
            mo = int(date_str[2:4])
            yy = int(date_str[4:6])
            iso = f"20{yy:02d}-{mo:02d}-{dd:02d}T{hh:02d}:{mm:02d}:{int(ss):02d}Z"
            return iso
        except Exception:
            return None
