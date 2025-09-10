# Road Damage CV

Computer vision desktop app (PySide6/Qt) for detecting road damage from RTMP video streams using YOLO/ONNX models, with optional Roboflow inference. Includes live performance metrics and CSV logging.

## Project Structure

```
road_damage_cv/
├── main.py                      # Application entry point
├── models/                      # Model files (.onnx, .pt)
├── logs/                        # CSV logs (auto-generated)
├── yolov5/                      # Upstream YOLOv5 code (export/utilities)
├── src/
│   ├── config/
│   │   └── settings.py          # Defaults (RTMP, ports, model names)
│   ├── core/
│   │   ├── class_counter.py
│   │   ├── logger.py
│   │   └── worker_thread.py     # Inference + I/O thread
│   └── ui/
│       ├── home_screen.py       # Inputs: RTMP, model type, GPS
│       ├── main_screen.py       # Video + metrics
│       ├── performance_chart.py # Chart widget (optional)
│       └── main_window.py       # Window orchestration
├── test_onnx_simple.py          # Smoke tests for ONNX
├── test_onnx_integration.py     # Integration tests
└── README.md
```

## Requirements

- Python 3.9+ (Windows recommended; app tested on Win10)
- GPU optional. CPU works but slower.

### Python packages

Install minimal runtime deps:
```bash
pip install PySide6 opencv-python numpy psutil
```

Optional backends/features:
- PyTorch (for local .pt models): `pip install torch --index-url https://download.pytorch.org/whl/cu121` (pick the right CUDA/CPU build)
- ONNXRuntime (for .onnx): `pip install onnxruntime-gpu` or `onnxruntime`
- GPS serial: `pip install pyserial`
- Roboflow inference: `pip install inference inference-sdk`
- NVIDIA metrics: `pip install nvidia-ml-py` (pynvml)

## Run

```bash
python main.py
```

In the Home screen, set:
- RTMP URL (e.g. `rtmp://192.168.1.102/live`)
- Model Type: Local Model | Roboflow Cloud API | Roboflow Local Inference
- For Local Model, pick a file from `models/` (e.g. `best_road_damage.onnx`)
- GPS: COM port (e.g. `COM8`) and baud

## Configuration

Defaults live in `src/config/settings.py`:
- `DEFAULT_RTMP`, `DEFAULT_COM_PORT`, `DEFAULT_BAUD`
- `DEFAULT_MODEL` (e.g., `best_road_damage.onnx`)
- Roboflow: `DEFAULT_ROBOFLOW_API_KEY`, `DEFAULT_WORKSPACE`, `DEFAULT_WORKFLOW_ID`, `DEFAULT_LOCAL_INFERENCE_URL`

Note: do not commit real API keys. Replace with placeholders before publishing.

## Models

Place `.onnx` or `.pt` files under `models/`. The UI auto-detects available files. Large weights should not be committed to Git.

## Logs

CSV files are written under `logs/` per session. These can grow quickly; keep them out of Git.

## Tests

Basic ONNX tests are provided:
```bash
python test_onnx_simple.py
python test_onnx_integration.py
```

## License

This project includes upstream `yolov5/` with its own license. Respect the YOLOv5 license when distributing.
