#!/usr/bin/env python3
"""
Simple model conversion script for best_road_damage.pt to ONNX
Based on the user's specific requirements
"""

import torch
import os
import sys
from pathlib import Path

# Add yolov5 to path for imports
yolov5_path = Path(__file__).parent / "yolov5"
if str(yolov5_path) not in sys.path:
    sys.path.append(str(yolov5_path))

# Windows compatibility fix
import platform
if platform.system() == "Windows":
    import pathlib
    _original_posix_path = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath

from models.experimental import attempt_load

def main():
    # Load your model
    print("Loading model...")
    model = attempt_load('models/best_road_damage.pt', device='cpu')
    
    # Set to evaluation mode
    model.eval()
    
    # Export to ONNX (more stable format)
    print("Exporting to ONNX...")
    torch.onnx.export(
        model,
        torch.randn(1, 3, 640, 640),
        'models/best_road_damage.onnx',
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output']
    )
    
    print("Conversion completed! ONNX model saved as 'models/best_road_damage.onnx'")

if __name__ == "__main__":
    main()
