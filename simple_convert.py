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
    model = attempt_load('models/best_model_1.3.pt', device='cpu')
    
    # Print original class names
    print("Original class names:", model.names)
    
    # Define your desired class mapping
    desired_classes = {
        'Alligator_Crack': 'alligator cracking',
        'Longitudinal_Crack': 'longitudinal cracking', 
        'Pothole': 'pothole',
        'Transverse_Crack': 'transverse cracking'
    }
    
    # Create new class names list with only desired classes
    new_names = {}
    class_mapping = {}
    
    for i, (new_name, old_name) in enumerate(desired_classes.items()):
        if old_name in model.names.values():
            # Find the original class index
            for orig_idx, orig_name in model.names.items():
                if orig_name == old_name:
                    class_mapping[orig_idx] = i
                    new_names[i] = new_name
                    break
    
    print("Filtered class names:", new_names)
    print("Class mapping:", class_mapping)
    
    # Update model class names
    model.names = new_names
    
    # Set to evaluation mode
    model.eval()
    
    # Export to ONNX (more stable format)
    print("Exporting to ONNX...")
    torch.onnx.export(
        model,
        torch.randn(1, 3, 640, 640),
        'models/best_model_1.3.onnx',
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output']
    )
    
    print("Conversion completed! ONNX model saved as 'models/best_model_1.3.onnx'")
    print("Filtered classes:", list(new_names.values()))

if __name__ == "__main__":
    main()
