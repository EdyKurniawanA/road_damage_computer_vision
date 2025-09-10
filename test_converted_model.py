#!/usr/bin/env python3
"""
Test script to verify the converted ONNX model works with the worker thread
"""

import os
import sys
import torch
import numpy as np
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

def test_original_model():
    """Test the original PyTorch model"""
    print("Testing original PyTorch model...")
    
    try:
        from models.experimental import attempt_load
        
        model = attempt_load('models/best_road_damage.pt', device='cpu')
        model.eval()
        
        # Test with dummy input
        dummy_input = torch.randn(1, 3, 640, 640)
        
        with torch.no_grad():
            output = model(dummy_input)
            print(f"Original model output type: {type(output)}")
            if isinstance(output, (list, tuple)):
                print(f"Output length: {len(output)}")
                for i, out in enumerate(output):
                    print(f"  Output {i} shape: {out.shape}")
            else:
                print(f"Output shape: {output.shape}")
        
        print("Original model test: PASSED")
        return True
        
    except Exception as e:
        print(f"Original model test: FAILED - {e}")
        return False

def test_onnx_model():
    """Test the converted ONNX model"""
    print("\nTesting ONNX model...")
    
    if not os.path.exists('models/best_road_damage.onnx'):
        print("ONNX model not found. Please run the conversion script first.")
        return False
    
    try:
        import onnx
        import onnxruntime as ort
        
        # Load ONNX model
        onnx_model = onnx.load('models/best_road_damage.onnx')
        onnx.checker.check_model(onnx_model)
        
        # Create ONNX Runtime session
        session = ort.InferenceSession('models/best_road_damage.onnx')
        
        # Test with dummy input
        dummy_input = np.random.randn(1, 3, 640, 640).astype(np.float32)
        
        # Run inference
        outputs = session.run(None, {'input': dummy_input})
        
        print(f"ONNX model output length: {len(outputs)}")
        for i, output in enumerate(outputs):
            print(f"  Output {i} shape: {output.shape}")
        
        print("ONNX model test: PASSED")
        return True
        
    except ImportError as e:
        print(f"ONNX model test: SKIPPED - Missing dependencies: {e}")
        print("Install with: pip install onnx onnxruntime")
        return False
    except Exception as e:
        print(f"ONNX model test: FAILED - {e}")
        return False

def test_worker_thread_compatibility():
    """Test if the model works with the worker thread's DetectMultiBackend"""
    print("\nTesting worker thread compatibility...")
    
    try:
        from models.common import DetectMultiBackend
        from utils.torch_utils import select_device
        
        # Test with DetectMultiBackend (same as worker_thread.py)
        device = select_device("")
        model = DetectMultiBackend('models/best_road_damage.pt', device=device, dnn=False, data=None, fp16=False)
        
        # Test inference
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        dummy_tensor = torch.from_numpy(dummy.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
        
        with torch.no_grad():
            output = model(dummy_tensor)
            print(f"DetectMultiBackend output type: {type(output)}")
            if isinstance(output, (list, tuple)):
                print(f"Output length: {len(output)}")
                for i, out in enumerate(output):
                    print(f"  Output {i} shape: {out.shape}")
            else:
                print(f"Output shape: {output.shape}")
        
        print("Worker thread compatibility test: PASSED")
        return True
        
    except Exception as e:
        print(f"Worker thread compatibility test: FAILED - {e}")
        return False

def main():
    print("=" * 60)
    print("Model Compatibility Test")
    print("=" * 60)
    
    # Test original model
    original_ok = test_original_model()
    
    # Test ONNX model
    onnx_ok = test_onnx_model()
    
    # Test worker thread compatibility
    worker_ok = test_worker_thread_compatibility()
    
    print("\n" + "=" * 60)
    print("Test Results Summary:")
    print(f"  Original PyTorch model: {'PASS' if original_ok else 'FAIL'}")
    print(f"  ONNX model: {'PASS' if onnx_ok else 'FAIL'}")
    print(f"  Worker thread compatibility: {'PASS' if worker_ok else 'FAIL'}")
    print("=" * 60)
    
    if original_ok and worker_ok:
        print("Your model should work fine with the current worker thread!")
    else:
        print("There may be compatibility issues. Check the error messages above.")

if __name__ == "__main__":
    main()
