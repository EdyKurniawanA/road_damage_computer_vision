#!/usr/bin/env python3
"""
Test script to verify ONNX model integration with the worker thread
"""

import os
import sys
from pathlib import Path

# Add yolov5 to path first (before src) to avoid import conflicts
yolov5_path = Path(__file__).parent / "yolov5"
if str(yolov5_path) not in sys.path:
    sys.path.insert(0, str(yolov5_path))

# Add src to path for imports
src_path = Path(__file__).parent / "src"
if str(src_path) not in sys.path:
    sys.path.append(str(src_path))

def test_onnx_model_loading():
    """Test loading the ONNX model"""
    print("Testing ONNX model loading...")
    
    # Check if ONNX model exists
    onnx_path = "models/best_road_damage.onnx"
    if not os.path.exists(onnx_path):
        print(f"Error: ONNX model not found at {onnx_path}")
        print("Please run simple_convert.py first to convert your model.")
        return False
    
    print(f"Found ONNX model: {onnx_path}")
    
    # Test ONNX Runtime availability
    try:
        import onnxruntime as ort
        print("ONNX Runtime is available")
    except ImportError:
        print("Error: ONNX Runtime not available")
        print("Install with: pip install onnxruntime")
        return False
    
    # Test loading the model
    try:
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        session = ort.InferenceSession(onnx_path, providers=providers)
        
        # Get model info
        input_info = session.get_inputs()[0]
        output_info = session.get_outputs()[0]
        
        print(f"ONNX Model loaded successfully:")
        print(f"  - Input name: {input_info.name}")
        print(f"  - Input shape: {input_info.shape}")
        print(f"  - Input type: {input_info.type}")
        print(f"  - Output name: {output_info.name}")
        print(f"  - Output shape: {output_info.shape}")
        print(f"  - Output type: {output_info.type}")
        
        return True
        
    except Exception as e:
        print(f"Error loading ONNX model: {e}")
        return False

def test_worker_thread_integration():
    """Test worker thread with ONNX model"""
    print("\nTesting worker thread integration...")
    
    try:
        from src.core.worker_thread import WorkerThread
        
        # Create a test config for ONNX model
        config = {
            "model_type": "local",
            "model": "best_road_damage.onnx",  # Use ONNX model
            "rtmp_url": "rtmp://test",  # Dummy URL for testing
            "com_port": "COM1",  # Dummy port
            "baud": 9600
        }
        
        print("Creating worker thread with ONNX model...")
        worker = WorkerThread(config)
        
        print("Worker thread created successfully")
        print("Note: This test only verifies the worker can be created with ONNX config")
        print("For full testing, you would need to run the actual application")
        
        return True
        
    except Exception as e:
        print(f"Error creating worker thread: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("=" * 60)
    print("ONNX Model Integration Test")
    print("=" * 60)
    
    # Test ONNX model loading
    onnx_ok = test_onnx_model_loading()
    
    # Test worker thread integration
    worker_ok = test_worker_thread_integration()
    
    print("\n" + "=" * 60)
    print("Test Results:")
    print(f"  ONNX model loading: {'PASS' if onnx_ok else 'FAIL'}")
    print(f"  Worker thread integration: {'PASS' if worker_ok else 'FAIL'}")
    print("=" * 60)
    
    if onnx_ok and worker_ok:
        print("\n✅ ONNX integration is ready!")
        print("\nTo use the ONNX model in your app:")
        print("1. Make sure your config uses 'best_road_damage.onnx' as the model")
        print("2. Run your main.py application")
        print("3. The worker thread will automatically detect and load the ONNX model")
    else:
        print("\n❌ ONNX integration needs fixes. Check the errors above.")

if __name__ == "__main__":
    main()
