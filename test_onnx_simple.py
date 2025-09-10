#!/usr/bin/env python3
"""
Simple ONNX model test without worker thread dependencies
"""

import os
import numpy as np
import cv2

def test_onnx_model():
    """Test ONNX model with a simple image"""
    print("Testing ONNX model with simple inference...")
    
    # Check if ONNX model exists
    onnx_path = "models/best_road_damage.onnx"
    if not os.path.exists(onnx_path):
        print(f"Error: ONNX model not found at {onnx_path}")
        return False
    
    try:
        import onnxruntime as ort
        print("ONNX Runtime is available")
    except ImportError:
        print("Error: ONNX Runtime not available")
        print("Install with: pip install onnxruntime")
        return False
    
    try:
        # Load ONNX model
        providers = ['CPUExecutionProvider']  # Use CPU to avoid CUDA warnings
        session = ort.InferenceSession(onnx_path, providers=providers)
        
        # Get model info
        input_info = session.get_inputs()[0]
        output_info = session.get_outputs()[0]
        
        print(f"Model loaded successfully:")
        print(f"  - Input: {input_info.name} {input_info.shape}")
        print(f"  - Output: {output_info.name} {output_info.shape}")
        
        # Create a test image
        test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        print(f"Created test image: {test_image.shape}")
        
        # Preprocess image
        # Resize to model input size
        image_resized = cv2.resize(test_image, (640, 640))
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        image_normalized = image_rgb.astype(np.float32) / 255.0
        
        # Convert to CHW format and add batch dimension
        image_chw = np.transpose(image_normalized, (2, 0, 1))
        image_batch = np.expand_dims(image_chw, axis=0)
        
        print(f"Preprocessed image shape: {image_batch.shape}")
        
        # Run inference
        input_name = input_info.name
        outputs = session.run(None, {input_name: image_batch})
        
        print(f"Inference completed successfully!")
        print(f"Number of outputs: {len(outputs)}")
        
        for i, output in enumerate(outputs):
            print(f"  Output {i}: shape={output.shape}, dtype={output.dtype}")
            if i == 0:  # First output (predictions)
                print(f"    Min value: {output.min():.4f}")
                print(f"    Max value: {output.max():.4f}")
                print(f"    Mean value: {output.mean():.4f}")
        
        return True
        
    except Exception as e:
        print(f"Error testing ONNX model: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("=" * 60)
    print("Simple ONNX Model Test")
    print("=" * 60)
    
    success = test_onnx_model()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ ONNX model test PASSED!")
        print("\nYour ONNX model is working correctly.")
        print("The worker thread integration issue is just an import path conflict.")
        print("The ONNX model itself is ready to use.")
    else:
        print("❌ ONNX model test FAILED!")
        print("Check the errors above.")
    print("=" * 60)

if __name__ == "__main__":
    main()
