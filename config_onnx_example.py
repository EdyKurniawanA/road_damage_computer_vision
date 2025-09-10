#!/usr/bin/env python3
"""
Example configuration for using the ONNX model in your application
"""

# Example configuration for ONNX model
ONNX_CONFIG = {
    "model_type": "local",
    "model": "best_road_damage.onnx",  # Use the converted ONNX model
    "rtmp_url": "rtmp://your-stream-url",  # Replace with your actual RTMP URL
    "com_port": "COM1",  # Replace with your GPS COM port
    "baud": 9600
}

# Example configuration for PyTorch model (original)
PYTORCH_CONFIG = {
    "model_type": "local", 
    "model": "best_road_damage.pt",  # Use the original PyTorch model
    "rtmp_url": "rtmp://your-stream-url",
    "com_port": "COM1",
    "baud": 9600
}

def get_config(use_onnx=True):
    """
    Get configuration for the worker thread
    
    Args:
        use_onnx (bool): If True, return ONNX config, else PyTorch config
    
    Returns:
        dict: Configuration dictionary
    """
    if use_onnx:
        return ONNX_CONFIG.copy()
    else:
        return PYTORCH_CONFIG.copy()

if __name__ == "__main__":
    print("ONNX Configuration:")
    print(ONNX_CONFIG)
    print("\nPyTorch Configuration:")
    print(PYTORCH_CONFIG)
