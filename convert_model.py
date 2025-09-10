#!/usr/bin/env python3
"""
Model conversion script for best_road_damage.pt
Converts the PyTorch model to ONNX format for better compatibility
"""

import os
import sys
import torch
import warnings
from pathlib import Path

# Add yolov5 to path for imports
yolov5_path = Path(__file__).parent / "yolov5"
if str(yolov5_path) not in sys.path:
    sys.path.append(str(yolov5_path))

# Windows compatibility fix for PosixPath
import platform
if platform.system() == "Windows":
    import pathlib
    _original_posix_path = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath

from models.experimental import attempt_load
from models.yolo import Model
import torch.nn as nn

class ONNXWrapper(nn.Module):
    """Wrapper to simplify YOLOv5 model output for ONNX export"""
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, x):
        # Get the raw model output
        outputs = self.model(x)
        
        # YOLOv5 returns a tuple of lists, we need to flatten and concatenate
        if isinstance(outputs, (list, tuple)):
            # Flatten all outputs into a single tensor
            flattened = []
            for output in outputs:
                if isinstance(output, (list, tuple)):
                    for item in output:
                        if isinstance(item, torch.Tensor):
                            flattened.append(item)
                elif isinstance(output, torch.Tensor):
                    flattened.append(output)
            
            if flattened:
                # Concatenate all tensors along the last dimension
                return torch.cat(flattened, dim=-1)
            else:
                # Fallback: return zeros
                return torch.zeros(1, 1, 1)
        else:
            return outputs

def convert_model_to_onnx():
    """Convert the PyTorch model to ONNX format"""
    
    # Model paths
    model_path = "models/best_road_damage.pt"
    onnx_path = "models/best_road_damage.onnx"
    
    print(f"Converting model: {model_path}")
    print(f"Output path: {onnx_path}")
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        return False
    
    try:
        # Load the model using attempt_load (same as worker_thread.py)
        print("Loading model using attempt_load...")
        model = attempt_load(model_path, device='cpu')
        
        # Print model info
        print(f"Model loaded successfully")
        print(f"Model type: {type(model)}")
        print(f"Model device: {next(model.parameters()).device}")
        
        # Get model attributes
        if hasattr(model, 'stride'):
            print(f"Model stride: {model.stride}")
        if hasattr(model, 'names'):
            print(f"Model names: {model.names}")
        if hasattr(model, 'yaml'):
            print(f"Model yaml: {model.yaml}")
        
        # Set model to evaluation mode
        model.eval()
        
        # Create dummy input tensor (batch_size=1, channels=3, height=640, width=640)
        dummy_input = torch.randn(1, 3, 640, 640)
        print(f"Dummy input shape: {dummy_input.shape}")
        
        # Test original model forward pass
        print("Testing original model forward pass...")
        with torch.no_grad():
            test_output = model(dummy_input)
            print(f"Original output type: {type(test_output)}")
            if isinstance(test_output, (list, tuple)):
                print(f"Original output length: {len(test_output)}")
                for i, out in enumerate(test_output):
                    if hasattr(out, 'shape'):
                        print(f"  Output {i} shape: {out.shape}")
                    else:
                        print(f"  Output {i} type: {type(out)}")
                        if isinstance(out, (list, tuple)):
                            print(f"    Length: {len(out)}")
                            for j, sub_out in enumerate(out):
                                if hasattr(sub_out, 'shape'):
                                    print(f"      Sub-output {j} shape: {sub_out.shape}")
                                else:
                                    print(f"      Sub-output {j} type: {type(sub_out)}")
            else:
                print(f"Original output shape: {test_output.shape}")
        
        # Wrap model for ONNX export
        print("Wrapping model for ONNX export...")
        wrapped_model = ONNXWrapper(model)
        wrapped_model.eval()
        
        # Test wrapped model
        print("Testing wrapped model forward pass...")
        with torch.no_grad():
            wrapped_output = wrapped_model(dummy_input)
            print(f"Wrapped output type: {type(wrapped_output)}")
            print(f"Wrapped output shape: {wrapped_output.shape}")
        
        # Export to ONNX
        print("Exporting to ONNX...")
        torch.onnx.export(
            wrapped_model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        print(f"Model successfully converted to ONNX: {onnx_path}")
        
        # Verify ONNX model
        try:
            import onnx
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)
            print("ONNX model verification passed")
        except ImportError:
            print("ONNX not available for verification")
        except Exception as e:
            print(f"ONNX verification failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"Error converting model: {e}")
        import traceback
        traceback.print_exc()
        return False

def convert_to_torchscript():
    """Convert the PyTorch model to TorchScript format as backup"""
    
    model_path = "models/best_road_damage.pt"
    torchscript_path = "models/best_road_damage_torchscript.pt"
    
    print(f"\nConverting to TorchScript: {torchscript_path}")
    
    try:
        # Load model
        model = attempt_load(model_path, device='cpu')
        model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(1, 3, 640, 640)
        
        # Convert to TorchScript
        traced_model = torch.jit.trace(model, dummy_input)
        traced_model.save(torchscript_path)
        
        print(f"Model successfully converted to TorchScript: {torchscript_path}")
        return True
        
    except Exception as e:
        print(f"Error converting to TorchScript: {e}")
        return False

def main():
    """Main conversion function"""
    print("=" * 60)
    print("Road Damage Model Conversion Script")
    print("=" * 60)
    
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Convert to ONNX
    success = convert_model_to_onnx()
    
    if success:
        print("\n" + "=" * 60)
        print("Conversion completed successfully!")
        print("=" * 60)
        
        # Also try TorchScript as backup
        convert_to_torchscript()
        
        print("\nAvailable model formats:")
        for file in os.listdir("models"):
            if file.endswith(('.pt', '.onnx')):
                file_path = os.path.join("models", file)
                size = os.path.getsize(file_path) / (1024 * 1024)  # MB
                print(f"  - {file} ({size:.1f} MB)")
    else:
        print("\n" + "=" * 60)
        print("Conversion failed!")
        print("=" * 60)
        return 1
    
    return 0

if __name__ == "__main__":
    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")
    
    exit_code = main()
    sys.exit(exit_code)
