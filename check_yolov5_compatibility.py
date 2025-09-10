#!/usr/bin/env python3
"""
YOLOv5 Compatibility Checker for Road Damage CV Application
Checks version compatibility, dependencies, and model loading capabilities.
"""

import sys
import os
import subprocess
import importlib
from pathlib import Path

def check_python_version():
    """Check Python version compatibility"""
    print("Python Version Check")
    print(f"   Current: {sys.version}")
    
    major, minor = sys.version_info[:2]
    if major == 3 and minor >= 8:
        print("   PASS Python 3.8+ (Compatible)")
        return True
    else:
        print("   FAIL Python 3.8+ required")
        return False

def check_torch_compatibility():
    """Check PyTorch installation and compatibility"""
    print("\nPyTorch Compatibility Check")
    
    try:
        import torch
        print(f"   PyTorch version: {torch.__version__}")
        
        # Check CUDA availability
        if torch.cuda.is_available():
            print(f"   CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA version: {torch.version.cuda}")
        else:
            print("   CUDA: Not available (CPU only)")
        
        # Check if version is compatible
        version_parts = torch.__version__.split('.')
        major, minor = int(version_parts[0]), int(version_parts[1])
        
        if major >= 1 and minor >= 7:
            print("   PASS PyTorch 1.7+ (Compatible)")
            return True
        else:
            print("   WARN PyTorch 1.7+ recommended")
            return True  # Still works but might be slower
            
    except ImportError:
        print("   FAIL PyTorch not installed")
        return False

def check_yolov5_installation():
    """Check YOLOv5 installation and version"""
    print("\nYOLOv5 Installation Check")
    
    # Check if yolov5 directory exists
    yolov5_path = Path("yolov5")
    if not yolov5_path.exists():
        print("   FAIL YOLOv5 directory not found")
        return False
    
    print("   PASS YOLOv5 directory found")
    
    # Check for key files
    key_files = ["hubconf.py", "models/yolo.py", "utils/general.py"]
    for file in key_files:
        if (yolov5_path / file).exists():
            print(f"   PASS {file} found")
        else:
            print(f"   FAIL {file} missing")
            return False
    
    # Try to import YOLOv5
    try:
        sys.path.insert(0, str(yolov5_path))
        import hubconf
        print("   PASS YOLOv5 import successful")
        
        # Check hubconf version
        if hasattr(hubconf, '__version__'):
            print(f"   YOLOv5 version: {hubconf.__version__}")
        else:
            print("   YOLOv5 version: Unknown (local installation)")
        
        return True
        
    except ImportError as e:
        print(f"   FAIL YOLOv5 import failed: {e}")
        return False

def check_model_loading():
    """Test model loading capabilities"""
    print("\nModel Loading Test")
    
    try:
        import torch
        sys.path.insert(0, "yolov5")
        from hubconf import custom
        
        # Test loading a model
        model_path = "models/best_road_damage.pt"
        if os.path.exists(model_path):
            print(f"   Testing model: {model_path}")
            try:
                model = custom(model_path)
                print("   PASS Model loaded successfully")
                
                # Test inference
                import numpy as np
                dummy_input = np.zeros((640, 640, 3), dtype=np.uint8)
                with torch.no_grad():
                    results = model(dummy_input)
                print("   PASS Model inference test passed")
                return True
                
            except Exception as e:
                print(f"   FAIL Model loading failed: {e}")
                return False
        else:
            print(f"   WARN Model file not found: {model_path}")
            print("   Testing with YOLOv5s.pt...")
            
            try:
                model = custom("yolov5s.pt")
                print("   PASS YOLOv5s model loaded successfully")
                return True
            except Exception as e:
                print(f"   FAIL YOLOv5s model loading failed: {e}")
                return False
                
    except Exception as e:
        print(f"   FAIL Model loading test failed: {e}")
        return False

def check_dependencies():
    """Check required dependencies"""
    print("\nDependencies Check")
    
    required_packages = [
        ("torch", "PyTorch"),
        ("torchvision", "TorchVision"),
        ("opencv-python", "OpenCV"),
        ("numpy", "NumPy"),
        ("PIL", "Pillow"),
        ("PySide6", "PySide6"),
        ("psutil", "psutil")
    ]
    
    missing_packages = []
    
    for package, name in required_packages:
        try:
            if package == "opencv-python":
                import cv2
                print(f"   PASS {name}: {cv2.__version__}")
            elif package == "PIL":
                from PIL import Image
                print(f"   PASS {name}: {Image.__version__}")
            else:
                module = importlib.import_module(package)
                version = getattr(module, '__version__', 'Unknown')
                print(f"   PASS {name}: {version}")
        except ImportError:
            print(f"   FAIL {name}: Not installed")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n   Missing packages: {', '.join(missing_packages)}")
        print("   Install with: pip install " + " ".join(missing_packages))
        return False
    else:
        print("   PASS All dependencies installed")
        return True

def check_application_compatibility():
    """Check if the application can run"""
    print("\nApplication Compatibility Check")
    
    try:
        # Test importing main modules
        sys.path.insert(0, "src")
        from core.class_counter import ClassCounter
        from core.worker_thread import WorkerThread
        from ui.main_window import MainWindow
        print("   PASS Core modules import successfully")
        
        # Test ClassCounter
        counter = ClassCounter(accumulate=True)
        counter.update_from_labels(["person", "car", "person"])
        counts = counter.get_counts()
        if counts["person"] == 2 and counts["car"] == 1:
            print("   PASS ClassCounter working correctly")
        else:
            print("   FAIL ClassCounter not working correctly")
            return False
        
        print("   PASS Application compatibility check passed")
        return True
        
    except Exception as e:
        print(f"   FAIL Application compatibility check failed: {e}")
        return False

def generate_compatibility_report():
    """Generate a comprehensive compatibility report"""
    print("=" * 60)
    print("YOLOv5 COMPATIBILITY CHECK REPORT")
    print("=" * 60)
    
    checks = [
        ("Python Version", check_python_version),
        ("PyTorch Compatibility", check_torch_compatibility),
        ("YOLOv5 Installation", check_yolov5_installation),
        ("Dependencies", check_dependencies),
        ("Model Loading", check_model_loading),
        ("Application Compatibility", check_application_compatibility)
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"   FAIL {name} check failed with error: {e}")
            results.append((name, False))
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"   {name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} checks passed")
    
    if passed == total:
        print("SUCCESS: All compatibility checks passed! Your setup is ready.")
    else:
        print("WARNING: Some compatibility issues found. Please address them before running the application.")
    
    return passed == total

def main():
    """Main compatibility check function"""
    print("Starting YOLOv5 compatibility check...")
    print("This will verify your setup for the Road Damage CV application.\n")
    
    try:
        success = generate_compatibility_report()
        return 0 if success else 1
    except Exception as e:
        print(f"\nFAIL: Compatibility check failed with error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
