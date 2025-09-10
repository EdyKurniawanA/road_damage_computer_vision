#!/usr/bin/env python3
"""
Diagnostic script to identify serial import issues
"""

import sys
import subprocess
import importlib

def check_python_environment():
    """Check Python environment and installed packages"""
    print("Python Environment Diagnosis")
    print("=" * 50)
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    print(f"Python path: {sys.path[:3]}...")  # Show first 3 paths
    
def check_installed_packages():
    """Check what serial-related packages are installed"""
    print("\nInstalled Serial Packages:")
    print("-" * 30)
    
    try:
        result = subprocess.run([sys.executable, "-m", "pip", "list"], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.split('\n')
            serial_packages = [line for line in lines if 'serial' in line.lower()]
            if serial_packages:
                for package in serial_packages:
                    print(f"  {package}")
            else:
                print("  No serial packages found")
        else:
            print("  Could not check installed packages")
    except Exception as e:
        print(f"  Error checking packages: {e}")

def test_serial_imports():
    """Test different ways to import serial"""
    print("\nSerial Import Tests:")
    print("-" * 20)
    
    # Test 1: Direct import
    print("1. Testing: import serial")
    try:
        import serial
        print(f"   ✓ Success - Module: {serial.__file__ if hasattr(serial, '__file__') else 'built-in'}")
        print(f"   ✓ Has Serial: {hasattr(serial, 'Serial')}")
        print(f"   ✓ Has SerialException: {hasattr(serial, 'SerialException')}")
        if hasattr(serial, '__version__'):
            print(f"   ✓ Version: {serial.__version__}")
        return serial
    except ImportError as e:
        print(f"   ✗ Failed: {e}")
    
    # Test 2: From serial import
    print("\n2. Testing: from serial import Serial, SerialException")
    try:
        from serial import Serial, SerialException
        print("   ✓ Success - Direct import works")
        return True
    except ImportError as e:
        print(f"   ✗ Failed: {e}")
    
    # Test 3: Try pyserial specifically
    print("\n3. Testing: import pyserial")
    try:
        import pyserial
        print("   ✓ pyserial module found")
        return True
    except ImportError:
        print("   ✗ pyserial module not found")
    
    return False

def suggest_fixes():
    """Suggest fixes based on the diagnosis"""
    print("\nSuggested Fixes:")
    print("-" * 15)
    
    print("1. Clean installation:")
    print("   pip uninstall serial pyserial")
    print("   pip install pyserial")
    
    print("\n2. Force reinstall:")
    print("   pip install --force-reinstall pyserial")
    
    print("\n3. Check for conflicts:")
    print("   pip list | findstr serial")
    
    print("\n4. Try different Python environment:")
    print("   python -m venv test_env")
    print("   test_env\\Scripts\\activate")
    print("   pip install pyserial")
    
    print("\n5. Alternative: Use the simple debug script:")
    print("   python debug_gps_simple.py")

def main():
    print("Serial Import Diagnostic Tool")
    print("=" * 50)
    
    check_python_environment()
    check_installed_packages()
    
    serial_available = test_serial_imports()
    
    if serial_available:
        print("\n✓ Serial module is working correctly!")
        print("You can now run: python debug_gps_issue.py")
    else:
        suggest_fixes()

if __name__ == "__main__":
    main()
