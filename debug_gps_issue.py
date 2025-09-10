#!/usr/bin/env python3
"""
Debug script to identify GPS data issues
"""

import time
import sys
from pathlib import Path

# Try to import pyserial with explicit handling
SERIAL_AVAILABLE = False
serial = None

try:
    # First try to import pyserial directly
    import serial
    # Verify it's the correct module
    if hasattr(serial, 'Serial') and hasattr(serial, 'SerialException'):
        SERIAL_AVAILABLE = True
        print("✓ pyserial imported successfully")
    else:
        print("✗ Wrong serial module detected")
        serial = None
except ImportError:
    print("✗ pyserial not found")

# If still not available, try alternative import methods
if not SERIAL_AVAILABLE:
    try:
        # Try importing from pyserial package
        from serial import Serial, SerialException
        # Create a mock serial module
        class MockSerial:
            Serial = Serial
            SerialException = SerialException
        serial = MockSerial()
        SERIAL_AVAILABLE = True
        print("✓ pyserial imported via alternative method")
    except ImportError:
        print("✗ Could not import pyserial via alternative method")

if not SERIAL_AVAILABLE:
    print("\n" + "="*60)
    print("SERIAL IMPORT FAILED")
    print("="*60)
    print("To fix this issue, try these steps:")
    print("1. pip uninstall serial")
    print("2. pip uninstall pyserial") 
    print("3. pip install pyserial")
    print("4. python -c \"import serial; print(serial.__version__)\"")
    print("\nIf that doesn't work, try:")
    print("pip install --force-reinstall pyserial")
    print("="*60)
    sys.exit(1)

def test_serial_connection(com_port, baud_rate):
    """Test basic serial connection"""
    print(f"Testing serial connection on {com_port} at {baud_rate} baud...")
    
    try:
        ser = serial.Serial(com_port, baud_rate, timeout=2)
        print(f"✓ Serial connection successful")
        return ser
    except serial.SerialException as e:
        print(f"✗ Serial connection failed: {e}")
        return None
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return None

def test_arduino_output(ser, duration=10):
    """Test Arduino output format"""
    print(f"\nReading Arduino output for {duration} seconds...")
    print("Expected format: 'Latitude: 12.345678', 'Longitude: 98.765432', etc.")
    print("-" * 60)
    
    start_time = time.time()
    received_data = {
        'latitude': False,
        'longitude': False,
        'altitude': False,
        'date': False,
        'time': False
    }
    
    while time.time() - start_time < duration:
        try:
            line = ser.readline().decode(errors="ignore").strip()
            if line:
                print(f"Received: {line}")
                
                # Check for expected Arduino output format
                if line.startswith("Latitude: "):
                    received_data['latitude'] = True
                elif line.startswith("Longitude: "):
                    received_data['longitude'] = True
                elif line.startswith("Altitude: "):
                    received_data['altitude'] = True
                elif line.startswith("Date: "):
                    received_data['date'] = True
                elif line.startswith("Time Local (WITA): ") or line.startswith("Time UTC: "):
                    received_data['time'] = True
                    
        except Exception as e:
            print(f"Error reading line: {e}")
            continue
    
    print("-" * 60)
    print("Data reception summary:")
    for key, received in received_data.items():
        status = "✓" if received else "✗"
        print(f"  {status} {key.capitalize()}")
    
    return received_data

def test_app_config():
    """Test app configuration"""
    print("\nTesting app configuration...")
    
    # Check if config file exists
    config_path = Path("src/config/settings.py")
    if config_path.exists():
        print("✓ Config file found")
        
        # Read default values
        with open(config_path, 'r') as f:
            content = f.read()
            
        if 'DEFAULT_COM_PORT' in content:
            print("✓ DEFAULT_COM_PORT found in config")
        if 'DEFAULT_BAUD' in content:
            print("✓ DEFAULT_BAUD found in config")
    else:
        print("✗ Config file not found")

def main():
    print("GPS Debug Script")
    print("=" * 50)
    
    # Get user input
    com_port = input("Enter COM port (default COM8): ").strip() or "COM8"
    baud_rate = input("Enter baud rate (default 9600): ").strip() or "9600"
    
    try:
        baud_rate = int(baud_rate)
    except ValueError:
        print("Invalid baud rate, using 9600")
        baud_rate = 9600
    
    # Test 1: Serial connection
    ser = test_serial_connection(com_port, baud_rate)
    if not ser:
        print("\n❌ Cannot proceed without serial connection")
        return
    
    # Test 2: Arduino output
    received_data = test_arduino_output(ser, 10)
    
    # Test 3: App configuration
    test_app_config()
    
    # Summary
    print("\n" + "=" * 50)
    print("DIAGNOSIS SUMMARY")
    print("=" * 50)
    
    if all(received_data.values()):
        print("✓ Arduino is sending data in correct format")
        print("✓ GPS module appears to be working")
        print("\n🔍 Next steps:")
        print("  1. Check if the app is using the same COM port and baud rate")
        print("  2. Check if the GPS thread is starting in the app")
        print("  3. Check for serial port conflicts (close Arduino IDE)")
    else:
        missing = [k for k, v in received_data.items() if not v]
        print(f"⚠ Arduino is not sending all expected data")
        print(f"   Missing: {', '.join(missing)}")
        print("\n🔍 Possible causes:")
        print("  1. GPS module not locked to satellites")
        print("  2. Arduino code not running")
        print("  3. Wrong COM port or baud rate")
        print("  4. GPS antenna issues")
    
    # Close serial connection
    try:
        ser.close()
        print(f"\n✓ Serial connection closed")
    except:
        pass

if __name__ == "__main__":
    main()
