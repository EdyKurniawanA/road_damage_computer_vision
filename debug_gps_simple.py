#!/usr/bin/env python3
"""
Simple GPS debug script that handles serial import issues
"""

import time
import sys
from pathlib import Path

def check_serial_availability():
    """Check if pyserial is available and working"""
    try:
        # Try to import pyserial
        import serial
        # Check if it has the required attributes
        if hasattr(serial, 'Serial') and hasattr(serial, 'SerialException'):
            print("✓ pyserial is available and working")
            return True
        else:
            print("✗ Wrong serial module imported")
            return False
    except ImportError:
        print("✗ pyserial not installed")
        return False

def test_gps_connection():
    """Test GPS connection with proper error handling"""
    print("GPS Connection Test")
    print("=" * 50)
    
    # Check serial availability
    if not check_serial_availability():
        print("\nTo fix this issue:")
        print("1. Install pyserial: pip install pyserial")
        print("2. If you have a conflicting 'serial' package, uninstall it:")
        print("   pip uninstall serial")
        print("3. Then reinstall pyserial: pip install pyserial")
        return
    
    # Import serial after checking
    import serial
    
    # Get user input
    com_port = input("Enter COM port (default COM8): ").strip() or "COM8"
    baud_rate = input("Enter baud rate (default 9600): ").strip() or "9600"
    
    try:
        baud_rate = int(baud_rate)
    except ValueError:
        print("Invalid baud rate, using 9600")
        baud_rate = 9600
    
    # Test serial connection
    print(f"\nTesting connection to {com_port} at {baud_rate} baud...")
    
    try:
        ser = serial.Serial(com_port, baud_rate, timeout=2)
        print("✓ Serial connection successful")
        
        # Test data reception
        print(f"\nReading data for 10 seconds...")
        print("Expected format: 'Latitude: 12.345678', 'Longitude: 98.765432', etc.")
        print("-" * 60)
        
        start_time = time.time()
        data_count = 0
        gps_data_received = False
        
        while time.time() - start_time < 10:
            try:
                line = ser.readline().decode(errors="ignore").strip()
                if line:
                    print(f"Received: {line}")
                    data_count += 1
                    
                    # Check for GPS data
                    if any(keyword in line for keyword in ["Latitude:", "Longitude:", "Altitude:", "Date:", "Time"]):
                        gps_data_received = True
                        
            except Exception as e:
                print(f"Error reading line: {e}")
                continue
        
        print("-" * 60)
        print(f"Data received: {data_count} lines")
        
        if gps_data_received:
            print("✓ GPS data detected in correct format")
            print("✓ Arduino and GPS module appear to be working")
        else:
            print("⚠ No GPS data detected")
            print("Possible causes:")
            print("  - GPS module not locked to satellites")
            print("  - Arduino code not running")
            print("  - Wrong COM port or baud rate")
            print("  - GPS antenna issues")
        
        ser.close()
        print("\n✓ Test completed")
        
    except serial.SerialException as e:
        print(f"✗ Serial connection failed: {e}")
        print("\nTroubleshooting:")
        print("  - Check if COM port is correct")
        print("  - Make sure Arduino is connected")
        print("  - Close Arduino IDE and other serial programs")
        print("  - Try different USB cable/port")
        
    except Exception as e:
        print(f"✗ Unexpected error: {e}")

if __name__ == "__main__":
    test_gps_connection()
