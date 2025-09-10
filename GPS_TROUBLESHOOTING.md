# GPS Data Troubleshooting Guide

## Problem: No GPS data appears in the app despite GPS module locking satellites

## Root Causes Identified

1. **Silent Serial Connection Failure** - The app fails to connect to the serial port without proper error reporting
2. **Baud Rate Type Mismatch** - Configuration stores baud rate as string, but serial library expects integer
3. **Missing Error Logging** - No status updates when GPS connection fails
4. **No GPS Data Validation** - App doesn't check if GPS data is actually being received

## Fixes Applied

### 1. Enhanced GPS Connection Handling
- Added proper error handling and status reporting
- Fixed baud rate type conversion (string to integer)
- Added connection retry logic (5 attempts with 2-second delays)
- Added detailed status messages for debugging

### 2. Improved GPS Data Parsing
- Enhanced parsing logic to match Arduino output format exactly
- Added data validation and tracking
- Added debug output for troubleshooting
- Added timeout detection for data reception

### 3. Visual GPS Status Indicator
- Added GPS status indicator in the UI
- Color-coded status: Green (Connected), Orange (No Fix), Red (No Data)
- Real-time status updates

## Testing Steps

### Step 1: Run the Diagnostic Script
```bash
python debug_gps_issue.py
```

This will:
- Test serial connection to your Arduino
- Verify Arduino is sending data in correct format
- Check if GPS module is locked to satellites
- Provide detailed diagnosis

### Step 2: Check App Configuration
1. Open the app and go to the Home screen
2. Verify the COM port matches your Arduino (usually COM8)
3. Verify the baud rate matches your Arduino (usually 9600)
4. Click "Start Detection"

### Step 3: Monitor Status Messages
Look for these status messages in the console:
- `GPS: Attempting to connect to COM8 at 9600 baud...`
- `GPS: Connected to COM8` (success)
- `GPS: Connection attempt X failed: [error]` (failure)
- `GPS: No data received - check Arduino connection and GPS lock`

### Step 4: Check GPS Status in UI
The GPS section now shows:
- **Status: Connected** (Green) - GPS working properly
- **Status: No Fix** (Orange) - GPS connected but no satellite lock
- **Status: No Data** (Red) - No GPS data received

## Common Issues and Solutions

### Issue 1: "GPS: pyserial not available"
**Solution:** Install pyserial
```bash
pip install pyserial
```

### Issue 2: "GPS: Connection attempt X failed"
**Possible causes:**
- Wrong COM port (check Device Manager)
- Arduino not connected
- Another program using the port (close Arduino IDE)
- Wrong baud rate

**Solutions:**
1. Check Device Manager for correct COM port
2. Close Arduino IDE and any other serial programs
3. Verify baud rate matches Arduino code (9600)
4. Try different USB cable/port

### Issue 3: "GPS: No data received"
**Possible causes:**
- Arduino not running GPS code
- GPS module not locked to satellites
- Wrong baud rate
- Hardware connection issues

**Solutions:**
1. Upload the GPS code to Arduino
2. Wait for GPS lock (can take 1-2 minutes)
3. Check GPS antenna has clear sky view
4. Verify wiring connections

### Issue 4: "Status: No Fix" in UI
**Possible causes:**
- GPS module not locked to satellites
- Poor signal quality
- Indoor testing

**Solutions:**
1. Move to outdoor location with clear sky view
2. Wait longer for GPS lock (up to 2 minutes)
3. Check GPS antenna connection
4. Verify GPS module is working (use Arduino IDE Serial Monitor)

## Arduino Code Verification

Your Arduino code should output data in this format:
```
Latitude: 12.345678
Longitude: 98.765432
Altitude: 123.4 meters
Date: 12/25/2024
Time Local (WITA): 14:30:45
Time UTC: 06:30:45
```

## Debug Output

The app now prints debug information to help diagnose issues:
- `GPS Debug: [line]` - Shows raw data from Arduino
- `GPS Error reading line: [error]` - Shows parsing errors
- Status messages in the console

## Expected Behavior

1. **Startup:** App attempts to connect to GPS
2. **Connection:** Status shows "Connected" if successful
3. **Data Reception:** GPS data appears in the UI
4. **Status Updates:** Real-time status in GPS section

## Still Having Issues?

If GPS data still doesn't appear:

1. **Run the diagnostic script first** - This will identify the exact issue
2. **Check the console output** - Look for error messages
3. **Verify Arduino output** - Use Arduino IDE Serial Monitor to confirm data format
4. **Test with different COM port/baud rate** - Some systems use different settings
5. **Check hardware connections** - Ensure GPS module is properly wired

## Files Modified

- `src/core/worker_thread.py` - Enhanced GPS parsing and error handling
- `src/ui/main_screen.py` - Added GPS status indicator
- `debug_gps_issue.py` - New diagnostic script
- `GPS_TROUBLESHOOTING.md` - This troubleshooting guide
