#!/usr/bin/env python3
"""
GPS Data Export Utility
Exports GPS data history to CSV and JSON formats
"""

import csv
import json
import time
from datetime import datetime
from pathlib import Path

def export_gps_to_csv(gps_history, filename="gps_data_export.csv"):
    """Export GPS history to CSV file"""
    if not gps_history:
        print("No GPS data to export")
        return False
    
    try:
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = [
                'timestamp', 'update_count', 'latitude', 'longitude', 'altitude',
                'date', 'time', 'utc', 'satellites', 'hdop', 'signal_quality'
            ]
            
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for gps_point in gps_history:
                # Convert timestamp to readable format
                if 'timestamp' in gps_point:
                    gps_point['timestamp'] = datetime.fromtimestamp(gps_point['timestamp']).isoformat()
                
                writer.writerow(gps_point)
        
        print(f"✓ GPS data exported to {filename}")
        return True
        
    except Exception as e:
        print(f"✗ Error exporting to CSV: {e}")
        return False

def export_gps_to_json(gps_history, filename="gps_data_export.json"):
    """Export GPS history to JSON file"""
    if not gps_history:
        print("No GPS data to export")
        return False
    
    try:
        # Convert timestamps to readable format
        export_data = []
        for gps_point in gps_history:
            point = gps_point.copy()
            if 'timestamp' in point:
                point['timestamp'] = datetime.fromtimestamp(point['timestamp']).isoformat()
            export_data.append(point)
        
        with open(filename, 'w', encoding='utf-8') as jsonfile:
            json.dump(export_data, jsonfile, indent=2, ensure_ascii=False)
        
        print(f"✓ GPS data exported to {filename}")
        return True
        
    except Exception as e:
        print(f"✗ Error exporting to JSON: {e}")
        return False

def analyze_gps_data(gps_history):
    """Analyze GPS data and provide statistics"""
    if not gps_history:
        print("No GPS data to analyze")
        return
    
    print("\nGPS Data Analysis")
    print("=" * 30)
    
    # Basic stats
    total_points = len(gps_history)
    print(f"Total GPS points: {total_points}")
    
    if total_points == 0:
        return
    
    # Time range
    timestamps = [point.get('timestamp', 0) for point in gps_history if 'timestamp' in point]
    if timestamps:
        start_time = min(timestamps)
        end_time = max(timestamps)
        duration = end_time - start_time
        print(f"Time range: {datetime.fromtimestamp(start_time).strftime('%H:%M:%S')} - {datetime.fromtimestamp(end_time).strftime('%H:%M:%S')}")
        print(f"Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
    
    # Location stats
    lats = [point.get('lat') for point in gps_history if point.get('lat') is not None]
    lons = [point.get('lon') for point in gps_history if point.get('lon') is not None]
    alts = [point.get('alt') for point in gps_history if point.get('alt') is not None]
    
    if lats and lons:
        print(f"Latitude range: {min(lats):.6f} to {max(lats):.6f}")
        print(f"Longitude range: {min(lons):.6f} to {max(lons):.6f}")
        print(f"Center: {sum(lats)/len(lats):.6f}, {sum(lons)/len(lons):.6f}")
    
    if alts:
        print(f"Altitude range: {min(alts):.1f}m to {max(alts):.1f}m")
        print(f"Average altitude: {sum(alts)/len(alts):.1f}m")
    
    # Signal quality stats
    signal_qualities = [point.get('signal_quality', 'Unknown') for point in gps_history]
    quality_counts = {}
    for quality in signal_qualities:
        quality_counts[quality] = quality_counts.get(quality, 0) + 1
    
    print(f"\nSignal Quality Distribution:")
    for quality, count in quality_counts.items():
        percentage = (count / total_points) * 100
        print(f"  {quality}: {count} points ({percentage:.1f}%)")
    
    # Satellite stats
    satellites = [point.get('satellites') for point in gps_history if point.get('satellites') is not None]
    if satellites:
        print(f"\nSatellite Stats:")
        print(f"  Average: {sum(satellites)/len(satellites):.1f}")
        print(f"  Range: {min(satellites)} to {max(satellites)}")
    
    # HDOP stats
    hdops = [point.get('hdop') for point in gps_history if point.get('hdop') is not None]
    if hdops:
        print(f"\nHDOP Stats:")
        print(f"  Average: {sum(hdops)/len(hdops):.2f}")
        print(f"  Range: {min(hdops):.2f} to {max(hdops):.2f}")

def main():
    """Main function for GPS data export"""
    print("GPS Data Export Utility")
    print("=" * 30)
    
    # This would typically be called from the worker thread
    # For now, we'll create a sample for demonstration
    sample_gps_data = [
        {
            'timestamp': time.time() - 60,
            'update_count': 1,
            'lat': -5.138019,
            'lon': 119.480655,
            'alt': 111.1,
            'date': '09/10/2025',
            'time': '01:57:38',
            'utc': '17:57:38',
            'satellites': 4,
            'hdop': 12.05,
            'signal_quality': 'Fair'
        },
        {
            'timestamp': time.time() - 30,
            'update_count': 2,
            'lat': -5.138020,
            'lon': 119.480656,
            'alt': 111.2,
            'date': '09/10/2025',
            'time': '01:58:08',
            'utc': '17:58:08',
            'satellites': 5,
            'hdop': 8.32,
            'signal_quality': 'Good'
        },
        {
            'timestamp': time.time(),
            'update_count': 3,
            'lat': -5.138021,
            'lon': 119.480657,
            'alt': 111.3,
            'date': '09/10/2025',
            'time': '01:58:38',
            'utc': '17:58:38',
            'satellites': 6,
            'hdop': 3.45,
            'signal_quality': 'Excellent'
        }
    ]
    
    print("Sample GPS data created for demonstration")
    
    # Analyze the data
    analyze_gps_data(sample_gps_data)
    
    # Export to CSV
    export_gps_to_csv(sample_gps_data, "sample_gps_data.csv")
    
    # Export to JSON
    export_gps_to_json(sample_gps_data, "sample_gps_data.json")
    
    print(f"\n✓ Export completed!")
    print(f"Files created:")
    print(f"  - sample_gps_data.csv")
    print(f"  - sample_gps_data.json")

if __name__ == "__main__":
    main()
