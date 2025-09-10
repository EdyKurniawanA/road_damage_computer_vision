"""
Main screen UI component for displaying video stream and performance metrics.
"""

import numpy as np
import cv2
from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QFont, QImage, QPixmap, QPainter, QDesktopServices
from PySide6.QtWidgets import (
    QWidget,
    QLabel,
    QVBoxLayout,
    QHBoxLayout,
    QListWidget,
    QListWidgetItem,
    QPushButton,
)

from PySide6.QtWidgets import QGridLayout


class MainScreen(QWidget):
    """Main screen for displaying video stream and performance metrics"""

    def __init__(self):
        super().__init__()
        self._setup_ui()

    def _setup_ui(self):
        """Setup the UI components"""
        # Left: video placeholder
        self.video_label = QLabel("Video Stream")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setFixedHeight(480)
        self.video_label.setStyleSheet(
            "background-color: #202020; color: #CCCCCC; border: 1px solid #404040;"
        )

        # Right: Performance charts and class counts
        right_col = QVBoxLayout()

        # Performance section (simple monitoring)
        perf_label = QLabel("Performance Metrics")
        perf_label.setFont(QFont("Segoe UI", 12, QFont.Bold))
        perf_label.setAlignment(Qt.AlignCenter)
        right_col.addWidget(perf_label)

        grid = QGridLayout()
        row = 0
        # FPS
        grid.addWidget(QLabel("FPS:"), row, 0)
        self.fps_value = QLabel("-")
        self.fps_value.setAlignment(Qt.AlignRight)
        grid.addWidget(self.fps_value, row, 1)
        row += 1
        # CPU
        grid.addWidget(QLabel("CPU:"), row, 0)
        self.cpu_value = QLabel("-")
        self.cpu_value.setAlignment(Qt.AlignRight)
        grid.addWidget(self.cpu_value, row, 1)
        row += 1
        # Memory
        grid.addWidget(QLabel("Memory:"), row, 0)
        self.mem_value = QLabel("-")
        self.mem_value.setAlignment(Qt.AlignRight)
        grid.addWidget(self.mem_value, row, 1)
        row += 1
        # GPU
        grid.addWidget(QLabel("GPU:"), row, 0)
        self.gpu_value = QLabel("-")
        self.gpu_value.setAlignment(Qt.AlignRight)
        grid.addWidget(self.gpu_value, row, 1)
        row += 1

        right_col.addLayout(grid)

        # Class counts section
        class_label = QLabel("Class Counts")
        class_label.setFont(QFont("Segoe UI", 12, QFont.Bold))
        class_label.setAlignment(Qt.AlignCenter)
        right_col.addWidget(class_label)

        self.class_list = QListWidget()
        self.class_list.addItem(QListWidgetItem("person: 0"))
        right_col.addWidget(self.class_list)

        # GPS data section
        gps_label = QLabel("GPS Data")
        gps_label.setFont(QFont("Segoe UI", 12, QFont.Bold))
        gps_label.setAlignment(Qt.AlignCenter)
        right_col.addWidget(gps_label)
        
        # GPS status indicator
        self.gps_status = QLabel("Status: Disconnected")
        self.gps_status.setStyleSheet("color: #ff6b6b; font-weight: bold;")
        self.gps_status.setAlignment(Qt.AlignCenter)
        right_col.addWidget(self.gps_status)
        
        # GPS signal quality and update info
        self.gps_info = QLabel("Signal: Unknown | Updates: 0")
        self.gps_info.setStyleSheet("color: #888888; font-size: 10px;")
        self.gps_info.setAlignment(Qt.AlignCenter)
        right_col.addWidget(self.gps_info)

        # GPS data grid
        gps_grid = QGridLayout()
        gps_row = 0
        
        # Latitude
        gps_grid.addWidget(QLabel("Latitude:"), gps_row, 0)
        self.lat_value = QLabel("-")
        self.lat_value.setAlignment(Qt.AlignRight)
        gps_grid.addWidget(self.lat_value, gps_row, 1)
        gps_row += 1
        
        # Longitude
        gps_grid.addWidget(QLabel("Longitude:"), gps_row, 0)
        self.lon_value = QLabel("-")
        self.lon_value.setAlignment(Qt.AlignRight)
        gps_grid.addWidget(self.lon_value, gps_row, 1)
        gps_row += 1
        
        # Altitude
        gps_grid.addWidget(QLabel("Altitude:"), gps_row, 0)
        self.alt_value = QLabel("-")
        self.alt_value.setAlignment(Qt.AlignRight)
        gps_grid.addWidget(self.alt_value, gps_row, 1)
        gps_row += 1
        
        # Date
        gps_grid.addWidget(QLabel("Date:"), gps_row, 0)
        self.date_value = QLabel("-")
        self.date_value.setAlignment(Qt.AlignRight)
        gps_grid.addWidget(self.date_value, gps_row, 1)
        gps_row += 1
        
        # Time
        gps_grid.addWidget(QLabel("Time:"), gps_row, 0)
        self.time_value = QLabel("-")
        self.time_value.setAlignment(Qt.AlignRight)
        gps_grid.addWidget(self.time_value, gps_row, 1)
        gps_row += 1
        
        # Satellites
        gps_grid.addWidget(QLabel("Satellites:"), gps_row, 0)
        self.satellites_value = QLabel("-")
        self.satellites_value.setAlignment(Qt.AlignRight)
        gps_grid.addWidget(self.satellites_value, gps_row, 1)
        gps_row += 1
        
        # HDOP
        gps_grid.addWidget(QLabel("HDOP:"), gps_row, 0)
        self.hdop_value = QLabel("-")
        self.hdop_value.setAlignment(Qt.AlignRight)
        gps_grid.addWidget(self.hdop_value, gps_row, 1)
        gps_row += 1
        
        # Last Update
        gps_grid.addWidget(QLabel("Last Update:"), gps_row, 0)
        self.last_update_value = QLabel("-")
        self.last_update_value.setAlignment(Qt.AlignRight)
        gps_grid.addWidget(self.last_update_value, gps_row, 1)
        gps_row += 1
        
        # Google Maps Link
        gps_grid.addWidget(QLabel("Google Maps:"), gps_row, 0)
        self.google_maps_link = QLabel("No location data")
        self.google_maps_link.setAlignment(Qt.AlignRight)
        self.google_maps_link.setStyleSheet("color: #1976d2; text-decoration: underline; cursor: pointer;")
        self.google_maps_link.setWordWrap(True)
        self.google_maps_link.mousePressEvent = self._open_google_maps
        gps_grid.addWidget(self.google_maps_link, gps_row, 1)
        gps_row += 1

        right_col.addLayout(gps_grid)

        right_col.addStretch(1)

        # Add stop button at bottom
        self.stop_btn = QPushButton("Stop Detection")
        self.stop_btn.setStyleSheet(
            "background-color: #d32f2f; color: white; padding: 8px;"
        )
        right_col.addWidget(self.stop_btn)

        # Main layout
        top = QHBoxLayout()
        top.addWidget(self.video_label, 3)
        top.addLayout(right_col, 1)

        self.setLayout(top)

    def update_metrics(
        self,
        fps: float = None,
        cpu_percent: float = None,
        cpu_hz: float = None,
        mem_used_gb: float = None,
        mem_percent: float = None,
        gpu_percent: float = None,
        gpu_temp_c: float = None,
    ):
        """Update simple monitoring labels with new values"""
        if fps is not None:
            self.fps_value.setText(f"{fps:.1f} fps")
        if cpu_percent is not None or cpu_hz is not None:
            cpu_text = []
            if cpu_percent is not None:
                cpu_text.append(f"{cpu_percent:.0f}%")
            if cpu_hz is not None and cpu_hz > 0:
                if cpu_hz >= 1e9:
                    cpu_text.append(f"{cpu_hz/1e9:.2f} GHz")
                elif cpu_hz >= 1e6:
                    cpu_text.append(f"{cpu_hz/1e6:.0f} MHz")
                else:
                    cpu_text.append(f"{cpu_hz:.0f} Hz")
            self.cpu_value.setText("  |  ".join(cpu_text) if cpu_text else "-")
        if mem_used_gb is not None or mem_percent is not None:
            mem_text = []
            if mem_used_gb is not None:
                mem_text.append(f"{mem_used_gb:.2f} GB")
            if mem_percent is not None:
                mem_text.append(f"{mem_percent:.0f}%")
            self.mem_value.setText("  |  ".join(mem_text) if mem_text else "-")
        if gpu_percent is not None or gpu_temp_c is not None:
            gpu_text = []
            if gpu_percent is not None:
                gpu_text.append(f"{gpu_percent:.0f}%")
            if gpu_temp_c is not None:
                gpu_text.append(f"{gpu_temp_c:.0f} °C")
            self.gpu_value.setText("  |  ".join(gpu_text) if gpu_text else "-")

    def update_class_counts(self, counts: dict):
        """Update the class counts display"""
        self.class_list.clear()
        for name, value in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
            self.class_list.addItem(QListWidgetItem(f"{name}: {value}"))

    def update_gps_data(self, gps_data: dict):
        """Update the GPS data display"""
        # Debug: Print GPS data received by UI
        print(f"GPS UI Received: {gps_data}")
        
        if not gps_data:
            self.gps_status.setText("Status: No Data")
            self.gps_status.setStyleSheet("color: #ff6b6b; font-weight: bold;")
            self.gps_info.setText("Signal: Unknown | Updates: 0")
            return
        
        # Check if we have valid GPS data
        has_location = "lat" in gps_data and "lon" in gps_data and gps_data["lat"] is not None and gps_data["lon"] is not None
        has_altitude = "alt" in gps_data and gps_data["alt"] is not None
        has_time = "time" in gps_data or "utc" in gps_data
        
        # Update status and signal quality
        signal_quality = gps_data.get("signal_quality", "Unknown")
        update_count = gps_data.get("update_count", 0)
        
        if has_location:
            self.gps_status.setText("Status: Connected")
            self.gps_status.setStyleSheet("color: #4caf50; font-weight: bold;")
        else:
            self.gps_status.setText("Status: No Fix")
            self.gps_status.setStyleSheet("color: #ff9800; font-weight: bold;")
        
        # Update signal quality and update count
        self.gps_info.setText(f"Signal: {signal_quality} | Updates: {update_count}")
        
        # Color code signal quality
        if signal_quality == "Excellent":
            self.gps_info.setStyleSheet("color: #4caf50; font-size: 10px;")
        elif signal_quality == "Good":
            self.gps_info.setStyleSheet("color: #8bc34a; font-size: 10px;")
        elif signal_quality == "Fair":
            self.gps_info.setStyleSheet("color: #ff9800; font-size: 10px;")
        elif signal_quality == "Poor":
            self.gps_info.setStyleSheet("color: #f44336; font-size: 10px;")
        else:
            self.gps_info.setStyleSheet("color: #888888; font-size: 10px;")
            
        # Update latitude
        if "lat" in gps_data and gps_data["lat"] is not None:
            self.lat_value.setText(f"{gps_data['lat']:.6f}")
        else:
            self.lat_value.setText("-")
            
        # Update longitude
        if "lon" in gps_data and gps_data["lon"] is not None:
            self.lon_value.setText(f"{gps_data['lon']:.6f}")
        else:
            self.lon_value.setText("-")
            
        # Update altitude
        if "alt" in gps_data and gps_data["alt"] is not None:
            self.alt_value.setText(f"{gps_data['alt']:.1f} m")
        else:
            self.alt_value.setText("-")
            
        # Update date
        if "date" in gps_data and gps_data["date"]:
            self.date_value.setText(gps_data["date"])
        else:
            self.date_value.setText("-")
            
        # Update time
        if "time" in gps_data and gps_data["time"]:
            self.time_value.setText(gps_data["time"])
        elif "utc" in gps_data and gps_data["utc"]:
            self.time_value.setText(gps_data["utc"])
        else:
            self.time_value.setText("-")
            
        # Update satellites
        if "satellites" in gps_data and gps_data["satellites"] is not None:
            self.satellites_value.setText(str(gps_data["satellites"]))
        else:
            self.satellites_value.setText("-")
            
        # Update HDOP
        if "hdop" in gps_data and gps_data["hdop"] is not None:
            self.hdop_value.setText(f"{gps_data['hdop']:.2f}")
        else:
            self.hdop_value.setText("-")
            
        # Update last update time (time since GPS data was received)
        if "timestamp" in gps_data and gps_data["timestamp"] is not None:
            import time
            current_time = time.time()
            last_update = gps_data["timestamp"]
            time_diff = current_time - last_update
            if time_diff < 60:
                self.last_update_value.setText(f"{time_diff:.1f}s ago")
            elif time_diff < 3600:
                self.last_update_value.setText(f"{time_diff/60:.1f}m ago")
            else:
                self.last_update_value.setText(f"{time_diff/3600:.1f}h ago")
        else:
            self.last_update_value.setText("-")
            
        # Update Google Maps link
        self._update_google_maps_link(gps_data)

    def _update_google_maps_link(self, gps_data: dict):
        """Update the Google Maps link with current GPS coordinates"""
        if "lat" in gps_data and "lon" in gps_data and gps_data["lat"] is not None and gps_data["lon"] is not None:
            lat = gps_data["lat"]
            lon = gps_data["lon"]
            
            # Create Google Maps URL with GPS coordinates
            # Format: https://www.google.com/maps?q=lat,lon
            google_maps_url = f"https://www.google.com/maps?q={lat},{lon}"
            
            # Store the URL for clicking
            self.google_maps_url = google_maps_url
            
            # Display clickable text
            self.google_maps_link.setText("Click to open in Google Maps")
            self.google_maps_link.setStyleSheet("color: #1976d2; text-decoration: underline; cursor: pointer;")
        else:
            self.google_maps_link.setText("No location data")
            self.google_maps_link.setStyleSheet("color: #888888; text-decoration: none; cursor: default;")
            self.google_maps_url = None

    def _open_google_maps(self, event):
        """Open Google Maps with current GPS coordinates"""
        if hasattr(self, 'google_maps_url') and self.google_maps_url:
            try:
                # Open the URL in the default web browser
                QDesktopServices.openUrl(QUrl(self.google_maps_url))
                print(f"Opening Google Maps: {self.google_maps_url}")
            except Exception as e:
                print(f"Error opening Google Maps: {e}")
        else:
            print("No GPS location data available for Google Maps")

    def add_stop_button(self, callback):
        """Add callback for stop button"""
        self.stop_btn.clicked.connect(callback)

    def set_frame(self, frame_bgr: np.ndarray):
        """Set the video frame to display"""
        if frame_bgr is None:
            return
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w

        qimg = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(qimg).scaled(
            self.video_label.width(),
            self.video_label.height(),
            Qt.KeepAspectRatio,
            Qt.FastTransformation,
        )
        self.video_label.setPixmap(pix)
