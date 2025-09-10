"""
Main screen UI component for displaying video stream and performance metrics.
"""

import numpy as np
import cv2
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QImage, QPixmap, QPainter
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
