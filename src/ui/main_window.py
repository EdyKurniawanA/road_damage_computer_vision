"""
Main application window that manages the overall UI and coordinates between components.
"""

import sys
import time
import threading
import tracemalloc
import gc
from typing import Optional

import numpy as np
import psutil
import torch
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QWidget, QVBoxLayout, QStackedWidget

from .home_screen import HomeScreen
from .main_screen import MainScreen
from ..core.worker_thread import WorkerThread


class MainWindow(QWidget):
    """Main application window"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Drone YOLO")
        self.resize(1400, 800)

        # Setup UI
        self._setup_ui()
        
        # Store chosen config
        self.config = {}
        self.worker_thread: Optional[WorkerThread] = None
        
        # Performance monitoring
        self.cpu_percent = 0.0
        self.cpu_hz = 0.0
        self.mem_percent = 0.0
        self.mem_used_gb = 0.0
        self.gpu_percent = 0.0
        self.gpu_temp_c = 0.0
        
        # UI update timer - slower to prevent lag
        self.timer = QTimer(self)
        self.timer.setInterval(100)  # 10 Hz UI refresh for stability
        self.timer.timeout.connect(self._tick)
        
        # Performance monitoring timer
        self.perf_timer = QTimer(self)
        self.perf_timer.setInterval(1000)  # 1 Hz performance monitoring
        self.perf_timer.timeout.connect(self._update_performance)
        
        # Start memory tracing
        tracemalloc.start()

    def _setup_ui(self):
        """Setup the main UI components"""
        self.stack = QStackedWidget()
        self.home = HomeScreen(self._start_clicked)
        self.main = MainScreen()
        self.stack.addWidget(self.home)
        self.stack.addWidget(self.main)

        layout = QVBoxLayout()
        layout.addWidget(self.stack)
        self.setLayout(layout)

    def _start_clicked(self, config: dict):
        """Start detection with proper thread management"""
        self.config = config
        
        # Create and start worker thread
        self.worker_thread = WorkerThread(config)
        self.worker_thread.frame_ready.connect(self._on_frame_ready)
        self.worker_thread.status_update.connect(self._on_status_update)
        self.worker_thread.start()
        
        # Switch to main screen
        self.stack.setCurrentWidget(self.main)
        self.main.add_stop_button(self._stop_detection)
        
        # Start timers
        self.timer.start()
        self.perf_timer.start()

    def _stop_detection(self):
        """Stop detection with graceful cleanup"""
        if self.worker_thread is not None:
            self.worker_thread.stop()
            self.worker_thread = None
            
        self.timer.stop()
        self.perf_timer.stop()
        self.stack.setCurrentWidget(self.home)

    def _on_frame_ready(self, frame: np.ndarray, counts: dict):
        """Handle frame from worker thread"""
        self.main.set_frame(frame)
        self.main.update_class_counts(counts)

    def _on_status_update(self, message: str):
        """Handle status updates from worker thread"""
        print(f"Status: {message}")

    def _update_performance(self):
        """Update performance metrics"""
        try:
            vm = psutil.virtual_memory()
            self.cpu_percent = psutil.cpu_percent()
            self.mem_percent = vm.percent
            self.mem_used_gb = (vm.total - vm.available) / (1024 ** 3)

            # CPU frequency (Hz)
            try:
                freq = psutil.cpu_freq()
                self.cpu_hz = (freq.current * 1e6) if freq and freq.current else 0.0
            except Exception:
                self.cpu_hz = 0.0

            # GPU utilization and temperature (if NVML available)
            self.gpu_percent = 0.0
            self.gpu_temp_c = 0.0
            try:
                import pynvml
                pynvml.nvmlInit()
                h = pynvml.nvmlDeviceGetHandleByIndex(0)
                util = pynvml.nvmlDeviceGetUtilizationRates(h)
                self.gpu_percent = float(util.gpu)
                try:
                    self.gpu_temp_c = float(pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU))
                except Exception:
                    self.gpu_temp_c = 0.0
                pynvml.nvmlShutdown()
            except Exception:
                # Fallback: try torch cuda util if available
                try:
                    if torch.cuda.is_available():
                        # torch has no direct util/temperature; keep defaults
                        self.gpu_percent = 0.0
                        self.gpu_temp_c = 0.0
                except Exception:
                    pass
                
            # Get FPS from worker thread
            fps = 0.0
            if self.worker_thread:
                fps = self.worker_thread.fps
            
            self.main.update_metrics(
                fps=fps,
                cpu_percent=self.cpu_percent,
                cpu_hz=self.cpu_hz,
                mem_used_gb=self.mem_used_gb,
                mem_percent=self.mem_percent,
                gpu_percent=self.gpu_percent,
                gpu_temp_c=self.gpu_temp_c,
            )
            
        except Exception as e:
            print(f"Performance update error: {e}")

    def _tick(self):
        """Main UI update loop"""
        # This is now handled by signals from worker thread
        pass

    def closeEvent(self, event):
        """Handle application close with graceful cleanup"""
        print("Application closing...")
        
        # Stop detection if running
        if self.worker_thread is not None:
            print("Stopping detection...")
            self.worker_thread.stop()
            self.worker_thread = None
            
        # Stop timers
        self.timer.stop()
        self.perf_timer.stop()
        
        # Print memory usage
        try:
            current, peak = tracemalloc.get_traced_memory()
            print(f"Memory usage - Current: {current / 1024 / 1024:.1f} MB, Peak: {peak / 1024 / 1024:.1f} MB")
        except:
            pass
            
        # Print active threads
        try:
            active_threads = threading.active_count()
            print(f"Active threads: {active_threads}")
        except:
            pass
            
        # Force cleanup
        gc.collect()
        
        print("Application closed gracefully")
        super().closeEvent(event)
