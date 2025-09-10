"""
Real-time performance chart widget for monitoring system metrics.
"""

import time
from datetime import datetime
from PySide6.QtCore import Qt
from PySide6.QtGui import QPainter
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PySide6.QtCharts import QChart, QChartView, QLineSeries, QValueAxis


class PerformanceChart(QWidget):
    """Real-time performance chart widget"""

    def __init__(self, title: str, y_max: float = 100.0, y_min: float = 0.0):
        super().__init__()
        self.y_max = y_max
        self.y_min = y_min

        # Create chart
        self.chart = QChart()
        self.chart.setTitle(title)
        self.chart.setAnimationOptions(QChart.SeriesAnimations)
        self.chart.legend().hide()

        # Create series for data
        self.series = QLineSeries()
        self.series.setName(title)
        self.chart.addSeries(self.series)

        # Setup axes
        self.axis_x = QValueAxis()
        self.axis_x.setRange(0, 60)  # Show last 60 seconds
        self.axis_x.setTitleText("Time (s)")
        self.axis_x.setLabelsVisible(True)
        self.axis_x.setLabelFormat("%.0f")
        self.axis_x.setTickCount(7)  # ticks at ~0,10,20,30,40,50,60
        self.axis_x.setMinorTickCount(1)
        self.axis_x.setGridLineVisible(True)

        self.axis_y = QValueAxis()
        self.axis_y.setRange(y_min, y_max)
        self.axis_y.setTitleText("Value")
        self.axis_y.setLabelsVisible(True)
        # Use integer labels for percentages and FPS, one decimal is fine too
        self.axis_y.setLabelFormat("%.0f")
        self.axis_y.setTickCount(6)
        self.axis_y.setMinorTickCount(1)
        self.axis_y.setGridLineVisible(True)

        self.chart.addAxis(self.axis_x, Qt.AlignBottom)
        self.chart.addAxis(self.axis_y, Qt.AlignLeft)

        self.series.attachAxis(self.axis_x)
        self.series.attachAxis(self.axis_y)

        # Create chart view
        self.chart_view = QChartView(self.chart)
        self.chart_view.setRenderHint(QPainter.Antialiasing)

        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.chart_view)
        # Add a small label to show current wall-clock time and value
        self.value_label = QLabel("--:--:--  |  --")
        self.value_label.setAlignment(Qt.AlignRight)
        layout.addWidget(self.value_label)
        self.setLayout(layout)

        # Data storage
        self.data_points = []
        self.max_points = 60
        self.start_time = time.time()

    def update_value(self, value: float):
        """Add new data point to chart"""
        current_time = time.time() - self.start_time

        # Add new point
        self.data_points.append((current_time, value))

        # Keep only last max_points
        if len(self.data_points) > self.max_points:
            self.data_points = self.data_points[-self.max_points :]

        # Update chart
        self.series.clear()
        for t, v in self.data_points:
            self.series.append(t, v)

        # Keep X-axis window to the last 60 seconds
        if self.data_points:
            latest_t = self.data_points[-1][0]
            window_start = max(0.0, latest_t - 60.0)
            self.axis_x.setRange(window_start, max(window_start + 60.0, latest_t))

        # Auto-adjust Y axis if needed
        if value > self.y_max * 0.9:
            self.y_max = value * 1.1
            self.axis_y.setRange(self.y_min, self.y_max)
        elif value < self.y_min * 1.1:
            self.y_min = max(0, value * 0.9)
            self.axis_y.setRange(self.y_min, self.y_max)

        # Update label with wall clock time and current value
        try:
            now_str = datetime.now().strftime('%H:%M:%S')
            self.value_label.setText(f"{now_str}  |  {value:.1f}")
        except Exception:
            pass
