"""
Main entry point for the Road Damage CV application.
"""

import sys
import signal
from PySide6.QtWidgets import QApplication

from src.ui.main_window import MainWindow


def run():
    """Run the application"""
    app = QApplication(sys.argv)
    
    # Set up signal handling for graceful shutdown
    def signal_handler(signum, frame):
        print(f"\nReceived signal {signum}, shutting down gracefully...")
        app.quit()
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    window = MainWindow()
    window.show()
    
    try:
        sys.exit(app.exec())
    except KeyboardInterrupt:
        print("\nKeyboard interrupt, shutting down...")
    except Exception as e:
        print(f"Application error: {e}")
    finally:
        print("Application terminated")


if __name__ == "__main__":
    run()
