"""
YOLO-Detector-Pro Entry Point
"""

import sys
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QIcon
from main_window import MainWindow


def main():
    """Application entry point"""
    # Set AppUserModelID for Windows taskbar icon
    import ctypes
    myappid = 'antigravity.yolodetector.pro.1.0'  # arbitrary string
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)

    app = QApplication(sys.argv)
    app.setApplicationName("YOLO-Detector-Pro")
    app.setOrganizationName("YOLO-Detector")
    
    # Set app icon explicitly
    import os
    icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "icons", "app_icon.ico")
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))
    
    # Create and show main window
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
