"""
YOLO-Detector-Pro - Main Window
Modern YOLO Object Detection GUI
"""

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFrame, QComboBox, QSlider,
    QDoubleSpinBox, QGroupBox, QButtonGroup, QStackedWidget,
    QFileDialog, QProgressBar, QMessageBox, QScrollArea,
    QLineEdit
)
from PySide6.QtCore import Qt, Signal, QSize, QTimer
from PySide6.QtGui import QIcon, QPixmap, QFont
import sys
import os

# Import custom modules
from core.yolo_worker import YOLOWorker
from core.detection_manager import DetectionManager
from widgets.dashboard_widget import DashboardWidget
from widgets.comparison_widget import ComparisonWidget
from utils.helper import (
    draw_detections, numpy_to_qpixmap, save_detection_result,
    save_batch_results, save_detection_json, get_available_cameras,
    get_available_models
)
from core.comparison_worker import ComparisonWorker


class StatusCard(QFrame):
    """Status display card for top bar"""
    def __init__(self, label_text, icon_name, parent=None):
        super().__init__(parent)
        self.setProperty("class", "status-card")
        
        # Main horizontal layout
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(15, 12, 15, 12)
        main_layout.setSpacing(12)
        
        # Left side: Large Icon
        self.icon_label = QLabel()
        icon_path = os.path.join("assets", "icons", f"{icon_name}.png")
        if os.path.exists(icon_path):
            pixmap = QPixmap(icon_path)
            # Scale icon to larger size (48x48)
            scaled_pixmap = pixmap.scaled(48, 48, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.icon_label.setPixmap(scaled_pixmap)
        self.icon_label.setAlignment(Qt.AlignCenter)
        self.icon_label.setFixedSize(48, 48)
        
        # Right side: Label + Value (vertical, centered)
        right_layout = QVBoxLayout()
        right_layout.setSpacing(6)
        right_layout.setAlignment(Qt.AlignCenter)
        
        # Label text (above value)
        self.label = QLabel(label_text)
        self.label.setProperty("class", "status-label")
        self.label.setAlignment(Qt.AlignCenter)
        
        # Value (below label)
        self.value = QLabel("0")
        self.value.setProperty("class", "status-value")
        self.value.setAlignment(Qt.AlignCenter)
        
        right_layout.addWidget(self.label)
        right_layout.addWidget(self.value)
        
        # Add to main layout
        main_layout.addWidget(self.icon_label)
        main_layout.addLayout(right_layout, 1)
        
        self.setLayout(main_layout)
    
    def set_value(self, value):
        """Update the status value with adaptive font size"""
        text = str(value)
        self.value.setText(text)
        self.value.setToolTip(text) # Show full text on hover
        
        # Adaptive font size based on length
        # Default is 32px defined in CSS
        if len(text) > 15:
            self.value.setStyleSheet("font-size: 14px; color: #1e293b; font-weight: bold;")
        elif len(text) > 10:
            self.value.setStyleSheet("font-size: 20px; color: #1e293b; font-weight: bold;")
        elif len(text) > 7:
            self.value.setStyleSheet("font-size: 24px; color: #1e293b; font-weight: bold;")
        else:
            self.value.setStyleSheet("font-size: 32px; color: #1e293b; font-weight: bold;")


class NavigationButton(QPushButton):
    """Custom navigation button for sidebar"""
    def __init__(self, icon_name, parent=None):
        super().__init__(parent)
        self.setProperty("class", "nav-button")
        self.setCheckable(True)
        self.setObjectName(f"{icon_name}Btn")
        self.setMinimumHeight(60)
        self.setIconSize(QSize(32, 32))
        
        # Load icon
        icon_path = os.path.join("assets", "icons", f"{icon_name}.png")
        if os.path.exists(icon_path):
            self.setIcon(QIcon(icon_path))
        
        # Set tooltip
        tooltips = {
            "image": "图片检测",
            "folder": "文件夹检测",
            "camera": "摄像头检测",
            "dashboard": "数据看板",
            "compare": "对比模式"
        }
        self.setToolTip(tooltips.get(icon_name, ""))


class CustomAdjuster(QWidget):
    """Custom adjustment widget with a value box and +/- buttons"""
    valueChanged = Signal(float)

    def __init__(self, value=0.0, min_val=0.0, max_val=1.0, step=0.01, decimals=2, compact=False, parent=None):
        super().__init__(parent)
        self.value = value
        self.min_val = min_val
        self.max_val = max_val
        self.step = step
        self.decimals = decimals

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(compact and 4 or 8)

        h = compact and 26 or 30
        self.setFixedHeight(h)

        # Value display box
        self.value_edit = QLineEdit(format(value, f".{decimals}f"))
        self.value_edit.setReadOnly(True)
        self.value_edit.setAlignment(Qt.AlignCenter)
        self.value_edit.setProperty("class", "adjuster-value")
        self.value_edit.setFixedWidth(compact and 45 or 70)
        self.value_edit.setFixedHeight(h)

        # Buttons container
        btn_container = QWidget()
        btn_layout = QHBoxLayout(btn_container)
        btn_layout.setContentsMargins(0, 0, 0, 0)
        btn_layout.setSpacing(compact and 1 or 2)

        self.minus_btn = QPushButton()
        self.minus_btn.setProperty("class", "adjuster-btn")
        self.minus_btn.setFixedSize(compact and 24 or 30, h)
        self.minus_btn.setCursor(Qt.PointingHandCursor)
        
        icon_minus_path = os.path.join("assets", "icons", "left-arrow.png")
        if os.path.exists(icon_minus_path):
            self.minus_btn.setIcon(QIcon(icon_minus_path))
            self.minus_btn.setIconSize(QSize(h-10, h-10))
        else:
            self.minus_btn.setText("◀")

        self.plus_btn = QPushButton()
        self.plus_btn.setProperty("class", "adjuster-btn")
        self.plus_btn.setFixedSize(compact and 24 or 30, h)
        self.plus_btn.setCursor(Qt.PointingHandCursor)
        
        icon_plus_path = os.path.join("assets", "icons", "right-arrow.png")
        if os.path.exists(icon_plus_path):
            self.plus_btn.setIcon(QIcon(icon_plus_path))
            self.plus_btn.setIconSize(QSize(h-10, h-10))
        else:
            self.plus_btn.setText("▶")

        btn_layout.addWidget(self.minus_btn)
        btn_layout.addWidget(self.plus_btn)

        layout.addWidget(self.value_edit)
        layout.addWidget(btn_container)
        if compact:
            layout.addStretch()

        # Connections
        self.minus_btn.clicked.connect(self.decrement)
        self.plus_btn.clicked.connect(self.increment)

    def increment(self):
        new_val = min(self.max_val, self.value + self.step)
        if new_val != self.value:
            self.value = new_val
            self.update_display()
            self.valueChanged.emit(self.value)

    def decrement(self):
        new_val = max(self.min_val, self.value - self.step)
        if new_val != self.value:
            self.value = new_val
            self.update_display()
            self.valueChanged.emit(self.value)

    def update_display(self):
        self.value_edit.setText(format(self.value, f".{self.decimals}f"))

    def set_value(self, value):
        self.value = max(self.min_val, min(self.max_val, value))
        self.update_display()


class MainWindow(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLO-Detector-Pro")
        self.setMinimumSize(1200, 800)
        self.resize(1400, 900)
        
        # Set window icon
        icon_path = os.path.join("assets", "icons", "app_icon.ico")
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))
        
        # State variables
        self.is_running = False
        self.current_mode = "image"
        self.current_image_path = None
        self.current_folder_path = None
        self.current_detections = []
        self.batch_results = []  # For folder mode
        
        # Detection manager for history tracking
        self.detection_manager = DetectionManager()
        
        # YOLO workers (main and comparison)
        self.worker = None
        self.comparison_worker = None
        self.comparison_fps_a = 0.0
        self.comparison_fps_b = 0.0
        self.is_comparing = False
        
        # Timer for camera display
        self.camera_timer = None
        
        # Initialize UI
        self.init_ui()
        self.load_stylesheet()
        
    def init_ui(self):
        """Initialize the user interface"""
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Create main sections
        self.create_sidebar()
        self.create_center_area()
        self.create_settings_panel()
        
        # Add to main layout
        main_layout.addWidget(self.sidebar)
        main_layout.addWidget(self.center_area, stretch=1)
        main_layout.addWidget(self.settings_panel)
    
    def create_sidebar(self):
        """Create left navigation sidebar"""
        self.sidebar = QFrame()
        self.sidebar.setObjectName("sidebar")
        
        layout = QVBoxLayout(self.sidebar)
        layout.setContentsMargins(0, 20, 0, 20)
        layout.setSpacing(20)  # Fixed spacing between buttons
        
        # Button group for exclusive selection
        self.nav_button_group = QButtonGroup(self)
        
        # Navigation buttons (icon only, no text)
        self.image_btn = NavigationButton("image")
        self.folder_btn = NavigationButton("folder")
        self.video_btn = NavigationButton("video")
        self.camera_btn = NavigationButton("camera")
        self.dashboard_btn = NavigationButton("dashboard")
        self.compare_btn = NavigationButton("compare")
        
        # Add stretch at top to center buttons vertically
        layout.addStretch()
        
        # Add all buttons with button group
        for btn in [self.image_btn, self.folder_btn, self.video_btn, self.camera_btn, 
                    self.dashboard_btn, self.compare_btn]:
            self.nav_button_group.addButton(btn)
            layout.addWidget(btn)
        
        # Add stretch at bottom to center buttons vertically
        layout.addStretch()
        
        # Set default selection
        self.image_btn.setChecked(True)
        
        # Connect signals
        self.image_btn.clicked.connect(lambda: self.switch_mode("image"))
        self.folder_btn.clicked.connect(lambda: self.switch_mode("folder"))
        self.video_btn.clicked.connect(lambda: self.switch_mode("video"))
        self.camera_btn.clicked.connect(lambda: self.switch_mode("camera"))
        self.dashboard_btn.clicked.connect(lambda: self.switch_mode("dashboard"))
        self.compare_btn.clicked.connect(lambda: self.switch_mode("compare"))
    
    def create_center_area(self):
        """Create center preview area with status bar"""
        self.center_area = QFrame()
        
        layout = QVBoxLayout(self.center_area)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Top status bar
        self.create_status_bar()
        layout.addWidget(self.status_bar)
        
        # Stacked widget for different modes
        self.mode_stack = QStackedWidget()
        
        # 1. Image/Camera/Folder preview area (shared)
        self.preview_widget = QWidget()
        self.preview_layout = QVBoxLayout(self.preview_widget)
        self.preview_layout.setContentsMargins(0, 0, 0, 0)
        
        # Center Action Button (Dynamic based on mode)
        self.action_button = QPushButton("Action")
        self.action_button.setCursor(Qt.PointingHandCursor)
        self.action_button.clicked.connect(lambda: None) # Dummy connection to prevent RuntimeWarning on first switch
        self.action_button.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #667eea, stop:1 #764ba2);
                color: white;
                border: none;
                border-radius: 8px;
                padding: 15px 30px;
                font-size: 16px;
                font-weight: bold;
                min-width: 200px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #764ba2, stop:1 #667eea);
            }
        """)
        
        # Container for centering button over preview
        self.button_container = QWidget(self.preview_widget)
        button_layout = QVBoxLayout(self.button_container)
        button_layout.addWidget(self.action_button)
        button_layout.setAlignment(Qt.AlignCenter)
        
        # 1. Standard Preview label (Actual display for Video/Camera)
        self.preview_label = QLabel("等待操作 Waiting for action...")
        self.preview_label.setObjectName("previewLabel")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setScaledContents(False)
        self.preview_label.setStyleSheet("background-color: #1a202c;")

        # 2. Side-by-Side Comparison Container (For Image/Folder)
        self.comparison_container = QWidget()
        self.comparison_layout = QHBoxLayout(self.comparison_container)
        self.comparison_layout.setContentsMargins(10, 10, 10, 10)
        self.comparison_layout.setSpacing(10)

        # Left side: Original
        source_vbox = QVBoxLayout()
        source_title = QLabel("原始图像 Original")
        source_title.setStyleSheet("color: #a0aec0; font-weight: bold; margin-bottom: 5px;")
        source_title.setAlignment(Qt.AlignCenter)
        self.source_label = QLabel()
        self.source_label.setStyleSheet("background-color: #000000; border: 1px solid #2d3748; border-radius: 4px;")
        self.source_label.setAlignment(Qt.AlignCenter)
        source_vbox.addWidget(source_title)
        source_vbox.addWidget(self.source_label, 1)

        # Right side: Result
        result_vbox = QVBoxLayout()
        result_title = QLabel("检测结果 Result")
        result_title.setStyleSheet("color: #a0aec0; font-weight: bold; margin-bottom: 5px;")
        result_title.setAlignment(Qt.AlignCenter)
        self.result_label = QLabel()
        self.result_label.setStyleSheet("background-color: #000000; border: 1px solid #2d3748; border-radius: 4px;")
        self.result_label.setAlignment(Qt.AlignCenter)
        result_vbox.addWidget(result_title)
        result_vbox.addWidget(self.result_label, 1)

        self.comparison_layout.addLayout(source_vbox)
        self.comparison_layout.addLayout(result_vbox)
        self.comparison_container.setVisible(False)
        
        # Add components to preview layout
        self.preview_layout.addWidget(self.preview_label, 1)
        self.preview_layout.addWidget(self.comparison_container, 1)
        self.preview_layout.addWidget(self.button_container, 0, Qt.AlignCenter)
        self.preview_layout.addSpacing(20) # Bottom padding

        
        # Progress bar for batch processing (hidden initially)
        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #4a5568;
                border-radius: 5px;
                background-color: #1a202c;
                text-align: center;
                color: #e2e8f0;
                font-weight: bold;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #667eea, stop:1 #764ba2);
                border-radius: 3px;
            }
        """)
        self.progress_bar.setVisible(False)
        self.preview_layout.addWidget(self.progress_bar)
        
        # 2. Dashboard widget
        self.dashboard_widget = DashboardWidget(self.detection_manager)
        self.dashboard_widget.export_json_btn.clicked.connect(self.export_dashboard_json)
        self.dashboard_widget.export_csv_btn.clicked.connect(self.export_dashboard_csv)
        self.dashboard_widget.clear_btn.clicked.connect(self.clear_dashboard)
        
        # 3. Comparison widget
        self.comparison_widget = ComparisonWidget()
        self.comparison_widget.load_image_btn.clicked.connect(self.load_comparison_image)
        self.comparison_widget.load_folder_btn.clicked.connect(self.load_comparison_folder)
        self.comparison_widget.load_video_btn.clicked.connect(self.load_comparison_video)
        self.comparison_widget.compare_btn.clicked.connect(self.run_comparison)
        self.comparison_widget.model_changed.connect(self.on_comparison_model_changed)
        
        # Add all widgets to stack
        self.mode_stack.addWidget(self.preview_widget)  # 0: image/camera/folder
        self.mode_stack.addWidget(self.dashboard_widget)  # 1: dashboard
        self.mode_stack.addWidget(self.comparison_widget)  # 2: comparison
        
        layout.addWidget(self.mode_stack)
    
    def create_status_bar(self):
        """Create top status bar with info cards"""
        self.status_bar = QFrame()
        self.status_bar.setObjectName("statusBar")
        
        layout = QHBoxLayout(self.status_bar)
        layout.setContentsMargins(20, 10, 20, 10)
        layout.setSpacing(15)
        layout.setAlignment(Qt.AlignVCenter)  # 垂直居中对齐
        
        # Status cards with icons
        self.classes_card = StatusCard("Classes", "classes")
        self.targets_card = StatusCard("Targets", "targets")
        self.fps_card = StatusCard("FPS", "fps")
        self.model_card = StatusCard("Model", "model")
        
        # Add cards with equal stretch to fill the width
        layout.addWidget(self.classes_card, 1, Qt.AlignVCenter)
        layout.addWidget(self.targets_card, 1, Qt.AlignVCenter)
        layout.addWidget(self.fps_card, 1, Qt.AlignVCenter)
        layout.addWidget(self.model_card, 1, Qt.AlignVCenter)
    
    def create_settings_panel(self):
        """Create right settings panel"""
        self.settings_panel = QFrame()
        self.settings_panel.setObjectName("settingsPanel")
        
        layout = QVBoxLayout(self.settings_panel)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)
        
        # Spacer at top
        layout.addStretch(1)
        
        # Add branding logo (part of the centered block)
        logo_path = os.path.join("assets", "icons", "LESSERAFIM-logo.png")
        if os.path.exists(logo_path):
            logo_label = QLabel()
            pixmap = QPixmap(logo_path)
            # Adjusted size for better fit
            scaled_pix = pixmap.scaledToWidth(200, Qt.SmoothTransformation)
            logo_label.setPixmap(scaled_pix)
            logo_label.setAlignment(Qt.AlignCenter)
            logo_label.setStyleSheet("background: transparent; margin-bottom: 10px;")
            layout.addWidget(logo_label)
        
        # Model Selection Group
        model_group = QGroupBox("模型选择 Model")
        model_layout = QVBoxLayout()
        model_layout.setSpacing(10)
        
        model_label = QLabel("选择模型:")
        model_label.setProperty("class", "setting-label")
        
        self.model_combo = QComboBox()
        # Load models from 'models' directory
        model_list = get_available_models()
        self.model_combo.addItems(model_list)
        self.model_combo.currentTextChanged.connect(self.on_model_changed)
        
        # Trigger initial update
        if model_list:
            self.on_model_changed(self.model_combo.currentText())
        
        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_combo)
        model_group.setLayout(model_layout)
        
        # Hyperparameters Group
        params_group = QGroupBox("超参数 Hyperparameters")
        params_layout = QVBoxLayout()
        params_layout.setSpacing(10)
        
        # Confidence threshold
        conf_label = QLabel("置信度 Confidence:")
        conf_label.setProperty("class", "setting-label")
        
        conf_container = QHBoxLayout()
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setMinimum(1)
        self.conf_slider.setMaximum(99)
        self.conf_slider.setValue(25)
        
        self.conf_adjuster = CustomAdjuster(0.25, 0.01, 0.99, 0.01, 2)
        self.conf_adjuster.valueChanged.connect(self.update_worker_parameters)
        
        # Connect slider and adjuster
        self.conf_slider.valueChanged.connect(
            lambda v: self.conf_adjuster.set_value(v / 100)
        )
        self.conf_adjuster.valueChanged.connect(
            lambda v: self.conf_slider.setValue(int(v * 100))
        )
        
        conf_container.addWidget(self.conf_slider, 1)
        conf_container.addWidget(self.conf_adjuster)
        
        # IOU threshold
        iou_label = QLabel("IOU 阈值 IOU Threshold:")
        iou_label.setProperty("class", "setting-label")
        
        iou_container = QHBoxLayout()
        self.iou_slider = QSlider(Qt.Horizontal)
        self.iou_slider.setMinimum(1)
        self.iou_slider.setMaximum(99)
        self.iou_slider.setValue(45)
        
        self.iou_adjuster = CustomAdjuster(0.45, 0.01, 0.99, 0.01, 2)
        self.iou_adjuster.valueChanged.connect(self.update_worker_parameters)
        
        # Connect slider and adjuster
        self.iou_slider.valueChanged.connect(
            lambda v: self.iou_adjuster.set_value(v / 100)
        )
        self.iou_adjuster.valueChanged.connect(
            lambda v: self.iou_slider.setValue(int(v * 100))
        )
        
        iou_container.addWidget(self.iou_slider, 1)
        iou_container.addWidget(self.iou_adjuster)
        
        # Line Width
        lw_label = QLabel("线宽 \nLine Width:")
        lw_label.setProperty("class", "setting-label")
        self.lw_adjuster = CustomAdjuster(1.0, 1.0, 10.0, 1.0, 0, compact=True)
        self.lw_adjuster.valueChanged.connect(self.update_worker_parameters)
        
        # Inference Delay
        delay_label = QLabel("推理延迟 \nDelay (ms):")
        delay_label.setProperty("class", "setting-label")
        self.delay_adjuster = CustomAdjuster(0.0, 0.0, 1000.0, 50.0, 0, compact=True)
        self.delay_adjuster.valueChanged.connect(self.update_worker_parameters)
        
        params_layout.addWidget(conf_label)
        params_layout.addLayout(conf_container)
        params_layout.addWidget(iou_label)
        params_layout.addLayout(iou_container)
        
        # New parameters layout
        lw_delay_layout = QHBoxLayout()
        
        lw_vbox = QVBoxLayout()
        lw_vbox.addWidget(lw_label)
        lw_vbox.addWidget(self.lw_adjuster)
        
        delay_vbox = QVBoxLayout()
        delay_vbox.addWidget(delay_label)
        delay_vbox.addWidget(self.delay_adjuster)
        
        lw_delay_layout.addLayout(lw_vbox)
        lw_delay_layout.addSpacing(10)
        lw_delay_layout.addLayout(delay_vbox)
        
        params_layout.addLayout(lw_delay_layout)
        params_group.setLayout(params_layout)
        
        layout.addStretch(1) # Gap
        
        # Control Buttons
        control_group = QGroupBox("控制 Control")
        control_layout = QVBoxLayout()
        control_layout.setSpacing(10)
        
        self.run_button = QPushButton("▶ 运行 RUN")
        self.run_button.setObjectName("runButton")
        self.run_button.setProperty("isRunning", False)
        self.run_button.setProperty("class", "shrink-btn")
        self.run_button.clicked.connect(self.toggle_detection)
        
        self.save_button = QPushButton("💾 保存结果 SAVE")
        self.save_button.setObjectName("saveButton")
        self.save_button.setProperty("class", "shrink-btn")
        self.save_button.setEnabled(False)
        self.save_button.clicked.connect(self.save_results)
        
        self.reset_button = QPushButton("🔄 重置 RESET")
        self.reset_button.setObjectName("resetButton")
        self.reset_button.setProperty("class", "shrink-btn")
        self.reset_button.clicked.connect(self.reset_all)
        self.reset_button.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #64748b, stop:1 #475569);
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #475569, stop:1 #64748b);
            }
        """)
        
        control_layout.addWidget(self.run_button)
        control_layout.addWidget(self.save_button)
        control_layout.addWidget(self.reset_button)
        control_group.setLayout(control_layout)
        
        # Add all groups to settings panel
        layout.addWidget(model_group)
        layout.addWidget(params_group)
        layout.addWidget(control_group)
        
        # Add bottom stretch for vertical centering
        layout.addStretch(1)
    
    def load_stylesheet(self):
        """Load QSS stylesheet"""
        qss_path = os.path.join(os.path.dirname(__file__), "ui", "style.qss")
        try:
            with open(qss_path, "r", encoding="utf-8") as f:
                self.setStyleSheet(f.read())
        except FileNotFoundError:
            print(f"Warning: Stylesheet not found at {qss_path}")
    
    def update_worker_parameters(self):
        """Update worker parameters in real-time if a worker is running"""
        conf = self.conf_adjuster.value
        iou = self.iou_adjuster.value
        lw = int(self.lw_adjuster.value)
        delay = int(self.delay_adjuster.value)
        
        if self.worker and self.worker.isRunning():
            self.worker.set_parameters(conf, iou, lw, delay)
            print(f"Worker parameters updated: Conf={conf}, IOU={iou}, LW={lw}, Delay={delay}")
        
        if self.comparison_worker and self.comparison_worker.isRunning():
            self.comparison_worker.set_parameters(conf, iou, lw, delay)
            print(f"Comparison Worker parameters updated: Conf={conf}, IOU={iou}, LW={lw}, Delay={delay}")
    
    def switch_mode(self, mode):
        """Switch detection mode"""
        # Stop any running worker
        if self.worker:
            self.worker.stop()
            self.worker.wait()
            self.worker = None
        
        self.is_running = False
        self.run_button.setProperty("isRunning", False)
        self.run_button.setText("▶ 运行 RUN")
        self.run_button.style().unpolish(self.run_button)
        self.run_button.style().polish(self.run_button)
        
        self.current_mode = mode
        print(f"Switched to mode: {mode}")
        
        # Switch stacked widget based on mode
        if mode in ["image", "folder", "camera", "video"]:
            self.mode_stack.setCurrentIndex(0)
            self.progress_bar.setVisible(False)
            self.run_button.setEnabled(False) # Disable run until loaded
            self.save_button.setEnabled(False)
            
            # Toggle Comparison View
            is_comp_mode = mode in ["image", "folder"]
            self.preview_label.setVisible(not is_comp_mode)
            self.comparison_container.setVisible(is_comp_mode)
            
            self.preview_label.clear()
            self.preview_label.setText("")
            if is_comp_mode:
                self.source_label.clear()
                self.result_label.clear()
            
            # Configure Action Button
            self.button_container.setVisible(True)
            try:
                # Safely disconnect to avoid RuntimeWarning
                self.action_button.clicked.disconnect() 
            except (RuntimeError, TypeError):
                pass
                
            if mode == "image":
                self.action_button.setText("📂 选择图片 Select Image")
                self.action_button.clicked.connect(self.select_image)
                self.preview_label.setText("请选择图片\nPlease Select Image")
            elif mode == "folder":
                self.action_button.setText("📁 选择文件夹 Select Folder")
                self.action_button.clicked.connect(self.select_folder)
                self.preview_label.setText("请选择文件夹\nPlease Select Folder")
            elif mode == "video":
                self.action_button.setText("🎥 选择视频 Select Video")
                self.action_button.clicked.connect(self.select_video)
                self.preview_label.setText("请选择视频\nPlease Select Video")
            elif mode == "camera":
                self.action_button.setText("📹 启动摄像头 Start Camera")
                self.action_button.clicked.connect(self.open_camera)
                self.preview_label.setText("点击启动摄像头\nClick to Start Camera")
            
        elif mode == "dashboard":
            self.mode_stack.setCurrentIndex(1)
            self.dashboard_widget.refresh_data()
            
        elif mode == "compare":
            self.mode_stack.setCurrentIndex(2)
    
    
    def toggle_detection(self):
        """Toggle detection on/off (Classic 'Run' button)"""
        if self.current_mode == "image":
             if self.current_image_path:
                 self.run_image_detection()
             else:
                 QMessageBox.warning(self, "Warning", "请先选择图片\nPlease select an image first")
                 
        elif self.current_mode == "folder":
             if self.current_folder_path:
                 self.run_folder_detection()
             else:
                 QMessageBox.warning(self, "Warning", "请先选择文件夹\nPlease select a folder first")

        elif self.current_mode == "video":
             if self.is_running:
                 self.stop_video_detection()
             elif getattr(self, 'current_video_path', None):
                 self.run_video_detection()
             else:
                 QMessageBox.warning(self, "Warning", "请先选择视频\nPlease select a video first")
                 
        elif self.current_mode == "camera":
            # Toggle inference state
            self.is_running = not self.is_running
            if self.worker:
                self.worker.set_inference_enabled(self.is_running)
            
            if self.is_running:
                self.run_button.setText("⏹ 停止检测 STOP Detection")
                self.run_button.setProperty("isRunning", True)
                self.detection_manager.start_session("camera", self.model_combo.currentText())
            else:
                self.run_button.setText("▶ 开始检测 Start Detection")
                self.run_button.setProperty("isRunning", False)
                self.detection_manager.end_session()
            
            self.run_button.style().unpolish(self.run_button)
            self.run_button.style().polish(self.run_button)

    def select_image(self):
        """Step 1: Select and load image (Preview only)"""
        filepath, _ = QFileDialog.getOpenFileName(
            self, "选择图片 Select Image", "", "Images (*.png *.jpg *.jpeg *.bmp *.webp)"
        )
        if filepath:
            self.current_image_path = filepath
            # Show on source label
            import cv2
            frame = cv2.imread(filepath)
            pixmap = numpy_to_qpixmap(frame)
            scaled = pixmap.scaled(self.source_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.source_label.setPixmap(scaled)
            self.result_label.clear()
            self.result_label.setText("点击运行开始检测\nClick RUN to start")
            
            self.run_button.setEnabled(True)
            self.save_button.setEnabled(False)
            self.button_container.setVisible(False) # Hide button after selection (can add a change button later if needed)

    def run_image_detection(self):
        """Step 2: Run inference on loaded image"""
        self.worker = YOLOWorker(self.model_combo.currentText())
        self.worker.set_parameters(
            self.conf_adjuster.value, 
            self.iou_adjuster.value,
            int(self.lw_adjuster.value),
            int(self.delay_adjuster.value)
        )
        self.worker.set_source(self.current_image_path, "image")
        
        self.worker.frame_processed.connect(self.on_image_processed)
        self.worker.stats_updated.connect(self.on_stats_updated)
        self.worker.error_occurred.connect(self.on_error)
        
        self.run_button.setEnabled(False) # Prevent double click
        self.detection_manager.start_session("image", self.model_combo.currentText())
        self.worker.start()

    def select_folder(self):
        """Step 1: Select folder"""
        folder = QFileDialog.getExistingDirectory(self, "选择文件夹 Select Folder")
        if folder:
             self.current_folder_path = folder
             self.source_label.clear()
             self.source_label.setText(f"已选择目录:\n{folder}")
             self.result_label.clear()
             self.result_label.setText("点击运行开始批量检测\nClick RUN to start")
             self.run_button.setEnabled(True)
             self.button_container.setVisible(False)

    def run_folder_detection(self):
        """Step 2: Run batch detection"""
        self.batch_results = []
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self.run_button.setEnabled(False)
        
        self.worker = YOLOWorker(self.model_combo.currentText())
        self.worker.set_parameters(
            self.conf_adjuster.value, 
            self.iou_adjuster.value,
            int(self.lw_adjuster.value),
            int(self.delay_adjuster.value)
        )
        self.worker.set_source(self.current_folder_path, "folder")
        
        self.worker.batch_item_processed.connect(self.on_batch_item_processed)
        self.worker.progress_updated.connect(self.on_progress_updated)
        self.worker.stats_updated.connect(self.on_stats_updated)
        self.worker.finished.connect(self.on_batch_finished)
        self.worker.error_occurred.connect(self.on_error)
        
        self.detection_manager.start_session("folder", self.model_combo.currentText())
        self.worker.start()

    def select_video(self):
        """Step 1: Select video"""
        filepath, _ = QFileDialog.getOpenFileName(
            self, "选择视频 Select Video", "", "Videos (*.mp4 *.avi *.mkv *.mov)"
        )
        if filepath:
            self.current_video_path = filepath
            # Show on source label
            pixmap = QPixmap(filepath)
            scaled = pixmap.scaled(self.source_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.source_label.setPixmap(scaled)
            self.result_label.clear()
            self.result_label.setText("点击运行开始检测\nClick RUN to start")
            
            self.run_button.setEnabled(True)
            self.button_container.setVisible(False)

    def run_video_detection(self):
        """Step 2: Run video detection"""
        self.worker = YOLOWorker(self.model_combo.currentText())
        self.worker.set_parameters(
            self.conf_adjuster.value, 
            self.iou_adjuster.value,
            int(self.lw_adjuster.value),
            int(self.delay_adjuster.value)
        )
        self.worker.set_source(self.current_video_path, "video")
        
        self.worker.frame_processed.connect(self.on_frame_processed) # Reuse frame processor
        self.worker.fps_updated.connect(self.on_fps_updated)
        self.worker.stats_updated.connect(self.on_stats_updated)
        self.worker.error_occurred.connect(self.on_error)
        self.worker.finished.connect(self.stop_video_detection) # Reset on finish
        
        self.is_running = True
        self.run_button.setText("⏹ 停止 STOP")
        self.run_button.setProperty("isRunning", True)
        self.run_button.style().unpolish(self.run_button)
        self.run_button.style().polish(self.run_button)
        self.run_button.setEnabled(True)
        
        self.detection_manager.start_session("video", self.model_combo.currentText())
        self.worker.start()

    def stop_video_detection(self):
        """Stop video detection"""
        if self.worker:
            self.worker.stop()
            self.worker.wait()
            self.worker = None
        
        self.is_running = False
        self.run_button.setText("▶ 运行 RUN")
        self.run_button.setProperty("isRunning", False)
        self.run_button.style().unpolish(self.run_button)
        self.run_button.style().polish(self.run_button)
        self.save_button.setEnabled(True) # Allow saving after stop?
        self.detection_manager.end_session()

    def load_comparison_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "选择文件夹 Select Folder")
        if folder:
             self.current_comparison_source = folder
             self.comparison_mode = "folder"
             QMessageBox.information(self, "Ready", f"已选择文件夹: {folder}\nSelected folder: {folder}")

    def load_comparison_video(self):
        file, _ = QFileDialog.getOpenFileName(self, "选择视频 Select Video", "", "Videos (*.mp4 *.avi *.mkv *.mov)")
        if file:
             self.current_comparison_source = file
             self.comparison_mode = "video"
             QMessageBox.information(self, "Ready", f"已选择视频: {file}\nSelected video: {file}")

    def open_camera(self):
        """Step 1: Start Camera Preview (No Inference)"""
        self.worker = YOLOWorker(self.model_combo.currentText())
        self.worker.set_parameters(
            self.conf_adjuster.value, 
            self.iou_adjuster.value,
            int(self.lw_adjuster.value),
            int(self.delay_adjuster.value)
        )
        self.worker.set_source(0, "camera")
        self.worker.set_inference_enabled(False) # Preview only
        
        self.worker.frame_processed.connect(self.on_frame_processed)
        self.worker.error_occurred.connect(self.on_error)
        
        self.worker.start()
        
        self.button_container.setVisible(False)
        self.run_button.setEnabled(True)
        self.run_button.setText("▶ 开始检测 Start Detection")
        self.is_running = False

    def load_image(self): # Legacy, redirect
        self.select_image()

    def load_folder(self): # Legacy, redirect
        self.select_folder()

    def start_camera_detection(self): # Legacy logic, integrated above
        pass 

    def stop_camera_detection(self): # Legacy logic
        pass
    
    def on_image_processed(self, frame, detections):
        """Handle single image detection result"""
        self.current_detections = detections
        
        # Draw detections on frame
        annotated_frame = draw_detections(frame, detections, line_width=int(self.lw_adjuster.value))
        
        # Display Original
        source_pixmap = numpy_to_qpixmap(frame)
        self.source_label.setPixmap(source_pixmap.scaled(
            self.source_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        ))
        
        # Display Result
        result_pixmap = numpy_to_qpixmap(annotated_frame)
        self.result_label.setPixmap(result_pixmap.scaled(
            self.result_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        ))
        
        # Add to detection manager
        self.detection_manager.add_detection("image", detections)
        
        # Enable save button
        self.save_button.setEnabled(True)
        
        # End session
        self.detection_manager.end_session()
    
    def on_frame_processed(self, frame, detections):
        """Handle camera frame detection result"""
        self.current_detections = detections
        
        # Draw detections on frame
        annotated_frame = draw_detections(frame, detections, line_width=int(self.lw_adjuster.value))
        
        # Convert to QPixmap and display
        pixmap = numpy_to_qpixmap(annotated_frame)
        scaled_pixmap = pixmap.scaled(
            self.preview_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.preview_label.setPixmap(scaled_pixmap)
        
        # Add to detection manager (every 30th frame to avoid memory issues)
        import random
        if random.randint(1, 30) == 1:
            self.detection_manager.add_detection(f"frame_{random.randint(1, 10000)}", detections)
    
    def on_batch_item_processed(self, filepath, frame, detections):
        """Handle batch processing of individual item"""
        self.batch_results.append((filepath, frame, detections))
        
        # Show last processed image
        annotated_frame = draw_detections(frame, detections, line_width=int(self.lw_adjuster.value))
        
        # Display Original
        source_pixmap = numpy_to_qpixmap(frame)
        self.source_label.setPixmap(source_pixmap.scaled(
            self.source_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        ))
        
        # Display Result
        result_pixmap = numpy_to_qpixmap(annotated_frame)
        self.result_label.setPixmap(result_pixmap.scaled(
            self.result_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        ))
        
        # Add to detection manager
        import os
        self.detection_manager.add_detection(os.path.basename(filepath), detections)
    
    def on_progress_updated(self, current, total):
        """Update progress bar"""
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)
        self.progress_bar.setFormat(f"Processing: {current}/{total} ({current*100//total}%)")
    
    def on_batch_finished(self):
        """Handle batch processing completion"""
        self.progress_bar.setVisible(False)
        self.run_button.setEnabled(True)
        self.save_button.setEnabled(True)
        
        # End session
        self.detection_manager.end_session()
        
        QMessageBox.information(
            self,
            "批量检测完成 Batch Complete",
            f"成功处理 {len(self.batch_results)} 张图片\nProcessed {len(self.batch_results)} images"
        )
    
    def on_stats_updated(self, stats):
        """Update statistics display"""
        self.classes_card.set_value(stats.get('classes', 0))
        self.targets_card.set_value(stats.get('targets', 0))
    
    def on_fps_updated(self, fps):
        """Update FPS display"""
        self.fps_card.set_value(f"{fps:.1f}")
    
    def on_error(self, error_msg):
        """Handle worker errors"""
        QMessageBox.critical(self, "错误 Error", error_msg)
        self.run_button.setEnabled(True)
    
    def on_model_changed(self, model_name):
        """Handle model selection change"""
        self.model_card.set_value(model_name)
        print(f"Model changed to: {model_name}")
    
    def save_results(self):
        """Save detection results"""
        if self.current_mode == "image" and self.current_detections:
            self.save_single_result()
        elif self.current_mode == "folder" and self.batch_results:
            self.save_batch_result()
        elif self.current_mode == "camera" and self.current_detections:
            self.save_single_result()
        else:
            QMessageBox.warning(self, "警告 Warning", "没有可保存的结果\nNo results to save")
    
    def save_single_result(self):
        """Save single detection result"""
        if not self.current_detections:
            return
        
        # Get save location
        save_path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "保存结果 Save Result",
            "",
            "JPEG Image (*.jpg);;PNG Image (*.png);;JSON Data (*.json)"
        )
        
        if save_path:
            if save_path.endswith('.json'):
                # Save as JSON
                save_detection_json(self.current_detections, save_path)
            else:
                # Save as image
                if self.current_image_path:
                    import cv2
                    frame = cv2.imread(self.current_image_path)
                    annotated = draw_detections(frame, self.current_detections)
                    cv2.imwrite(save_path, annotated)
                
            QMessageBox.information(self, "成功 Success", f"结果已保存到:\nSaved to: {save_path}")
    
    def save_batch_result(self):
        """Save batch detection results"""
        folder = QFileDialog.getExistingDirectory(
            self,
            "选择保存文件夹 Select Save Folder"
        )
        
        if folder:
            saved_dir = save_batch_results(self.batch_results, folder)
            QMessageBox.information(
                self,
                "成功 Success",
                f"批量结果已保存到:\nBatch results saved to:\n{saved_dir}"
            )
    
    # Dashboard methods
    def export_dashboard_json(self):
        """Export dashboard data to JSON"""
        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "导出 JSON Export JSON",
            "detection_history.json",
            "JSON Files (*.json)"
        )
        
        if filepath:
            self.detection_manager.export_to_json(filepath)
            QMessageBox.information(self, "成功 Success", f"数据已导出到:\nExported to: {filepath}")
    
    def export_dashboard_csv(self):
        """Export dashboard data to CSV"""
        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "导出 CSV Export CSV",
            "detection_summary.csv",
            "CSV Files (*.csv)"
        )
        
        if filepath:
            self.detection_manager.export_to_csv(filepath)
            QMessageBox.information(self, "成功 Success", f"数据已导出到:\nExported to: {filepath}")
    
    def clear_dashboard(self):
        """Clear dashboard history"""
        reply = QMessageBox.question(
            self,
            "确认 Confirm",
            "确定要清空所有历史记录吗?\nClear all history?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.detection_manager.clear_history()
            self.dashboard_widget.refresh_data()
            QMessageBox.information(self, "成功 Success", "历史记录已清空\nHistory cleared")
    
    # Comparison mode methods
    def load_comparison_image(self):
        """Load image for comparison"""
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "选择图片 Select Image",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tiff *.webp)"
        )
        
        if filepath:
            self.current_comparison_source = filepath
            self.comparison_mode = "image"
            # Show image in both panels
            import cv2
            frame = cv2.imread(filepath)
            pixmap = numpy_to_qpixmap(frame)
            
            self.comparison_widget.panel_left.set_image(pixmap)
            self.comparison_widget.panel_right.set_image(pixmap)
    
    def run_comparison(self):
        """Run comparison detection"""
        
        if self.is_comparing:
            self.stop_comparison()
            return

        if not getattr(self, 'current_comparison_source', None):
            QMessageBox.warning(self, "警告 Warning", "请先选择对比源 (图片/文件夹/视频)\nPlease select a source (image/folder/video) first")
            return
        
        # Get models
        model_a = self.comparison_widget.panel_left.model_combo.currentText()
        model_b = self.comparison_widget.panel_right.model_combo.currentText()
        
        self.comparison_worker = ComparisonWorker(model_a, model_b)
        self.comparison_worker.set_source(self.current_comparison_source, self.comparison_mode)
        self.comparison_worker.set_parameters(
            self.conf_adjuster.value, 
            self.iou_adjuster.value,
            int(self.lw_adjuster.value),
            int(self.delay_adjuster.value)
        )
        
        self.comparison_worker.frame_processed.connect(self.on_comparison_frame)
        self.comparison_worker.fps_updated.connect(self.on_comparison_fps)
        self.comparison_worker.finished.connect(self.stop_comparison)
        self.comparison_worker.error_occurred.connect(self.on_error)
        
        self.comparison_worker.start()
        self.is_comparing = True
        self.comparison_widget.compare_btn.setText("⏹ 停止对比 Stop Comparison")
        # Update button style for running state
        self.comparison_widget.compare_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #ef4444, stop:1 #dc2626);
                color: white;
                border: none;
                border-radius: 5px;
                padding: 10px 20px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #dc2626, stop:1 #ef4444);
            }
        """)

    def stop_comparison(self):
        """Stop comparison detection"""
        if self.comparison_worker:
            self.comparison_worker.stop()
            self.comparison_worker = None
            
        self.is_comparing = False
        self.comparison_widget.compare_btn.setText("▶ 开始对比 Start Comparison")
        # Reset button style
        self.comparison_widget.compare_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #f093fb, stop:1 #f5576c);
                color: white;
                border: none;
                border-radius: 5px;
                padding: 10px 20px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #f5576c, stop:1 #f093fb);
            }
        """)

    def on_comparison_frame(self, frame, det_a, det_b, time_a, time_b):
        """Handle synchronized comparison results"""
        # Process A
        annotated_a = draw_detections(frame.copy(), det_a, line_width=int(self.lw_adjuster.value))
        pixmap_a = numpy_to_qpixmap(annotated_a)
        classes_a = len(set(d['class_id'] for d in det_a))
        targets_a = len(det_a)
        
        # Process B
        annotated_b = draw_detections(frame.copy(), det_b, line_width=int(self.lw_adjuster.value))
        pixmap_b = numpy_to_qpixmap(annotated_b)
        classes_b = len(set(d['class_id'] for d in det_b))
        targets_b = len(det_b)
        
        fps_a = self.comparison_fps_a
        fps_b = self.comparison_fps_b
        
        self.comparison_widget.update_left_panel(pixmap_a, classes_a, targets_a, fps_a, time_a)
        self.comparison_widget.update_right_panel(pixmap_b, classes_b, targets_b, fps_b, time_b)

    def on_comparison_fps(self, fps_a, fps_b):
        self.comparison_fps_a = fps_a
        self.comparison_fps_b = fps_b
    
    def on_comparison_model_changed(self, panel_index, model_name):
        """Handle comparison model change"""
        print(f"Comparison panel {panel_index} model changed to: {model_name}")

    def reset_all(self):
        """Reset application state"""
        # Stop everything
        if self.is_running:
            if self.current_mode == "video":
                self.stop_video_detection()
            elif self.current_mode == "camera":
                self.toggle_detection()
        
        if self.is_comparing:
            self.stop_comparison()
            
        # Clear data
        self.current_image_path = None
        self.current_folder_path = None
        self.current_video_path = None
        self.current_comparison_source = None
        self.current_detections = []
        self.batch_results = []
        
        # Reset UI
        self.preview_label.clear()
        
        # Reset cards
        self.classes_card.set_value(0)
        self.targets_card.set_value(0)
        self.fps_card.set_value(0)
        
        self.progress_bar.setVisible(False)
        self.run_button.setEnabled(False)
        self.save_button.setEnabled(False)
        
        # Reset comparison widget
        self.comparison_widget.reset_ui()
        
        # Restore mode default state
        self.switch_mode(self.current_mode)
        
        QMessageBox.information(self, "Success", "所有数据已重置\nAll data reset")


if __name__ == "__main__":
    from PySide6.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

