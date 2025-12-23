"""
Comparison Widget - Side-by-side model comparison
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QComboBox, QPushButton, QFrame, QGroupBox
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap


class ComparisonPanel(QFrame):
    """Single comparison panel for one model"""
    
    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet("""
            ComparisonPanel {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #2d3748, stop:1 #1a202c);
                border-radius: 10px;
                border: 2px solid #4a5568;
            }
        """)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # Title
        title_label = QLabel(title)
        title_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #00D9FF; background: transparent;")
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)
        
        # Model selection
        model_layout = QHBoxLayout()
        model_label = QLabel("模型 Model:")
        model_label.setStyleSheet("color: #e2e8f0; background: transparent;")
        
        self.model_combo = QComboBox()
        # Load models dynamically
        from utils.helper import get_available_models
        self.model_combo.addItems(get_available_models())
        
        self.model_combo.setStyleSheet("""
            QComboBox {
                background-color: #1a202c;
                color: #e2e8f0;
                border: 1px solid #4a5568;
                border-radius: 5px;
                padding: 5px;
            }
            QComboBox:hover {
                border: 1px solid #00D9FF;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox QAbstractItemView {
                background-color: #2d3748;
                color: #e2e8f0;
                selection-background-color: #4a5568;
            }
        """)
        
        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_combo, 1)
        layout.addLayout(model_layout)
        
        # Preview area
        self.preview_label = QLabel("等待检测...\nWaiting for detection...")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setStyleSheet("""
            QLabel {
                background-color: #1a202c;
                color: #718096;
                border: 2px dashed #4a5568;
                border-radius: 8px;
                font-size: 16px;
                min-height: 400px;
            }
        """)
        self.preview_label.setScaledContents(False)
        layout.addWidget(self.preview_label, 1)
        
        # Statistics
        stats_group = QGroupBox("统计 Statistics")
        stats_group.setStyleSheet("""
            QGroupBox {
                background-color: rgba(45, 55, 72, 0.5);
                color: #00D9FF;
                font-weight: bold;
                border: 1px solid #4a5568;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                background-color: transparent;
            }
        """)
        
        stats_layout = QVBoxLayout()
        
        self.classes_label = QLabel("检测类别 Classes: 0")
        self.targets_label = QLabel("目标数量 Targets: 0")
        self.fps_label = QLabel("帧率 FPS: 0.0")
        self.time_label = QLabel("推理时间 Inference Time: 0 ms")
        
        for label in [self.classes_label, self.targets_label, self.fps_label, self.time_label]:
            label.setStyleSheet("color: #e2e8f0; font-size: 12px; background: transparent;")
            stats_layout.addWidget(label)
        
        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)
        
    def set_image(self, pixmap):
        """Set preview image"""
        if pixmap:
            scaled_pixmap = pixmap.scaled(
                self.preview_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.preview_label.setPixmap(scaled_pixmap)
            self.preview_label.setStyleSheet("""
                QLabel {
                    background-color: #1a202c;
                    border: 2px solid #4a5568;
                    border-radius: 8px;
                }
            """)
    
    def update_stats(self, classes, targets, fps, inference_time):
        """Update statistics display"""
        self.classes_label.setText(f"检测类别 Classes: {classes}")
        self.targets_label.setText(f"目标数量 Targets: {targets}")
        self.fps_label.setText(f"帧率 FPS: {fps:.1f}")
        self.time_label.setText(f"推理时间 Inference Time: {inference_time:.1f} ms")

    def reset(self):
        """Reset panel state"""
        self.preview_label.clear()
        self.preview_label.setText("等待检测...\nWaiting for detection...")
        self.preview_label.setStyleSheet("""
            QLabel {
                background-color: #1a202c;
                color: #718096;
                border: 2px dashed #4a5568;
                border-radius: 8px;
                font-size: 16px;
                min-height: 400px;
            }
        """)
        self.update_stats(0, 0, 0, 0)


class ComparisonWidget(QWidget):
    """Widget for comparing two models side-by-side"""
    
    model_changed = Signal(int, str)  # (panel_index, model_name)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        
    def init_ui(self):
        """Initialize UI"""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)
        
        # Title
        title = QLabel("⚖️ 模型对比 Model Comparison")
        title.setStyleSheet("font-size: 24px; font-weight: bold; color: #00D9FF;")
        title.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title)
        
        # Comparison panels
        panels_layout = QHBoxLayout()
        panels_layout.setSpacing(20)
        
        self.panel_left = ComparisonPanel("模型 A Model A")
        self.panel_right = ComparisonPanel("模型 B Model B")
        
        # Set default models
        self.panel_left.model_combo.setCurrentText("YOLOv8n")
        self.panel_right.model_combo.setCurrentText("YOLOv8s")
        
        # Connect signals
        self.panel_left.model_combo.currentTextChanged.connect(
            lambda model: self.model_changed.emit(0, model)
        )
        self.panel_right.model_combo.currentTextChanged.connect(
            lambda model: self.model_changed.emit(1, model)
        )
        
        panels_layout.addWidget(self.panel_left)
        panels_layout.addWidget(self.panel_right)
        
        main_layout.addLayout(panels_layout, 1)
        
        # Comparison metrics
        metrics_group = QGroupBox("对比指标 Comparison Metrics")
        metrics_group.setStyleSheet("""
            QGroupBox {
                font-size: 16px;
                font-weight: bold;
                color: #00D9FF;
                border: 2px solid #4a5568;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)
        
        metrics_layout = QHBoxLayout()
        metrics_layout.setSpacing(20)
        
        self.detection_diff_label = QLabel("检测差异 Detection Diff: --")
        self.fps_diff_label = QLabel("FPS 差异 FPS Diff: --")
        self.speed_diff_label = QLabel("速度差异 Speed Diff: --")
        
        for label in [self.detection_diff_label, self.fps_diff_label, self.speed_diff_label]:
            label.setStyleSheet("""
                QLabel {
                    background-color: #2d3748;
                    color: #e2e8f0;
                    padding: 15px;
                    border-radius: 8px;
                    font-size: 14px;
                    font-weight: bold;
                }
            """)
            label.setAlignment(Qt.AlignCenter)
            metrics_layout.addWidget(label)
        
        metrics_group.setLayout(metrics_layout)
        main_layout.addWidget(metrics_group)
        
        # Control buttons
        control_layout = QHBoxLayout()
        control_layout.addStretch()
        
        # Styles
        btn_style = """
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #667eea, stop:1 #764ba2);
                color: white;
                border: none;
                border-radius: 5px;
                padding: 10px 20px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #764ba2, stop:1 #667eea);
            }
        """
        
        compare_btn_style = """
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
        """
        
        self.load_image_btn = QPushButton("📁 加载图片 Load Image")
        self.load_image_btn.setStyleSheet(btn_style)
        
        self.load_folder_btn = QPushButton("📂 加载文件夹 Load Folder")
        self.load_folder_btn.setStyleSheet(btn_style)
        
        self.load_video_btn = QPushButton("🎥 加载视频 Load Video")
        self.load_video_btn.setStyleSheet(btn_style)
        
        self.compare_btn = QPushButton("▶ 开始对比 Start Comparison")
        self.compare_btn.setStyleSheet(compare_btn_style)
        
        control_layout.addWidget(self.load_image_btn)
        control_layout.addWidget(self.load_folder_btn)
        control_layout.addWidget(self.load_video_btn)
        control_layout.addWidget(self.compare_btn)
        
        main_layout.addLayout(control_layout)
        
    def update_left_panel(self, pixmap, classes, targets, fps, inference_time):
        """Update left panel with detection results"""
        self.panel_left.set_image(pixmap)
        self.panel_left.update_stats(classes, targets, fps, inference_time)
        self._update_comparison_metrics()
        
    def update_right_panel(self, pixmap, classes, targets, fps, inference_time):
        """Update right panel with detection results"""
        self.panel_right.set_image(pixmap)
        self.panel_right.update_stats(classes, targets, fps, inference_time)
        self._update_comparison_metrics()
        
    def _update_comparison_metrics(self):
        """Calculate and display comparison metrics"""
        # Extract values from panels
        try:
            left_targets = int(self.panel_left.targets_label.text().split(":")[1].strip())
            right_targets = int(self.panel_right.targets_label.text().split(":")[1].strip())
            
            left_fps_text = self.panel_left.fps_label.text().split(":")[1].strip()
            right_fps_text = self.panel_right.fps_label.text().split(":")[1].strip()
            left_fps = float(left_fps_text) if left_fps_text != "--" else 0
            right_fps = float(right_fps_text) if right_fps_text != "--" else 0
            
            left_time_text = self.panel_left.time_label.text().split(":")[1].replace("ms", "").strip()
            right_time_text = self.panel_right.time_label.text().split(":")[1].replace("ms", "").strip()
            left_time = float(left_time_text) if left_time_text != "--" else 0
            right_time = float(right_time_text) if right_time_text != "--" else 0
            
            # Calculate differences (Right - Left) (B - A)
            detection_diff = right_targets - left_targets
            fps_diff = right_fps - left_fps
            speed_diff = right_time - left_time 
            
            # Update labels with color coding and clear direction
            self.detection_diff_label.setText(
                f"检测差异 (B-A) Diff: {detection_diff:+d}"
            )
            
            self.fps_diff_label.setText(
                f"FPS 差异 (B-A) Diff: {fps_diff:+.1f}"
            )
            
            self.speed_diff_label.setText(
                f"速度差异 (B-A) Diff: {speed_diff:+.1f} ms"
            )
            
        except (ValueError, IndexError):
            pass

    def reset_ui(self):
        """Reset comparison UI"""
        self.panel_left.reset()
        self.panel_right.reset()
        
        self.detection_diff_label.setText("检测差异 (B-A) Diff: --")
        self.fps_diff_label.setText("FPS 差异 (B-A) Diff: --")
        self.speed_diff_label.setText("速度差异 (B-A) Diff: --")
