"""
Dashboard Widget - Displays detection statistics and history
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QTableWidget, QTableWidgetItem, QHeaderView,
    QPushButton, QFrame, QGroupBox, QScrollArea
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont


class StatCard(QFrame):
    """Small statistic card widget"""
    
    def __init__(self, title, value="0", unit="", parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet("""
            StatCard {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #2d3748, stop:1 #1a202c);
                border-radius: 10px;
                border: 1px solid #4a5568;
                padding: 15px;
            }
        """)
        
        layout = QVBoxLayout(self)
        layout.setSpacing(5)
        
        # Title
        title_label = QLabel(title)
        title_label.setStyleSheet("color: #a0aec0; font-size: 12px;")
        title_label.setAlignment(Qt.AlignCenter)
        
        # Value
        self.value_label = QLabel(f"{value}{unit}")
        self.value_label.setStyleSheet("color: #00D9FF; font-size: 28px; font-weight: bold;")
        self.value_label.setAlignment(Qt.AlignCenter)
        
        layout.addWidget(title_label)
        layout.addWidget(self.value_label)
        
    def update_value(self, value, unit=""):
        """Update card value"""
        self.value_label.setText(f"{value}{unit}")


class DashboardWidget(QWidget):
    """Dashboard for displaying detection statistics and history"""
    
    def __init__(self, detection_manager, parent=None):
        super().__init__(parent)
        self.detection_manager = detection_manager
        self.init_ui()
        
    def init_ui(self):
        """Initialize UI"""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)
        
        # Title
        title = QLabel("📊 检测数据看板 Detection Dashboard")
        title.setStyleSheet("font-size: 24px; font-weight: bold; color: #00D9FF;")
        title.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title)
        
        # Statistics cards row
        stats_layout = QHBoxLayout()
        stats_layout.setSpacing(15)
        
        self.total_sessions_card = StatCard("总会话数\nTotal Sessions", "0")
        self.total_detections_card = StatCard("总检测数\nTotal Detections", "0")
        self.unique_classes_card = StatCard("检测类别\nUnique Classes", "0")
        self.avg_confidence_card = StatCard("平均置信度\nAvg Confidence", "0.00")
        
        stats_layout.addWidget(self.total_sessions_card)
        stats_layout.addWidget(self.total_detections_card)
        stats_layout.addWidget(self.unique_classes_card)
        stats_layout.addWidget(self.avg_confidence_card)
        
        main_layout.addLayout(stats_layout)
        
        # Class distribution section
        class_group = QGroupBox("类别分布 Class Distribution")
        class_group.setStyleSheet("""
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
        
        class_layout = QVBoxLayout()
        
        self.class_table = QTableWidget()
        self.class_table.setColumnCount(2)
        self.class_table.setHorizontalHeaderLabels(["类别 Class", "数量 Count"])
        self.class_table.horizontalHeader().setStretchLastSection(True)
        self.class_table.setAlternatingRowColors(True)
        self.class_table.setStyleSheet("""
            QTableWidget {
                background-color: #1a202c;
                color: #e2e8f0;
                gridline-color: #4a5568;
                border: none;
            }
            QTableWidget::item {
                padding: 8px;
            }
            QHeaderView::section {
                background-color: #2d3748;
                color: #00D9FF;
                font-weight: bold;
                padding: 8px;
                border: none;
            }
        """)
        
        class_layout.addWidget(self.class_table)
        class_group.setLayout(class_layout)
        main_layout.addWidget(class_group)
        
        # Session history section
        history_group = QGroupBox("检测历史 Detection History")
        history_group.setStyleSheet("""
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
        
        history_layout = QVBoxLayout()
        
        self.history_table = QTableWidget()
        self.history_table.setColumnCount(6)
        self.history_table.setHorizontalHeaderLabels([
            "会话ID\nSession ID", "模式\nMode", "模型\nModel", 
            "帧数\nFrames", "检测数\nDetections", "时间\nTime"
        ])
        self.history_table.horizontalHeader().setStretchLastSection(True)
        self.history_table.setAlternatingRowColors(True)
        self.history_table.setStyleSheet("""
            QTableWidget {
                background-color: #1a202c;
                color: #e2e8f0;
                gridline-color: #4a5568;
                border: none;
            }
            QTableWidget::item {
                padding: 8px;
            }
            QHeaderView::section {
                background-color: #2d3748;
                color: #00D9FF;
                font-weight: bold;
                padding: 8px;
                border: none;
            }
        """)
        
        history_layout.addWidget(self.history_table)
        history_group.setLayout(history_layout)
        main_layout.addWidget(history_group)
        
        # Export buttons
        export_layout = QHBoxLayout()
        export_layout.addStretch()
        
        self.export_json_btn = QPushButton("📄 导出 JSON Export JSON")
        self.export_json_btn.setStyleSheet("""
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
        """)
        
        self.export_csv_btn = QPushButton("📊 导出 CSV Export CSV")
        self.export_csv_btn.setStyleSheet("""
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
        
        self.clear_btn = QPushButton("🗑️ 清空历史 Clear History")
        self.clear_btn.setStyleSheet("""
            QPushButton {
                background-color: #742a2a;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 10px 20px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #9b2c2c;
            }
        """)
        
        export_layout.addWidget(self.export_json_btn)
        export_layout.addWidget(self.export_csv_btn)
        export_layout.addWidget(self.clear_btn)
        
        main_layout.addLayout(export_layout)
        
    def refresh_data(self):
        """Refresh dashboard data from detection manager"""
        sessions = self.detection_manager.get_all_sessions()
        
        # Calculate overall statistics
        total_sessions = len(sessions)
        total_detections = 0
        all_classes = set()
        total_confidence = 0
        confidence_count = 0
        class_counts = {}
        
        for session in sessions:
            stats = session['stats']
            total_detections += stats['total_detections']
            all_classes.update(stats['unique_classes'])
            
            # Aggregate class counts
            for cls, count in stats['class_counts'].items():
                class_counts[cls] = class_counts.get(cls, 0) + count
            
            # Calculate confidence
            for det_frame in session['detections']:
                for det in det_frame['detections']:
                    total_confidence += det['confidence']
                    confidence_count += 1
        
        avg_confidence = total_confidence / confidence_count if confidence_count > 0 else 0
        
        # Update stat cards
        self.total_sessions_card.update_value(total_sessions)
        self.total_detections_card.update_value(total_detections)
        self.unique_classes_card.update_value(len(all_classes))
        self.avg_confidence_card.update_value(f"{avg_confidence:.2f}")
        
        # Update class distribution table
        self.class_table.setRowCount(len(class_counts))
        for i, (cls, count) in enumerate(sorted(class_counts.items(), key=lambda x: x[1], reverse=True)):
            self.class_table.setItem(i, 0, QTableWidgetItem(cls))
            self.class_table.setItem(i, 1, QTableWidgetItem(str(count)))
        
        # Update history table
        self.history_table.setRowCount(len(sessions))
        for i, session in enumerate(reversed(sessions)):  # Most recent first
            stats = session['stats']
            duration = (session['end_time'] - session['start_time']).total_seconds()
            
            self.history_table.setItem(i, 0, QTableWidgetItem(session['session_id']))
            self.history_table.setItem(i, 1, QTableWidgetItem(session['mode']))
            self.history_table.setItem(i, 2, QTableWidgetItem(session['model']))
            self.history_table.setItem(i, 3, QTableWidgetItem(str(stats['total_frames'])))
            self.history_table.setItem(i, 4, QTableWidgetItem(str(stats['total_detections'])))
            self.history_table.setItem(i, 5, QTableWidgetItem(f"{duration:.1f}s"))
