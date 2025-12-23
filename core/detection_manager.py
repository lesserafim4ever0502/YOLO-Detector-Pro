"""
Detection Manager - Handles detection history and data management
"""

from datetime import datetime
import json
import csv
import numpy as np
from typing import List, Dict


class DetectionEncoder(json.JSONEncoder):
    """Custom JSON encoder for detection data"""
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


class DetectionManager:
    """Manages detection history and provides data for analytics"""
    
    def __init__(self):
        self.detection_history = []  # List of detection sessions
        self.current_session = None
        
    def start_session(self, mode, model_name):
        """Start a new detection session"""
        self.current_session = {
            'session_id': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'mode': mode,
            'model': model_name,
            'start_time': datetime.now(),
            'detections': [],
            'stats': {
                'total_frames': 0,
                'total_detections': 0,
                'unique_classes': set(),
                'avg_confidence': 0.0,
                'class_counts': {}
            }
        }
        
    def add_detection(self, frame_id, detections):
        """Add detection result to current session"""
        if self.current_session is None:
            return
            
        self.current_session['detections'].append({
            'frame_id': frame_id,
            'timestamp': datetime.now(),
            'detections': detections
        })
        
        # Update statistics
        stats = self.current_session['stats']
        stats['total_frames'] += 1
        stats['total_detections'] += len(detections)
        
        for det in detections:
            class_name = det['class_name']
            stats['unique_classes'].add(class_name)
            stats['class_counts'][class_name] = stats['class_counts'].get(class_name, 0) + 1
            
    def end_session(self):
        """End current detection session and add to history"""
        if self.current_session is None:
            return
            
        self.current_session['end_time'] = datetime.now()
        
        # Convert set to list for JSON serialization
        stats = self.current_session['stats']
        stats['unique_classes'] = list(stats['unique_classes'])
        
        # Calculate average confidence
        total_conf = 0
        total_count = 0
        for det_frame in self.current_session['detections']:
            for det in det_frame['detections']:
                total_conf += det['confidence']
                total_count += 1
        
        if total_count > 0:
            stats['avg_confidence'] = total_conf / total_count
            
        self.detection_history.append(self.current_session)
        self.current_session = None
        
    def get_session_stats(self):
        """Get statistics for current session"""
        if self.current_session is None:
            return None
        return self.current_session['stats']
        
    def get_all_sessions(self):
        """Get all detection sessions"""
        return self.detection_history
        
    def export_to_json(self, filepath):
        """Export detection history to JSON file"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.detection_history, f, indent=2, ensure_ascii=False, cls=DetectionEncoder)
            
    def export_to_csv(self, filepath):
        """Export detection summary to CSV file"""
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Session ID', 'Mode', 'Model', 'Start Time', 'End Time',
                'Total Frames', 'Total Detections', 'Unique Classes', 'Avg Confidence'
            ])
            
            for session in self.detection_history:
                stats = session['stats']
                writer.writerow([
                    session['session_id'],
                    session['mode'],
                    session['model'],
                    session['start_time'].strftime("%Y-%m-%d %H:%M:%S"),
                    session['end_time'].strftime("%Y-%m-%d %H:%M:%S"),
                    stats['total_frames'],
                    stats['total_detections'],
                    len(stats['unique_classes']),
                    f"{stats['avg_confidence']:.2f}"
                ])
                
    def clear_history(self):
        """Clear all detection history"""
        self.detection_history = []
        self.current_session = None
