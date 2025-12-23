"""
Comparison Worker - QThread for dual-model inference
"""

from PySide6.QtCore import QThread, Signal
import cv2
import time
import os
from ultralytics import YOLO

class ComparisonWorker(QThread):
    """Worker thread for running two YOLO models simultaneously"""
    
    # Signals
    # Video/Image: (frame, detections_A, detections_B, inference_time_A_ms, inference_time_B_ms)
    frame_processed = Signal(object, list, list, float, float)
    
    # Stats: (fps_A, fps_B)
    fps_updated = Signal(float, float) 
    
    # Batch processing
    progress_updated = Signal(int, int)
    batch_item_processed = Signal(str, object, list, list)
    
    finished = Signal()
    error_occurred = Signal(str)
    
    def __init__(self, model_a_name, model_b_name):
        super().__init__()
        self.model_a_name = model_a_name
        self.model_b_name = model_b_name
        self.model_a = None
        self.model_b = None
        self.running = False
        
        # Settings
        self.conf = 0.25
        self.iou = 0.45
        self.line_width = 1
        self.inference_delay = 0
        
        self.source = None
        self.mode = "image"
        
    def set_parameters(self, conf, iou, line_width=1, delay=0):
        self.conf = conf
        self.iou = iou
        self.line_width = line_width
        self.inference_delay = delay
        
    def set_source(self, source, mode="image"):
        self.source = source
        self.mode = mode
        
    def _load_model(self, name):
        """Load a single model"""
        try:
            model_path = os.path.join("models", f"{name}.pt")
            if os.path.exists(model_path):
                return YOLO(model_path)
            else:
                return YOLO(f"{name}.pt")
        except Exception as e:
            raise Exception(f"Failed to load model {name}: {str(e)}")

    def run(self):
        self.running = True
        try:
            # Load models
            if not self.model_a:
                self.model_a = self._load_model(self.model_a_name)
            if not self.model_b:
                self.model_b = self._load_model(self.model_b_name)
                
            if self.mode == "video":
                self._process_video()
            elif self.mode == "folder":
                self._process_folder()
            elif self.mode == "image":
                self._process_image()
                
        except Exception as e:
            self.error_occurred.emit(str(e))
        finally:
            self.running = False
            self.finished.emit()
            
    def _process_video(self):
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            raise Exception(f"Failed to open video: {self.source}")
            
        fps_counter = 0
        sum_time_a = 0
        sum_time_b = 0
        
        try:
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Model A
                start_a = time.time()
                res_a = self.model_a.predict(frame, conf=self.conf, iou=self.iou, verbose=False)
                time_a = (time.time() - start_a) * 1000
                det_a = self._extract_detections(res_a[0])
                
                # Model B
                start_b = time.time()
                res_b = self.model_b.predict(frame, conf=self.conf, iou=self.iou, verbose=False)
                time_b = (time.time() - start_b) * 1000
                det_b = self._extract_detections(res_b[0])
                
                # Emit
                self.frame_processed.emit(frame, det_a, det_b, time_a, time_b)
                
                # Delay
                if self.inference_delay > 0:
                    time.sleep(self.inference_delay / 1000.0)
                
                # FPS
                fps_counter += 1
                sum_time_a += time_a
                sum_time_b += time_b
                
                if fps_counter >= 5:
                    avg_a = sum_time_a / fps_counter
                    avg_b = sum_time_b / fps_counter
                    # Theoretical FPS based on inference only
                    fps_a = 1000.0 / avg_a if avg_a > 0 else 0
                    fps_b = 1000.0 / avg_b if avg_b > 0 else 0
                    self.fps_updated.emit(fps_a, fps_b)
                    
                    fps_counter = 0
                    sum_time_a = 0
                    sum_time_b = 0
                    
        finally:
            cap.release()
            
    def _process_folder(self):
        if not os.path.isdir(self.source):
            raise Exception("Invalid folder path")
            
        from utils.helper import get_image_files
        files = get_image_files(self.source)
        total = len(files)
        
        for i, filepath in enumerate(files):
            if not self.running:
                break
                
            self.progress_updated.emit(i+1, total)
            
            frame = cv2.imread(filepath)
            if frame is None:
                continue
                
            # Model A
            start_a = time.time()
            res_a = self.model_a.predict(frame, conf=self.conf, iou=self.iou, verbose=False)
            time_a = (time.time() - start_a) * 1000
            det_a = self._extract_detections(res_a[0])
            
            # Model B
            start_b = time.time()
            res_b = self.model_b.predict(frame, conf=self.conf, iou=self.iou, verbose=False)
            time_b = (time.time() - start_b) * 1000
            det_b = self._extract_detections(res_b[0])
            
            # Emit for visualization (reuse video signal)
            self.frame_processed.emit(frame, det_a, det_b, time_a, time_b)
            self.batch_item_processed.emit(filepath, frame, det_a, det_b)
            
            # Delay
            if self.inference_delay > 0:
                time.sleep(self.inference_delay / 1000.0)
            
    def _process_image(self):
        # Single image processing
        frame = cv2.imread(self.source)
        if frame is None:
            raise Exception("Failed to load image")
            
        start_a = time.time()
        res_a = self.model_a.predict(frame, conf=self.conf, iou=self.iou, verbose=False)
        time_a = (time.time() - start_a) * 1000
        det_a = self._extract_detections(res_a[0])
        
        start_b = time.time()
        res_b = self.model_b.predict(frame, conf=self.conf, iou=self.iou, verbose=False)
        time_b = (time.time() - start_b) * 1000
        det_b = self._extract_detections(res_b[0])
        
        self.frame_processed.emit(frame, det_a, det_b, time_a, time_b)
        
        if self.inference_delay > 0:
            time.sleep(self.inference_delay / 1000.0)

    def _extract_detections(self, result):
        detections = []
        if result.boxes is not None:
            boxes = result.boxes
            for i in range(len(boxes)):
                detections.append({
                    'bbox': boxes.xyxy[i].cpu().numpy(),
                    'confidence': float(boxes.conf[i].cpu().numpy()),
                    'class_id': int(boxes.cls[i].cpu().numpy()),
                    'class_name': result.names[int(boxes.cls[i])]
                })
        return detections

    def stop(self):
        self.running = False
        self.wait()
