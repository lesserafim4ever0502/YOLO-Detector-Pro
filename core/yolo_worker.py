"""
YOLO Detection Worker - QThread for non-blocking inference
"""

from PySide6.QtCore import QThread, Signal
import cv2
import time
import os
from ultralytics import YOLO


class YOLOWorker(QThread):
    """Worker thread for YOLO detection"""

    # Signals
    frame_processed = Signal(object, list)  # (frame, detections)
    fps_updated = Signal(float)
    stats_updated = Signal(dict)  # {classes, targets, etc}
    error_occurred = Signal(str)
    # (current, total) for batch processing
    progress_updated = Signal(int, int)
    # (filepath, frame, detections)
    batch_item_processed = Signal(str, object, list)

    def __init__(self, model_name="yolo11n.pt"):
        super().__init__()
        self.model_name = model_name
        self.model = None
        self.running = False

        # Detection settings
        self.conf_threshold = 0.25
        self.iou_threshold = 0.45
        self.line_width = 1
        self.inference_delay = 0  # in ms
        self.inference_enabled = True  # Control actual detection

        # Source settings
        self.source = None  # Can be image path, video path, or camera index
        self.mode = "image"  # image, folder, camera, video

    def _load_model(self):
        """Helper to load YOLO model with correct path"""
        try:
            # Check if model exists in models directory
            model_path = os.path.join("models", f"{self.model_name}.pt")
            if os.path.exists(model_path):
                self.model = YOLO(model_path)
            else:
                # Fallback to current directory or download
                self.model = YOLO(f"{self.model_name}.pt")
        except Exception as e:
            self.error_occurred.emit(f"Error loading model: {str(e)}")
            raise e

    def set_model(self, model_name):
        """Set YOLO model"""
        self.model_name = model_name
        self.model = None  # Reset model to force reload

    def set_parameters(self, conf, iou, line_width=1, delay=0):
        """Set detection parameters"""
        self.conf_threshold = conf
        self.iou_threshold = iou
        self.line_width = line_width
        self.inference_delay = delay

    def set_inference_enabled(self, enabled):
        """Enable/disable inference (for camera preview)"""
        self.inference_enabled = enabled

    def set_source(self, source, mode="image"):
        """Set detection source"""
        self.source = source
        self.mode = mode

    def run(self):
        """Main detection loop"""
        self.running = True

        try:
            # Load model if not loaded
            if self.model is None and self.inference_enabled:
                self._load_model()

            if self.mode == "image":
                self._process_image()
            elif self.mode == "camera":
                self._process_camera()
            elif self.mode == "video":
                self._process_video()
            elif self.mode == "folder":
                self._process_folder()

        except Exception as e:
            self.error_occurred.emit(str(e))
        finally:
            self.running = False

    def _process_image(self):
        """Process single image"""
        if self.source is None:
            return

        try:
            frame = cv2.imread(self.source)
            if frame is None:
                self.error_occurred.emit(
                    f"Failed to load image: {self.source}")
                return

            # Lazy load model if needed
            if self.model is None:
                self._load_model()

            results = self.model.predict(
                frame,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                verbose=False
            )

            # Extract detections
            detections = self._extract_detections(results[0])

            # Emit results
            self.frame_processed.emit(frame, detections)
            self._update_stats(detections)

            # Application of delay for image mode (might not be needed but consistent)
            if self.inference_delay > 0:
                time.sleep(self.inference_delay / 1000.0)

        except Exception as e:
            self.error_occurred.emit(f"Image processing error: {str(e)}")

    def _process_video(self):
        """Process video file and auto-save annotated results"""
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            self.error_occurred.emit(f"Failed to open video: {self.source}")
            return

        fps_counter = 0
        start_time = time.time()

        # Auto-save setup
        output_writer = None
        from utils.helper import draw_detections

        try:
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    break

                # Run inference
                results = self.model.predict(
                    frame,
                    conf=self.conf_threshold,
                    iou=self.iou_threshold,
                    verbose=False
                )

                # Extract detections
                detections = self._extract_detections(results[0])

                # Draw detections for auto-save
                annotated_frame = draw_detections(
                    frame, detections, line_width=self.line_width)

                # Initialize video writer on first frame
                if output_writer is None:
                    os.makedirs("results", exist_ok=True)
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    output_path = os.path.join(
                        "results", f"output_{timestamp}.mp4")

                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    h, w = frame.shape[:2]
                    # Using 20 FPS as default, could be fetched from cap.get(cv2.CAP_PROP_FPS)
                    src_fps = cap.get(cv2.CAP_PROP_FPS)
                    if src_fps <= 0:
                        src_fps = 20.0

                    output_writer = cv2.VideoWriter(
                        output_path, fourcc, src_fps, (w, h))

                # Write to output file
                output_writer.write(annotated_frame)

                # Calculate FPS for UI
                fps_counter += 1
                if fps_counter >= 10:
                    elapsed = time.time() - start_time
                    fps = fps_counter / elapsed
                    self.fps_updated.emit(fps)
                    fps_counter = 0
                    start_time = time.time()

                # Emit results for UI
                self.frame_processed.emit(frame, detections)
                self._update_stats(detections)

                # Apply inference delay
                if self.inference_delay > 0:
                    time.sleep(self.inference_delay / 1000.0)

        except Exception as e:
            self.error_occurred.emit(f"Video error: {str(e)}")
        finally:
            if output_writer:
                output_writer.release()
            cap.release()

    def _process_camera(self):
        """Process camera stream"""
        cap = cv2.VideoCapture(self.source if self.source is not None else 0)
        fps_counter = 0
        start_time = time.time()

        try:
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    break

                detections = []

                # Run inference only if enabled
                if self.inference_enabled:
                    if self.model is None:
                        self._load_model()

                    results = self.model.predict(
                        frame,
                        conf=self.conf_threshold,
                        iou=self.iou_threshold,
                        verbose=False
                    )
                    # Extract detections
                    detections = self._extract_detections(results[0])

                # Calculate FPS
                fps_counter += 1
                if fps_counter >= 10:
                    elapsed = time.time() - start_time
                    fps = fps_counter / elapsed
                    self.fps_updated.emit(fps)
                    fps_counter = 0
                    start_time = time.time()

                # Emit results
                self.frame_processed.emit(frame, detections)

                if self.inference_enabled:
                    self._update_stats(detections)

                # Apply inference delay
                if self.inference_delay > 0:
                    time.sleep(self.inference_delay / 1000.0)

        except Exception as e:
            self.error_occurred.emit(f"Camera error: {str(e)}")
        finally:
            cap.release()

    def _process_folder(self):
        """Process folder of images"""
        if self.source is None or not os.path.isdir(self.source):
            self.error_occurred.emit("Invalid folder path")
            return

        try:
            # Get all image files
            from utils.helper import get_image_files
            image_files = get_image_files(self.source)

            if not image_files:
                self.error_occurred.emit("No image files found in folder")
                return

            total = len(image_files)

            # Process each image
            for i, filepath in enumerate(image_files):
                if not self.running:
                    break

                # Update progress
                self.progress_updated.emit(i + 1, total)

                # Load and process image
                frame = cv2.imread(filepath)
                if frame is None:
                    continue

                # Run inference
                results = self.model.predict(
                    frame,
                    conf=self.conf_threshold,
                    iou=self.iou_threshold,
                    verbose=False
                )

                # Extract detections
                detections = self._extract_detections(results[0])

                # Emit individual result
                self.batch_item_processed.emit(filepath, frame, detections)
                self._update_stats(detections)

                # Apply inference delay
                if self.inference_delay > 0:
                    print(
                        f"Applying inference delay: {self.inference_delay}ms")
                    time.sleep(self.inference_delay / 1000.0)

        except Exception as e:
            self.error_occurred.emit(f"Folder processing error: {str(e)}")

    def _extract_detections(self, result):
        """Extract detection information from YOLO result"""
        detections = []

        if result.boxes is not None:
            boxes = result.boxes
            for i in range(len(boxes)):
                detection = {
                    'bbox': boxes.xyxy[i].cpu().numpy(),
                    'confidence': float(boxes.conf[i].cpu().numpy()),
                    'class_id': int(boxes.cls[i].cpu().numpy()),
                    'class_name': result.names[int(boxes.cls[i])]
                }
                detections.append(detection)

        return detections

    def _update_stats(self, detections):
        """Update and emit statistics"""
        unique_classes = set(d['class_id'] for d in detections)

        stats = {
            'classes': len(unique_classes),
            'targets': len(detections)
        }

        self.stats_updated.emit(stats)

    def stop(self):
        """Stop detection"""
        self.running = False
        self.wait()
