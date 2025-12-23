"""
Utility Helper Functions
"""

import cv2
import numpy as np
from datetime import datetime
import os


def draw_detections(frame, detections, color_map=None, line_width=2):
    """
    Draw bounding boxes and labels on frame
    
    Args:
        frame: Input image (numpy array)
        detections: List of detection dictionaries
        color_map: Dict mapping class_id to BGR color
        line_width: Width of bounding box lines
    
    Returns:
        Annotated frame
    """
    annotated = frame.copy()
    
    if color_map is None:
        # Default color map
        np.random.seed(42)
        color_map = {}
    
    for det in detections:
        bbox = det['bbox'].astype(int)
        class_name = det['class_name']
        confidence = det['confidence']
        class_id = det['class_id']
        
        # Get or generate color
        if class_id not in color_map:
            color_map[class_id] = tuple(
                int(c) for c in np.random.randint(0, 255, 3)
            )
        color = color_map[class_id]
        
        # Draw bounding box
        cv2.rectangle(
            annotated,
            (bbox[0], bbox[1]),
            (bbox[2], bbox[3]),
            color,
            line_width
        )
        
        # Prepare label
        label = f"{class_name} {confidence:.2f}"
        
        # Calculate label size
        (label_w, label_h), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            1
        )
        
        # Draw label background
        cv2.rectangle(
            annotated,
            (bbox[0], bbox[1] - label_h - baseline - 5),
            (bbox[0] + label_w, bbox[1]),
            color,
            -1
        )
        
        # Draw label text
        cv2.putText(
            annotated,
            label,
            (bbox[0], bbox[1] - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )
    
    return annotated


def resize_with_aspect_ratio(image, target_width=None, target_height=None):
    """
    Resize image while maintaining aspect ratio
    
    Args:
        image: Input image
        target_width: Target width (optional)
        target_height: Target height (optional)
    
    Returns:
        Resized image
    """
    h, w = image.shape[:2]
    
    if target_width is None and target_height is None:
        return image
    
    if target_width is None:
        ratio = target_height / h
        target_width = int(w * ratio)
    elif target_height is None:
        ratio = target_width / w
        target_height = int(h * ratio)
    else:
        # Fit within both constraints
        ratio = min(target_width / w, target_height / h)
        target_width = int(w * ratio)
        target_height = int(h * ratio)
    
    resized = cv2.resize(image, (target_width, target_height))
    return resized


def save_detection_result(frame, detections, output_dir="results", line_width=2):
    """
    Save detection result with timestamp
    
    Args:
        frame: Annotated frame
        detections: List of detections
        output_dir: Output directory
        line_width: Width of bounding box lines
    
    Returns:
        Saved file path
    """
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"detection_{timestamp}.jpg"
    filepath = os.path.join(output_dir, filename)
    
    # Draw detections on frame
    annotated = draw_detections(frame, detections, line_width=line_width)
    
    # Save image
    cv2.imwrite(filepath, annotated)
    
    # Save detection info as text
    txt_path = filepath.replace('.jpg', '.txt')
    with open(txt_path, 'w') as f:
        f.write(f"Detection Results - {timestamp}\n")
        f.write(f"Total Detections: {len(detections)}\n\n")
        
        for i, det in enumerate(detections, 1):
            f.write(f"{i}. {det['class_name']}: {det['confidence']:.2f}\n")
    
    return filepath


def get_available_cameras():
    """
    Get list of available camera indices
    
    Returns:
        List of available camera indices
    """
    available = []
    
    for i in range(10):  # Check first 10 indices
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available.append(i)
            cap.release()
    
    return available


def numpy_to_qpixmap(image):
    """
    Convert numpy array to QPixmap
    
    Args:
        image: Numpy array (BGR format)
    
    Returns:
        QPixmap
    """
    from PySide6.QtGui import QImage, QPixmap
    
    # Convert BGR to RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    h, w, ch = rgb_image.shape
    bytes_per_line = ch * w
    
    # Create QImage
    q_image = QImage(
        rgb_image.data,
        w,
        h,
        bytes_per_line,
        QImage.Format_RGB888
    )
    
    # Convert to QPixmap
    return QPixmap.fromImage(q_image)


def get_image_files(directory):
    """
    Get all image files from a directory
    
    Args:
        directory: Path to directory
    
    Returns:
        List of image file paths
    """
    supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    image_files = []
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if any(file.lower().endswith(fmt) for fmt in supported_formats):
                image_files.append(os.path.join(root, file))
    
    return sorted(image_files)


def save_batch_results(results, output_dir="results/batch", line_width=2):
    """
    Save batch detection results
    
    Args:
        results: List of (filepath, frame, detections) tuples
        output_dir: Output directory
        line_width: Width of bounding box lines
    
    Returns:
        Directory path where results were saved
    """
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_dir = os.path.join(output_dir, f"batch_{timestamp}")
    os.makedirs(batch_dir, exist_ok=True)
    
    # Save each annotated image
    for i, (filepath, frame, detections) in enumerate(results):
        # Draw detections
        annotated = draw_detections(frame, detections, line_width=line_width)
        
        # Create filename
        original_name = os.path.splitext(os.path.basename(filepath))[0]
        output_path = os.path.join(batch_dir, f"{original_name}_detected.jpg")
        
        # Save image
        cv2.imwrite(output_path, annotated)
    
    # Create summary report
    _create_batch_report(results, batch_dir, timestamp)
    
    return batch_dir


def _create_batch_report(results, output_dir, timestamp):
    """Create a summary report for batch detection"""
    report_path = os.path.join(output_dir, "detection_report.txt")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"Batch Detection Report\n")
        f.write(f"Generated: {timestamp}\n")
        f.write(f"=" * 60 + "\n\n")
        f.write(f"Total Images Processed: {len(results)}\n\n")
        
        total_detections = sum(len(dets) for _, _, dets in results)
        f.write(f"Total Detections: {total_detections}\n\n")
        
        # Per-image breakdown
        f.write("Per-Image Results:\n")
        f.write("-" * 60 + "\n")
        
        for i, (filepath, _, detections) in enumerate(results, 1):
            filename = os.path.basename(filepath)
            f.write(f"\n{i}. {filename}\n")
            f.write(f"   Detections: {len(detections)}\n")
            
            if detections:
                f.write(f"   Classes: ")
                classes = list(set(d['class_name'] for d in detections))
                f.write(", ".join(classes) + "\n")
                
                for j, det in enumerate(detections, 1):
                    f.write(f"      {j}. {det['class_name']}: {det['confidence']:.2f}\n")


def save_detection_json(detections, output_path):
    """
    Save detections in JSON format
    
    Args:
        detections: List of detection dictionaries
        output_path: Output file path
    """
    import json
    
    # Convert numpy arrays to lists for JSON serialization
    json_data = []
    for det in detections:
        json_det = {
            'class_name': det['class_name'],
            'class_id': int(det['class_id']),
            'confidence': float(det['confidence']),
            'bbox': det['bbox'].tolist()
        }
        json_data.append(json_det)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)


def is_image_file(filepath):
    """
    Check if file is a supported image format
    
    Args:
        filepath: Path to file
    
    Returns:
        Boolean
    """
    supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']

def get_available_models(models_dir="models"):
    """
    Get list of available YOLO model files (.pt)
    
    Args:
        models_dir: Directory containing model files (default: "models")
    
    Returns:
        List of model names (without .pt extension)
    """
    if not os.path.exists(models_dir):
        return ["YOLOv8n"]  # Return default if directory doesn't exist
        
    models = []
    for file in os.listdir(models_dir):
        if file.endswith('.pt'):
            models.append(os.path.splitext(file)[0])
            
    # Sort models, but prioritize specific versions if needed
    models.sort()
    
    # If no models found, return default
    if not models:
        return ["YOLOv8n"]
        
    return models
