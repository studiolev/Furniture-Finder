from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import io

class FurnitureDetector:
    def __init__(self):
        # Initialize YOLO model
        self.model = YOLO('yolov8n.pt')  # Using YOLOv8 nano model
        
        # Define furniture classes (you can modify this list based on your needs)
        self.furniture_classes = [
            'chair', 'couch', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
            'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
            'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
            'hair drier', 'toothbrush'
        ]
    
    def detect_furniture(self, image, confidence_threshold=0.5):
        """
        Detect furniture in the image
        Args:
            image: PIL Image or numpy array
            confidence_threshold: float between 0 and 1
        Returns:
            dict with detections and processed image
        """
        # Convert PIL Image to numpy array if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Run YOLO detection
        results = self.model(image, conf=confidence_threshold)[0]
        
        # Process results
        detections = []
        for box in results.boxes:
            # Get box coordinates
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = results.names[class_id]
            
            # Only include furniture classes
            if class_name.lower() in self.furniture_classes:
                detections.append({
                    'class': class_name,
                    'confidence': confidence,
                    'bbox': [int(x1), int(y1), int(x2), int(y2)]
                })
        
        # Draw boxes on image
        processed_image = image.copy()
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            cv2.rectangle(processed_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{detection['class']} {detection['confidence']:.2f}"
            cv2.putText(processed_image, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Convert back to PIL Image
        processed_image = Image.fromarray(processed_image)
        
        return {
            'detections': detections,
            'image': processed_image
        } 