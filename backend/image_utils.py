import cv2
import numpy as np
from PIL import Image

def preprocess_image(image: Image.Image) -> np.ndarray:
    """Convert PIL image to OpenCV format (BGR)"""
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

def draw_detections(image: np.ndarray, detections: list) -> bytes:
    """Draw bounding boxes on image and return as JPEG bytes"""
    img_h, img_w = image.shape[:2]
    
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        conf = det["confidence"]
        label = det["class"]
        
        # Scale coordinates if needed
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # Draw rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw label background
        label_text = f"{label} {conf:.2f}"
        (text_width, text_height), _ = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        cv2.rectangle(
            image,
            (x1, y1 - text_height - 15),
            (x1 + text_width, y1),
            (0, 255, 0),
            -1
        )
        
        # Put text
        cv2.putText(
            image,
            label_text,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2
        )
    
    # Convert to JPEG bytes
    _, encoded_img = cv2.imencode(".jpg", image)
    return encoded_img.tobytes()