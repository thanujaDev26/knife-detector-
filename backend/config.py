import os

class Settings:
    # Path to your trained model (use best.pt from training)
    MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "model.pt")
    
    # Confidence threshold (matches your training command)
    CONFIDENCE_THRESHOLD = 0.25
    
    # Image size (matches your training imgsz)
    IMAGE_SIZE = 640
    
    # Class name (should match your data.yaml)
    CLASS_NAME = "knife"
    
    # Class ID (usually 0 for single-class models)
    CLASS_ID = 0

settings = Settings()