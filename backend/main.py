from fastapi import FastAPI
from config import settings
from ultralytics import YOLO
import os
from routes import detect

app = FastAPI(
    title="Knife Detection API",
    description="API for detecting knives using YOLO",
    version="1.0.0"
)

# Verify model exists before loading
if not os.path.exists(settings.MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {settings.MODEL_PATH}")

# Load model during startup
@app.on_event("startup")
async def load_model():
    try:
        app.state.model = YOLO(settings.MODEL_PATH)
        app.state.model.fuse()  # Optimize model
        print(f"✅ Model loaded: {settings.MODEL_PATH}")
        print(f"   Classes: {app.state.model.names}")
        print(f"   Confidence threshold: {settings.CONFIDENCE_THRESHOLD}")
        print(f"   Image size: {settings.IMAGE_SIZE}")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        raise RuntimeError(f"Model loading failed: {str(e)}")

# Health check endpoint
@app.get("/")
async def health_check():
    return {
        "status": "active",
        "model": "YOLOv8",
        "class_name": settings.CLASS_NAME,
        "confidence_threshold": settings.CONFIDENCE_THRESHOLD,
        "image_size": settings.IMAGE_SIZE
    }

# Include knife detection routes
app.include_router(detect.router)
