from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from PIL import Image
import io
from config import settings
from image_utils import preprocess_image, draw_detections

router = APIRouter()

@router.post("/detect")
async def detect_knife(request: Request, file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "Supported formats: JPEG, PNG, WEBP")

    try:
        contents = await file.read()
        pil_image = Image.open(io.BytesIO(contents))
        np_image = preprocess_image(pil_image)

        model = request.app.state.model  # ✅ Access model from app state

        results = model.predict(
            np_image,
            conf=settings.CONFIDENCE_THRESHOLD,
            imgsz=settings.IMAGE_SIZE,
            classes=[settings.CLASS_ID]
        )

        detections = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                detections.append({
                    "class": settings.CLASS_NAME,
                    "confidence": round(float(box.conf[0]), 4),
                    "box": [x1, y1, x2, y2]
                })

        return JSONResponse(content={
            "detections": detections,
            "count": len(detections),
            "image_size": [np_image.shape[1], np_image.shape[0]]
        })

    except Exception as e:
        raise HTTPException(500, f"Processing error: {str(e)}")

@router.post("/detect-visual")
async def detect_knife_visual(request: Request, file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "Supported formats: JPEG, PNG, WEBP")

    try:
        contents = await file.read()
        pil_image = Image.open(io.BytesIO(contents))
        original_image = preprocess_image(pil_image)
        display_image = original_image.copy()

        model = request.app.state.model  # ✅ Access model here too

        results = model.predict(
            original_image,
            conf=settings.CONFIDENCE_THRESHOLD,
            imgsz=settings.IMAGE_SIZE,
            classes=[settings.CLASS_ID]
        )

        detections = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                detections.append({
                    "class": settings.CLASS_NAME,
                    "confidence": float(box.conf[0]),
                    "box": [x1, y1, x2, y2]
                })

        output_image = draw_detections(display_image, detections)

        return StreamingResponse(io.BytesIO(output_image), media_type="image/jpeg")

    except Exception as e:
        raise HTTPException(500, f"Processing error: {str(e)}")
