from fastapi import APIRouter, UploadFile, File
from app.services.inference import detect_knife
import shutil

router = APIRouter()

@router.post("/detect")
async def detect(file: UploadFile = File(...)):
    with open("temp.jpg", "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    results = detect_knife("temp.jpg")
    return results
