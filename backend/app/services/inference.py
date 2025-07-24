from ultralytics import YOLO

model = YOLO("app/models/knife_model.pt")

def detect_knife(image_path: str):
    results = model.predict(source=image_path, save=False)
    detections = results[0].boxes
    objects = []
    for box in detections:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        bbox = box.xyxy[0].tolist()
        objects.append({
            "class": model.names[cls],
            "confidence": conf,
            "box": bbox
        })
    return {"detections": objects}
