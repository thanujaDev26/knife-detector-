from ultralytics import YOLO


model = YOLO("yolov8n.pt")
model.train(data="data/knife.yaml", epochs=30, imgsz=416, name="knife_model")
