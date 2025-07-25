# Knife Detection in Warehouse CCTV Footage

This project is an AI-based system to detect **unwanted and insecure objects**specifically **knives**from CCTV footage of warehouse environments. It combines computer vision and machine learning to identify knives in static images or video streams, aiming to enhance safety and automate surveillance.

---

## Project Features

- Object detection using YOLOv8
- Upload and detect knives from images
- Real-time detection using webcam or video feed
- RESTful API built with FastAPI
- Custom-trained AI model
- ⚙React-based frontend (optional)

---

## Tech Stack

| Layer        | Tool/Library     |
|--------------|------------------|
| Backend API  | FastAPI          |
| AI Model     | YOLOv5 (Ultralytics) |
| Image & Video Processing | OpenCV      |
| Annotation   | LabelImg         |
| Frontend     | React.js (optional) |
| Language     | Python           |

---


---

## How to Run (Backend)

**Clone the repo**  
```bash
git clone https://github.com/your-username/knife-detector.git
cd knife-detector/backend

**create a virtual environment*
*windows
python -m venv venv  
*Linux/macOS
python3 -m venv venv

**Activate the virtual environment*
*windows
venv\Scripts\activate

*Linux/macOS
source venv/bin/activate


**LABELING THE IMAGES**
first install the label studio  
    pip install label-studio

Next run the label studio
    label-studio start

**Install Dependencies*
pip install -r requirements.txt

**Run the Project** 
uvicorn app.main:app --reload

```


---

## How to Train the Model
```bash
git clone https://github.com/ultralytics/yolov5

cd yolov5

pip install -r requirements.txt
```

```bash
python train.py --img 416 --batch 8 --epochs 30 --data knife.yaml --weights yolov5s.pt --name knife_detector


uvicorn main:app --reload


