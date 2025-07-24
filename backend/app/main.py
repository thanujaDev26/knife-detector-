from fastapi import FastAPI
from app.routes import detect

app = FastAPI(title="Knife Detection API")

app.include_router(detect.router)
