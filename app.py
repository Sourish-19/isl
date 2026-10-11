"""
FastAPI Server Entry Point for Indian Sign Language Recognition & Translation System.
Serves the REST API, WebSocket streams, and responsive web frontend.
"""

import os
import sys
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from core.model_engine import ISLModelEngine
from core.phrase_buffer import PhraseBuffer
from api.routes import router, init_api_dependencies

app = FastAPI(
    title="Indian Sign Language (ISL) Recognition Platform",
    description="Real-Time Computer Vision and Deep Learning System for ISL Alphanumeric & Phrase Recognition",
    version="2.0.0"
)

# Enable CORS for open integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Base directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")

# Ensure static directory exists
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(os.path.join(STATIC_DIR, "reference"), exist_ok=True)

# Mount static assets
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Initialize models and buffers
model_engine = ISLModelEngine()
phrase_buffer = PhraseBuffer(window_size=10, hold_threshold=15, cooldown_frames=18, confidence_threshold=0.50)
init_api_dependencies(model_engine, phrase_buffer)

# Register API routes
app.include_router(router, prefix="/api")


@app.get("/")
def serve_index():
    """Serves the main web application frontend."""
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {
        "message": "Indian Sign Language Recognition API is active. Frontend index.html initializing...",
        "api_docs": "/docs",
        "health": "/api/health"
    }


if __name__ == "__main__":
    print("[INFO] Launching ISL Recognition Web Application on http://localhost:8000 ...")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
