"""
Inference API for the crack classifier.

Serves the ResNet18 checkpoint written by model_training.py (best_model.pth)
and, if the web UI has been built (web/dist), the UI itself.

Run from the repo root:
    uvicorn api.main:app --port 8000
"""

from __future__ import annotations

import io
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import torch
import torch.nn as nn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = Path(os.environ.get("CRACK_MODEL_PATH", ROOT / "best_model.pth"))
WEB_DIST = ROOT / "web" / "dist"

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
# Pillow warns above this many pixels and raises above twice as many.
Image.MAX_IMAGE_PIXELS = 40_000_000

# Must match test_transform in data_preprocessing.py.
preprocess = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model: nn.Module | None = None


def load_model(path: Path) -> nn.Module | None:
    if not path.is_file():
        return None
    net = models.resnet18(weights=None)
    net.fc = nn.Linear(net.fc.in_features, 2)
    net.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    return net.to(device).eval()


@asynccontextmanager
async def lifespan(_: FastAPI):
    global model
    model = load_model(MODEL_PATH)
    yield


app = FastAPI(title="Crack Analysis API", lifespan=lifespan)

# Only needed when the UI is hosted on a different origin (e.g. Vercel).
# Comma-separated list, e.g. CRACK_CORS_ORIGINS=https://crack-analysis.vercel.app
cors_origins = [o.strip() for o in os.environ.get("CRACK_CORS_ORIGINS", "").split(",") if o.strip()]
if cors_origins:
    app.add_middleware(CORSMiddleware, allow_origins=cors_origins, allow_methods=["GET", "POST"])


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok" if model is not None else "no_model",
        "model_loaded": model is not None,
        "device": device.type,
    }


@app.post("/api/predict")
def predict(file: UploadFile = File(...)) -> dict:
    if model is None:
        raise HTTPException(
            503,
            "Model weights not found. Train the model with model_training.py "
            "to produce best_model.pth, then restart the API.",
        )

    data = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Image is larger than 10 MB.")

    try:
        image = Image.open(io.BytesIO(data))
        width, height = image.size
        image = image.convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise HTTPException(400, "That file could not be read as an image.")

    started = time.perf_counter()
    batch = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(batch), dim=1)[0].tolist()
    elapsed_ms = (time.perf_counter() - started) * 1000

    # Label 1 = cracked, matching 3_data_cleaning.py.
    p_no_crack, p_crack = probs
    is_crack = p_crack >= p_no_crack
    return {
        "label": "crack" if is_crack else "no_crack",
        "confidence": p_crack if is_crack else p_no_crack,
        "probabilities": {"no_crack": p_no_crack, "crack": p_crack},
        "inference_ms": round(elapsed_ms, 1),
        "width": width,
        "height": height,
        "device": device.type,
    }


# Registered last so it never shadows /api routes.
if WEB_DIST.is_dir():
    app.mount("/", StaticFiles(directory=WEB_DIST, html=True), name="web")
