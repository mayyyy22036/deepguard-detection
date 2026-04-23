"""
FastAPI Backend pour DeepGuard - Détection de Deepfakes
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch
import torch.nn as nn
import timm
import cv2
import numpy as np
from PIL import Image
import io
import time
from pathlib import Path

import albumentations as A
from albumentations.pytorch import ToTensorV2


# ==================== PATHS ====================
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR.parent / "models" / "deepguard_xception.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 299


# ==================== APP ====================
app = FastAPI(
    title="DeepGuard API",
    description="API de détection de deepfakes",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== TRANSFORM ====================
transform = A.Compose([
    A.Resize(299, 299),
    A.Normalize(mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]),
    ToTensorV2()
])

# ==================== MODEL ====================
# ==================== MODEL ====================
class DeepfakeClassifierNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            'xception',
            pretrained=False,
            num_classes=0,
            global_pool='avg'  # ← 'avg' comme sur Kaggle
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),        # ← 0.5 comme sur Kaggle
            nn.Linear(2048, 2)
        )

    def forward(self, x):
        features = self.backbone(x)  # (B, 2048) directement
        return self.classifier(features)
# ==================== MODEL LOADER ====================
class ModelLoader:
    _model = None

    @classmethod
    def get_model(cls):
        if cls._model is None:

            if not MODEL_PATH.exists():
                raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

            print("Loading model from:", MODEL_PATH)

            model = DeepfakeClassifierNew()

            checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
            model.load_state_dict(checkpoint)

            model.to(DEVICE)
            model.eval()

            cls._model = model

        return cls._model


# ==================== IMAGE PROCESS ====================
def process_image(image_bytes: bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return np.array(image)


# ==================== PREDICT ====================
def predict_image(image_np):

    model = ModelLoader.get_model()

    transformed = transform(image=image_np)
    image_tensor = transformed["image"].unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        prediction = "Fake" if pred_idx == 1 else "Real"
        confidence = probs[0][pred_idx].item() * 100

        return {
            "prediction": prediction,
            "confidence": round(confidence, 2),
            "probabilities": {
                "real": round(probs[0][0].item() * 100, 2),
                "fake": round(probs[0][1].item() * 100, 2)
            }
        }


# ==================== ROUTES ====================
@app.get("/")
def root():
    return {"status": "ok", "device": DEVICE}


@app.get("/health")
def health():
    return {
        "model_exists": MODEL_PATH.exists(),
        "model_path": str(MODEL_PATH)
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    start = time.time()
    ext = file.filename.split(".")[-1].lower()
    data = await file.read()

    try:

        if ext in ["jpg", "jpeg", "png"]:
            img = process_image(data)
            result = predict_image(img)

        else:
            raise HTTPException(400, "Only images supported for now")

        result["processing_time"] = round(time.time() - start, 3)
        return JSONResponse(result)

    except Exception as e:
        raise HTTPException(500, str(e))


# ==================== STARTUP ====================
@app.on_event("startup")
def startup():
    print("\n🚀 DeepGuard starting...")

    try:
        ModelLoader.get_model()
        print("Model loaded OK")
    except Exception as e:
        print("Model load error:", e)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)