# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
from fastapi.middleware.cors import CORSMiddleware
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_ID = os.environ.get("MODEL_ID", "Karthik-K-lab/fake-news-model")
HUGGINGFACE_TOKEN = os.environ.get("HUGGINGFACE_TOKEN", None)  # optional if model private
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load model + tokenizer at startup
logger.info(f"Loading model {MODEL_ID} to device {DEVICE} ...")
if HUGGINGFACE_TOKEN:
    from huggingface_hub import hf_hub_download, login as hf_login
    # optionally login so AutoModel can fetch private model
    # hf_login(HUGGINGFACE_TOKEN)  # uncomment if using huggingface_hub login flow

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
model.to(DEVICE)
model.eval()
logger.info("Model loaded.")

# FastAPI app
app = FastAPI(title="Fake News Detection API")

# Allow CORS from frontend (set to your Vercel app domain in production)
origins = os.environ.get("CORS_ORIGINS", "*")  # use comma-separated origins in prod
origins_list = [o.strip() for o in origins.split(",")] if origins != "*" else ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextIn(BaseModel):
    text: str

@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE, "model": MODEL_ID}

@app.post("/predict")
def predict(payload: TextIn):
    text = payload.text
    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="Empty text")
    # Tokenize
    inputs = tokenizer(text, truncation=True, padding=True, return_tensors="pt").to(DEVICE)
    # Inference
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0].tolist()
    # id2label mapping (model.config may contain it)
    if hasattr(model.config, "id2label") and model.config.id2label is not None:
        id2label = model.config.id2label
    else:
        id2label = {0: "REAL", 1: "FAKE"}
    result = [{"label": id2label.get(i, str(i)), "score": probs[i]} for i in range(len(probs))]
    result = sorted(result, key=lambda x: x["score"], reverse=True)
    return {"predictions": result}
