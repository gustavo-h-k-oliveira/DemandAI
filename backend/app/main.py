import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import joblib

app = FastAPI(title="Backend FastAPI")

# CORS
frontend_origin = os.getenv("FRONTEND_ORIGIN", "http://localhost:5173")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[frontend_origin],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modelo (opcional)
MODEL_DIR = os.getenv("MODEL_DIR", "/models")
MODEL_PATH = os.path.join(MODEL_DIR, "model.joblib")
_model = None

@app.on_event("startup")
def load_model_if_present():
    global _model
    try:
        if os.path.exists(MODEL_PATH):
            _model = joblib.load(MODEL_PATH)
            print(f"Modelo carregado de {MODEL_PATH}")
        else:
            print(f"Nenhum modelo encontrado em {MODEL_PATH}. Endpoint /api/predict responderá 503.")
    except Exception as e:
        print(f"Falha ao carregar modelo: {e}")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/")
def root():
    return {"message": "API online"}

class PredictRequest(BaseModel):
    features: List[float]

class PredictResponse(BaseModel):
    prediction: float

@app.post("/api/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if _model is None:
        raise HTTPException(status_code=503, detail="Modelo não carregado")
    try:
        import numpy as np
        x = np.array([req.features])
        y = _model.predict(x)
        return PredictResponse(prediction=float(y[0]))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erro ao predizer: {e}")