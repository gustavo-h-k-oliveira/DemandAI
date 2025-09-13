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

# Endpoints
@app.get("")
def read_root():
    return {"message": "Hello World"}

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

# ====== Auth endpoints (mock simples) ======
@app.post("/auth/login")
def login(email: str, password: str):
    # Mock: sempre retorna sucesso
    return {"access_token": "mock-token-123", "token_type": "bearer"}

@app.post("/auth/register") 
def register(email: str, password: str):
    # Mock: sempre "cria" usuário
    return {"id": 1, "email": email, "message": "Usuário criado"}

@app.get("/auth/me")
def me():
    # Mock: usuário fixo
    return {"id": 1, "email": "demo@example.com", "name": "Demo User"}

# ====== Dashboard ======
@app.get("/api/dashboard/summary")
def dashboard_summary():
    return {
        "total_predictions": 0,
        "last_prediction_at": None,
        "user_email": "demo@example.com"
    }

# ====== Histórico ======
@app.get("/api/predictions")
def list_predictions():
    return {"items": [], "total": 0}

@app.get("/api/predictions/{prediction_id}")
def get_prediction(prediction_id: int):
    return {"id": prediction_id, "prediction": 0.0, "features": []}