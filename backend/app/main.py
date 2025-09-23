import os
import numpy as np
import joblib
import pandas as pd
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

load_dotenv()

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

# Modelo e histórico
MODEL_DIR = os.getenv("MODEL_DIR", "ml-service/models")
MODEL_PATH = os.path.join(MODEL_DIR, "bombom_lasso_model.pkl")
HIST_DATA = os.getenv(
    "HIST_DATA",
    "/workspaces/DemandAI/ml-service/dataset_with_features.csv"  # caminho de desenvolvimento
)
DEFAULT_PRODUCT = os.getenv("PRODUCT_NAME", "BOMBOM MORANGUETE 13G 160UN")

_model = None

def _std_cols(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "Produto": "product",
        "Ano": "year",
        "Mês": "month",
        "M?s": "month",
        "Mes": "month",
        "Campanha": "campaign",
        "Sazonalidade": "seasonality",
        "QUANTIDADE": "quantity",
        "Quantidade": "quantity",
        "PESOL": "PESOL",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    return df

def _load_history(hist_path: str, product_name: str) -> pd.DataFrame:
    if not os.path.exists(hist_path):
        raise FileNotFoundError(f"Histórico não encontrado: {hist_path}")
    df = pd.read_csv(hist_path, encoding="latin1")
    df = _std_cols(df)
    base_cols = ["product", "year", "month", "campaign", "seasonality", "quantity"]
    missing = [c for c in base_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Histórico inválido, faltam colunas: {missing}")
    dfx = df[df["product"] == product_name].copy()
    if dfx.empty:
        raise ValueError(f"Nenhum histórico para produto: {product_name}")
    dfx = dfx.sort_values(["year", "month"]).reset_index(drop=True)
    return dfx[base_cols]  # começamos do mínimo necessário

def _compute_runtime_features(
    history_df: pd.DataFrame,
    year: int, month: int, campaign: int, seasonality: int,
    feature_columns: List[str],
) -> pd.DataFrame:
    # Usa apenas o histórico deste produto
    df = history_df.copy()
    # Se já existe a data solicitada, reutiliza a linha (e atualiza campanha/seasonality)
    mask = (df["year"] == year) & (df["month"] == month)
    if mask.any():
        row = df[mask].tail(1).copy()
        row.loc[:, "campaign"] = campaign
        row.loc[:, "seasonality"] = seasonality
        df2 = pd.concat([df.iloc[:-1], row], ignore_index=True) if mask.index[-1] == len(df) - 1 else df.copy()
    else:
        # Anexa nova linha com quantity desconhecida (NaN)
        row = {
            "product": df["product"].iloc[0],
            "year": year,
            "month": month,
            "campaign": campaign,
            "seasonality": seasonality,
            "quantity": np.nan,
        }
        df2 = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

    # Ordena e constrói features necessárias (sem vazamento; quantity atual é NaN)
    df2 = df2.sort_values(["year", "month"]).reset_index(drop=True)
    # Tempo e cíclicas
    df2["quarter"] = pd.to_datetime(df2[["year", "month"]].assign(day=1), errors="coerce").dt.quarter
    df2["month_sin"] = np.sin(2 * np.pi * df2["month"] / 12.0)
    df2["month_cos"] = np.cos(2 * np.pi * df2["month"] / 12.0)
    df2["quarter_sin"] = np.sin(2 * np.pi * df2["quarter"] / 4.0)
    df2["quarter_cos"] = np.cos(2 * np.pi * df2["quarter"] / 4.0)

    df2["trend"] = np.arange(len(df2))
    df2["trend_squared"] = df2["trend"] ** 2
    df2["trend_cubed"] = df2["trend"] ** 3

    # Lags
    for lag in [1, 2, 3, 4, 6, 12]:
        col = f"quantity_lag{lag}"
        if col in feature_columns:
            df2[col] = df2["quantity"].shift(lag)

    # Rolling
    for w in [3, 6, 9, 12]:
        if f"quantity_ma{w}" in feature_columns:
            df2[f"quantity_ma{w}"] = df2["quantity"].rolling(window=w, min_periods=1).mean()
        if f"quantity_std{w}" in feature_columns:
            df2[f"quantity_std{w}"] = df2["quantity"].rolling(window=w, min_periods=1).std()
        if f"quantity_min{w}" in feature_columns:
            df2[f"quantity_min{w}"] = df2["quantity"].rolling(window=w, min_periods=1).min()
        if f"quantity_max{w}" in feature_columns:
            df2[f"quantity_max{w}"] = df2["quantity"].rolling(window=w, min_periods=1).max()

    # EWM
    for span in [3, 6, 12]:
        col = f"quantity_ewm{span}"
        if col in feature_columns:
            df2[col] = df2["quantity"].ewm(span=span, adjust=False, min_periods=1).mean()

    # Crescimento/diferenças (dependem do valor atual; ficarão NaN para a última linha)
    if "quantity_pct_change" in feature_columns:
        df2["quantity_pct_change"] = df2["quantity"].pct_change()
    if "pct_change_smooth" in feature_columns:
        df2["pct_change_smooth"] = df2["quantity"].pct_change().rolling(3, min_periods=1).mean()
    if "quantity_diff" in feature_columns:
        df2["quantity_diff"] = df2["quantity"].diff()
    if "quantity_diff2" in feature_columns:
        df2["quantity_diff2"] = df2["quantity"].diff(2)

    # Flags sazonais
    df2["is_high_season"] = ((df2["month"] >= 7) & (df2["month"] <= 10)).astype(int)
    df2["is_low_season"] = ((df2["month"] >= 1) & (df2["month"] <= 3)).astype(int)
    df2["is_holiday_season"] = ((df2["month"] == 12) | (df2["month"] == 1)).astype(int)
    df2["is_summer"] = df2["month"].isin([12, 1, 2]).astype(int)
    df2["is_winter"] = df2["month"].isin([6, 7, 8]).astype(int)

    # Interações
    if "campaign_season_interaction" in feature_columns or "campaign_season" in feature_columns:
        df2["campaign_season_interaction"] = df2["campaign"] * df2["seasonality"]
        df2["campaign_season"] = df2["campaign"] * df2["seasonality"]
    if "campaign_month" in feature_columns:
        df2["campaign_month"] = df2["campaign"] * df2["month"]
    if "seasonality_month" in feature_columns:
        df2["seasonality_month"] = df2["seasonality"] * df2["month"]
    if "campaign_high_season" in feature_columns:
        df2["campaign_high_season"] = df2["campaign"] * df2["is_high_season"]
    if "seasonality_summer" in feature_columns:
        df2["seasonality_summer"] = df2["seasonality"] * df2["is_summer"]

    # Volatilidade/estabilidade
    if "quantity_volatility3" in feature_columns:
        df2["quantity_volatility3"] = df2["quantity_std3"] / df2["quantity_ma3"].replace(0, np.nan)
    if "quantity_volatility6" in feature_columns:
        df2["quantity_volatility6"] = df2["quantity_std6"] / df2["quantity_ma6"].replace(0, np.nan)
    if "quantity_stability_3" in feature_columns:
        df2["quantity_stability_3"] = 1.0 / (1.0 + df2["quantity_std3"].fillna(0))
    if "quantity_stability_6" in feature_columns:
        df2["quantity_stability_6"] = 1.0 / (1.0 + df2["quantity_std6"].fillna(0))

    # YoY
    if "quantity_yoy" in feature_columns or "yoy_growth" in feature_columns:
        df2["quantity_yoy"] = df2["quantity"].shift(12)
        if "yoy_growth" in feature_columns:
            df2["yoy_growth"] = (df2["quantity"] - df2["quantity_yoy"]) / df2["quantity_yoy"].replace(0, np.nan)

    # Dummies
    for m in range(1, 13):
        col = f"month_{m}"
        if col in feature_columns:
            df2[col] = (df2["month"] == m).astype(int)
    for q in range(1, 5):
        col = f"quarter_{q}"
        if col in feature_columns:
            df2[col] = (df2["quarter"] == q).astype(int)

    # Seleciona a última linha (solicitada) e ordena colunas conforme modelo
    last = df2.iloc[[-1]].copy()
    X = last.reindex(columns=feature_columns)

    # Tratamento numérico: NaN/inf -> 0 (sem vazamento; só usamos histórico)
    num_cols = X.select_dtypes(include=["number"]).columns
    X[num_cols] = X[num_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    return X

@app.on_event("startup")
def load_model_if_present():
    global _model
    try:
        if os.path.exists(MODEL_PATH):
            _model = joblib.load(MODEL_PATH)
            print(f"[backend] Modelo carregado: {MODEL_PATH}")
        else:
            print(f"[backend] Nenhum modelo encontrado em {MODEL_PATH}")
    except Exception as e:
        print(f"[backend] Falha ao carregar modelo: {e}")

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
    # Apenas 4 features primárias
    features: List[float]  # [year, month, campaign, seasonality]

class PredictResponse(BaseModel):
    prediction: float

@app.post("/api/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if _model is None:
        raise HTTPException(status_code=503, detail="Modelo não carregado")

    # Validação mínima dos 4 campos
    if not isinstance(req.features, list) or len(req.features) != 4:
        raise HTTPException(status_code=400, detail="Envie exatamente 4 valores: [ano, mês, campanha, sazonalidade].")

    try:
        year, month, campaign, seasonality = map(int, req.features)

        # Suporta modelo salvo como dict {model, scaler, feature_columns, product_name}
        if not isinstance(_model, dict) or "model" not in _model:
            raise HTTPException(status_code=500, detail="Modelo salvo em formato incompatível.")

        product_name = _model.get("product_name", DEFAULT_PRODUCT)
        feature_columns = _model.get("feature_columns", [])
        scaler = _model.get("scaler")
        model = _model["model"]

        # Carrega histórico do produto
        hist = _load_history(HIST_DATA, product_name)

        # Constrói features compostas para a data solicitada
        X = _compute_runtime_features(
            hist, year, month, campaign, seasonality, feature_columns
        )

        # Escala e prediz
        X_scaled = scaler.transform(X) if scaler is not None else X.values
        y = model.predict(X_scaled)
        return PredictResponse(prediction=float(y[0]))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erro ao predizer: {e}")

# ====== Auth endpoints (mock) ======
@app.post("/auth/login")
def login(email: str, password: str):
    return {"access_token": "mock-token-123", "token_type": "bearer"}

@app.post("/auth/register") 
def register(email: str, password: str):
    return {"id": 1, "email": email, "message": "Usuário criado"}

@app.get("/auth/me")
def me():
    return {"id": 1, "email": "demo@example.com", "name": "Demo User"}

# ====== Dashboard ======
@app.get("/api/dashboard/summary")
def dashboard_summary():
    return {"total_predictions": 0, "last_prediction_at": None, "user_email": "demo@example.com"}

# ====== Histórico ======
@app.get("/api/predictions")
def list_predictions():
    return {"items": [], "total": 0}

@app.get("/api/predictions/{prediction_id}")
def get_prediction(prediction_id: int):
    return {"id": prediction_id, "prediction": 0.0, "features": []}