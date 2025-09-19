import pandas as pd

# Função utilitária para calcular features compostas para o próximo mês
def compute_features_from_history(df_hist: pd.DataFrame, model_data) -> pd.DataFrame:
    df = df_hist.sort_values(['year', 'month']).reset_index(drop=True).copy()
    target = df.iloc[-1].copy()

    q = df['quantity'].iloc[:-1] if len(df) > 1 else pd.Series([0.0])

    # Base temporal
    target['month_sin'] = np.sin(2 * np.pi * target['month'] / 12)
    target['month_cos'] = np.cos(2 * np.pi * target['month'] / 12)
    target['quarter'] = pd.to_datetime({'year':[int(target['year'])], 'month':[int(target['month'])], 'day':[1]}).dt.quarter[0]
    target['quarter_sin'] = np.sin(2 * np.pi * target['quarter'] / 4)
    target['quarter_cos'] = np.cos(2 * np.pi * target['quarter'] / 4)

    # Lags necessários (cobrindo ambos modelos)
    for lag in [1, 2, 3, 4, 6, 12]:
        col = f'quantity_lag{lag}'
        target[col] = float(df.iloc[-(lag+1)]['quantity']) if len(df) > lag else 0.0

    # Rolling stats para Lasso (min_periods=1)
    for window in [3, 6, 9, 12]:
        r = q.rolling(window=window, min_periods=1)
        target[f'quantity_ma{window}'] = float(r.mean().iloc[-1]) if len(q) >= 1 else 0.0
        target[f'quantity_std{window}'] = float(r.std().iloc[-1]) if len(q) >= 1 else 0.0
        target[f'quantity_min{window}'] = float(r.min().iloc[-1]) if len(q) >= 1 else 0.0
        target[f'quantity_max{window}'] = float(r.max().iloc[-1]) if len(q) >= 1 else 0.0

    # EWMs e médias móveis para Ridge (min_periods ~ window//2)
    for window in [3, 6, 12]:
        r_cons = q.rolling(window=window, min_periods=max(1, window//2))
        target[f'quantity_ma{window}'] = float(r_cons.mean().iloc[-1]) if len(q) >= 1 else target.get(f'quantity_ma{window}', 0.0)
        target[f'quantity_ewm{window}'] = float(q.ewm(span=window).mean().iloc[-1]) if len(q) >= 1 else 0.0

    # Tendência
    # Lasso usa índice absoluto; Ridge usa normalizado
    series_len = max(1, len(df) - 1)
    target['trend'] = series_len - 1
    target['trend_squared'] = float(target['trend']) ** 2
    target['trend_cubed'] = float(target['trend']) ** 3
    # Variante normalizada (Ridge): usaremos ao alimentar colunas específicas
    trend_norm = (series_len - 1) / series_len

    # Crescimento
    pct_change = q.pct_change().fillna(0.0)
    target['quantity_pct_change'] = float(pct_change.iloc[-1]) if len(q) > 0 else 0.0
    target['quantity_pct_change_smooth'] = float(pct_change.rolling(window=3).mean().fillna(0).iloc[-1]) if len(q) > 0 else 0.0
    target['quantity_diff'] = float(q.diff().iloc[-1]) if len(q) > 1 else 0.0
    target['quantity_diff2'] = float(q.diff().diff().iloc[-1]) if len(q) > 2 else 0.0

    # Sazonalidade
    target['is_high_season'] = float(7 <= int(target['month']) <= 10)
    target['is_low_season'] = float(1 <= int(target['month']) <= 3)
    target['is_holiday_season'] = float(int(target['month']) == 12 or int(target['month']) == 1)
    target['is_summer'] = float(6 <= int(target['month']) <= 8)
    target['is_winter'] = float(int(target['month']) == 12 or int(target['month']) <= 2)

    # Dummies de mês e quarter (Lasso)
    for m in range(1, 13):
        target[f'month_{m}'] = int(int(target['month']) == m)
    for qtr in range(1, 5):
        target[f'quarter_{qtr}'] = int(int(target['quarter']) == qtr)

    # Encodings de mês (Ridge, conforme treino)
    for i in range(1, 13):
        val = 1 if int(target['month']) == i else 0
        target[f'month_{i}_sin'] = float(np.sin(2 * np.pi * val))
        target[f'month_{i}_cos'] = float(np.cos(2 * np.pi * val))

    # Interações (ambos, com nomes específicos)
    target['campaign_season_interaction'] = float(target['campaign']) * float(target['seasonality'])
    target['campaign_month'] = float(target['campaign']) * float(target['month'])
    target['seasonality_month'] = float(target['seasonality']) * float(target['month'])
    target['campaign_high_season'] = float(target['campaign']) * float(target['is_high_season'])
    target['seasonality_summer'] = float(target['seasonality']) * float(target['is_summer'])
    target['campaign_season'] = float(target['campaign']) * float(target['seasonality'])

    # Volatilidade (Lasso)
    ma3 = target.get('quantity_ma3', 0.0) or 0.0
    std3 = target.get('quantity_std3', 0.0) or 0.0
    target['quantity_volatility3'] = float(std3 / ma3) if ma3 not in (0, 0.0) else 0.0
    ma6 = target.get('quantity_ma6', 0.0) or 0.0
    std6 = target.get('quantity_std6', 0.0) or 0.0
    target['quantity_volatility6'] = float(std6 / ma6) if ma6 not in (0, 0.0) else 0.0

    # Estabilidade (Ridge)
    target['quantity_stability_3'] = float(q.rolling(window=3).std().fillna(0).iloc[-1]) if len(q) > 0 else 0.0
    target['quantity_stability_6'] = float(q.rolling(window=6).std().fillna(0).iloc[-1]) if len(q) > 0 else 0.0

    # YoY (ambos, mas Ridge não usa acceleration)
    if len(df) > 12:
        yoy = float(df.iloc[-13]['quantity'])
        target['quantity_yoy'] = yoy
        target['yoy_growth'] = float(((df.iloc[-2]['quantity'] - yoy) / yoy)) if yoy not in (0, 0.0) else 0.0
        target['yoy_acceleration'] = float('nan')  # não usada no Ridge
    else:
        target['quantity_yoy'] = 0.0
        target['yoy_growth'] = 0.0
        target['yoy_acceleration'] = 0.0

    # Ajustes finais: se colunas do Ridge pedirem trend normalizado, substituímos
    if 'trend' in model_data['feature_columns']:
        # Se o Ridge estiver usando trend normalizado, manteremos trend_norm em uma chave auxiliar e trocaremos ao projetar
        pass

    # Projeção final: somente as colunas pedidas, na ordem do treino
    row = {}
    is_ridge = 'quantity_ewm3' in model_data['feature_columns']
    for col in model_data['feature_columns']:
        if is_ridge and col == 'trend':
            row[col] = trend_norm
            continue
        if is_ridge and col == 'trend_squared':
            row[col] = float(trend_norm ** 2)
            continue
        val = target.get(col, 0.0)
        # Substituir NaN/inf por 0 de forma conservadora
        if isinstance(val, (int, float)):
            if pd.isna(val) or val in (np.inf, -np.inf):
                val = 0.0
        row[col] = val

    return pd.DataFrame([row])

from fastapi import FastAPI, HTTPException, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
import joblib
import numpy as np
import pandas as pd
from typing import List, Optional
from typing import List, Optional
import os

# Carregar modelos treinados
LASSO_MODEL_PATH = 'models/topbel_lasso_model.pkl'
RIDGE_MODEL_PATH = 'models/topbel_ridge_conservative_model.pkl'
BOMBOM_MODEL_PATH = 'models/bombom_lasso_model.pkl'


app = FastAPI(title="DemandAI - Predição de Demanda")
templates = Jinja2Templates(directory="templates")
@app.get("/form", response_class=HTMLResponse)
async def form_get(request: Request):
    return templates.TemplateResponse("form.html", {"request": request, "result": None})

# Rota para processar o formulário e mostrar a predição
@app.post("/form", response_class=HTMLResponse)
async def form_post(request: Request,
                   model_type: str = Form(...),
                   year: int = Form(...),
                   month: int = Form(...),
                   campaign: int = Form(...),
                   seasonality: int = Form(...)):
    # Montar input para API interna
    input_data = PredictionInput(
        model_type=model_type,
        year=year,
        month=month,
        campaign=campaign,
        seasonality=seasonality
    )
    try:
        result = predict(input_data)
        prediction = result["prediction"]
    except Exception as e:
        prediction = f"Erro: {str(e)}"
    return templates.TemplateResponse("form.html", {"request": request, "result": prediction})


# Novo modelo para histórico
class HistoryRecord(BaseModel):
    year: int
    month: int
    campaign: int
    seasonality: int
    quantity: float

class PredictionInput(BaseModel):
    model_type: str  # 'topbel_lasso', 'topbel_ridge', 'bombom_lasso'
    # Parâmetros do mês alvo (opcional se history for fornecido)
    year: Optional[int] = None
    month: Optional[int] = None
    campaign: Optional[int] = None
    seasonality: Optional[int] = None
    # Histórico de registros, ordenado do mais antigo para o mais recente (opcional)
    history: Optional[List[HistoryRecord]] = Field(None, description="Histórico do produto; se ausente, usa dataset interno")

def _product_name_from_model(model_type: str) -> str:
    if model_type in ('topbel_lasso', 'topbel_ridge'):
        return "TOPBEL LEITE CONDENSADO 50UN"
    if model_type == 'bombom_lasso':
        return "BOMBOM MORANGUETE 13G 160UN"
    raise HTTPException(status_code=400, detail="Modelo não suportado.")

def build_history_from_dataset(model_type: str, year: int, month: int, campaign: int, seasonality: int) -> pd.DataFrame:
    try:
        df = pd.read_csv('dataset.csv', sep=',', encoding='latin1')
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao ler dataset.csv: {e}")
    # Remover NaN e coluna PESOL, padronizar colunas
    df = df.dropna()
    if 'PESOL' in df.columns:
        df = df.drop('PESOL', axis=1)
    # Renomear colunas por posição conforme usado nos treinos
    df.columns = ['product', 'year', 'month', 'campaign', 'seasonality', 'quantity']
    product_name = _product_name_from_model(model_type)
    df = df[df['product'] == product_name].copy()
    if df.empty:
        raise HTTPException(status_code=404, detail=f"Sem histórico no dataset para {product_name}")
    # Ordenar e filtrar até mês anterior ao alvo
    df = df.sort_values(['year', 'month']).reset_index(drop=True)
    df_hist = df[(df['year'] < year) | ((df['year'] == year) & (df['month'] < month))].copy()
    if df_hist.empty:
        raise HTTPException(status_code=400, detail="Histórico insuficiente antes do mês alvo no dataset.")
    # Idealmente manter pelo menos 12 meses
    if len(df_hist) > 24:
        df_hist = df_hist.iloc[-24:].copy()
    # Adicionar linha do alvo com quantity=0
    target_row = {
        'product': product_name,
        'year': year,
        'month': month,
        'campaign': campaign,
        'seasonality': seasonality,
        'quantity': 0.0
    }
    df_hist = pd.concat([df_hist, pd.DataFrame([target_row])], ignore_index=True)
    return df_hist[['year','month','campaign','seasonality','quantity']]

@app.on_event("startup")
def load_models():
    global lasso_model, ridge_model, bombom_model
    lasso_model = joblib.load(LASSO_MODEL_PATH) if os.path.exists(LASSO_MODEL_PATH) else None
    ridge_model = joblib.load(RIDGE_MODEL_PATH) if os.path.exists(RIDGE_MODEL_PATH) else None
    bombom_model = joblib.load(BOMBOM_MODEL_PATH) if os.path.exists(BOMBOM_MODEL_PATH) else None


# Novo endpoint que aceita histórico
@app.post("/predict")
def predict(input: PredictionInput):
    # Selecionar modelo
    if input.model_type == 'topbel_lasso':
        model_data = lasso_model
    elif input.model_type == 'topbel_ridge':
        model_data = ridge_model
    elif input.model_type == 'bombom_lasso':
        model_data = bombom_model
    else:
        raise HTTPException(status_code=400, detail="Modelo não suportado.")
    if model_data is None:
        raise HTTPException(status_code=500, detail="Modelo não carregado.")

    # Obter histórico: usar o fornecido ou montar a partir do dataset
    if input.history is not None and len(input.history) > 0:
        df_hist = pd.DataFrame([r.dict() for r in input.history])
    else:
        # Validar parâmetros do mês alvo
        missing = [name for name, val in {
            'year': input.year,
            'month': input.month,
            'campaign': input.campaign,
            'seasonality': input.seasonality
        }.items() if val is None]
        if missing:
            raise HTTPException(status_code=400, detail=f"Parâmetros ausentes para montar histórico: {', '.join(missing)}")
        df_hist = build_history_from_dataset(
            input.model_type,
            int(input.year),
            int(input.month),
            int(input.campaign),
            int(input.seasonality)
        )
    if df_hist is None or len(df_hist) < 2:
        raise HTTPException(status_code=400, detail="Histórico insuficiente para calcular features compostas.")

    # Calcular features compostas para o mês alvo
    X = compute_features_from_history(df_hist, model_data)
    X_scaled = model_data['scaler'].transform(X)
    pred = model_data['model'].predict(X_scaled)
    pred = float(np.maximum(pred, 0)[0])
    return {"prediction": pred, "model": input.model_type}

# Endpoint de debug: inspeciona features e contribuições do modelo linear
@app.post("/_debug/inspect")
def debug_inspect(input: PredictionInput):
    # Reutiliza o fluxo de seleção de modelo e histórico
    if input.model_type == 'topbel_lasso':
        model_data = lasso_model
    elif input.model_type == 'topbel_ridge':
        model_data = ridge_model
    elif input.model_type == 'bombom_lasso':
        model_data = bombom_model
    else:
        raise HTTPException(status_code=400, detail="Modelo não suportado.")
    if model_data is None:
        raise HTTPException(status_code=500, detail="Modelo não carregado.")

    if input.history is not None and len(input.history) > 0:
        df_hist = pd.DataFrame([r.dict() for r in input.history])
    else:
        missing = [name for name, val in {
            'year': input.year,
            'month': input.month,
            'campaign': input.campaign,
            'seasonality': input.seasonality
        }.items() if val is None]
        if missing:
            raise HTTPException(status_code=400, detail=f"Parâmetros ausentes para montar histórico: {', '.join(missing)}")
        df_hist = build_history_from_dataset(
            input.model_type,
            int(input.year),
            int(input.month),
            int(input.campaign),
            int(input.seasonality)
        )
    X = compute_features_from_history(df_hist, model_data)

    # Alinhamento de colunas
    expected = list(model_data['feature_columns'])
    got = list(X.columns)
    missing_cols = [c for c in expected if c not in got]
    extra_cols = [c for c in got if c not in expected]

    # Escala e contribuições lineares
    X_scaled = model_data['scaler'].transform(X)
    coef = getattr(model_data['model'], 'coef_', None)
    intercept = getattr(model_data['model'], 'intercept_', 0.0)
    contribs = None
    pred_raw = None
    if coef is not None:
        contribs = (coef * X_scaled[0]).tolist()
        pred_raw = float(np.dot(coef, X_scaled[0]) + intercept)

    feature_values = X.iloc[0].to_dict()
    scaled_values = {expected[i]: float(X_scaled[0][i]) for i in range(len(expected))}

    # Top contribuições
    top_pos = []
    top_neg = []
    if contribs is not None:
        pairs = list(zip(expected, contribs))
        top_pos = sorted([p for p in pairs if p[1] > 0], key=lambda x: x[1], reverse=True)[:10]
        top_neg = sorted([p for p in pairs if p[1] < 0], key=lambda x: x[1])[:10]

    return {
        "model": input.model_type,
        "target": df_hist.iloc[-1][['year','month','campaign','seasonality']].to_dict(),
        "missing_columns": missing_cols,
        "extra_columns": extra_cols,
        "feature_values": feature_values,
        "scaled_values_preview": {k: scaled_values[k] for k in list(scaled_values)[:20]},
        "prediction_linear_raw": pred_raw,
        "prediction_after_clip": None if pred_raw is None else float(max(pred_raw, 0.0)),
        "top_positive_contribs": top_pos,
        "top_negative_contribs": top_neg
    }

@app.get("/")
def root():
    return {"message": "DemandAI FastAPI está rodando! Use o endpoint /predict."}
