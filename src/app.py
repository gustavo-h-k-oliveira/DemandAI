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
    target['trend_norm'] = trend_norm

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
        target['yoy_diff'] = float(df.iloc[-2]['quantity'] - yoy)
        target['yoy_growth'] = float(((df.iloc[-2]['quantity'] - yoy) / yoy)) if yoy not in (0, 0.0) else 0.0
        target['yoy_acceleration'] = float('nan')  # não usada no Ridge
    else:
        target['quantity_yoy'] = 0.0
        target['yoy_diff'] = 0.0
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
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import joblib
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple
import os
import csv
import json
import logging
from collections import deque
from datetime import datetime
import unicodedata
from fastapi.responses import JSONResponse

# Caminhos absolutos baseados no local deste arquivo (robusto para Docker e execução local)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.abspath(os.path.join(BASE_DIR, "../models"))
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, "../data"))
DATASET_PATH = os.path.join(DATA_DIR, "dataset.csv")
PREDICTIONS_LOG_PATH = os.path.join(DATA_DIR, "predictions_log.csv")
FLAG_PREDICTIONS_PATH = os.path.join(DATA_DIR, "predicted_flags_h1.csv")


def _normalize_column_name(value: str) -> str:
    """Normalize dataset column headers (strip, lowercase, remove accents/spaces)."""
    if value is None:
        return ""
    normalized = unicodedata.normalize("NFKD", str(value))
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    return normalized.strip().lower()

DEFAULT_RF_FEATURES = [
    'year', 'month', 'campaign', 'seasonality', 'quarter', 'month_sin', 'month_cos',
    'quarter_sin', 'quarter_cos', 'quantity_lag1', 'quantity_lag2', 'quantity_lag3',
    'quantity_lag4', 'quantity_lag6', 'quantity_lag12', 'quantity_ma3', 'quantity_std3',
    'quantity_min3', 'quantity_max3', 'quantity_ma6', 'quantity_std6', 'quantity_min6',
    'quantity_max6', 'quantity_ma9', 'quantity_std9', 'quantity_min9', 'quantity_max9',
    'quantity_ma12', 'quantity_std12', 'quantity_min12', 'quantity_max12', 'quantity_ewm3',
    'quantity_ewm6', 'quantity_ewm12', 'quantity_volatility3', 'quantity_volatility6',
    'quantity_stability_3', 'quantity_stability_6', 'trend', 'trend_norm', 'trend_squared', 'trend_cubed',
    'quantity_pct_change', 'quantity_pct_change_smooth', 'quantity_diff', 'quantity_diff2',
    'is_high_season', 'is_low_season', 'is_holiday_season', 'is_summer', 'is_winter',
    'quantity_yoy', 'yoy_diff', 'yoy_growth', 'campaign_season_interaction', 'campaign_season',
    'campaign_month', 'seasonality_month', 'campaign_high_season', 'seasonality_summer',
    'month_1', 'month_2', 'month_3', 'month_4', 'month_5', 'month_6', 'month_7', 'month_8',
    'month_9', 'month_10', 'month_11', 'month_12', 'quarter_1', 'quarter_2', 'quarter_3',
    'quarter_4'
]

RF_MODEL_SPECS: Dict[str, Dict[str, Any]] = {
    "rf_bombom": {
        "product": "BOMBOM MORANGUETE 13G 160UN",
        "model_path": os.path.join(MODELS_DIR, "bombom_moranguete_rf_model.pkl"),
        "info_path": os.path.join(MODELS_DIR, "bombom_moranguete_rf_model_info.json"),
    },
    "rf_teta_bel": {
        "product": "TETA BEL TRADICIONAL 50UN",
        "model_path": os.path.join(MODELS_DIR, "teta_bel_rf_model.pkl"),
        "info_path": os.path.join(MODELS_DIR, "teta_bel_rf_model_info.json"),
    },
    "rf_topbel_leite": {
        "product": "TOPBEL LEITE CONDENSADO 50UN",
        "model_path": os.path.join(MODELS_DIR, "topbel_leite_condensado_rf_model.pkl"),
        "info_path": os.path.join(MODELS_DIR, "topbel_leite_condensado_rf_model_info.json"),
    },
    "rf_topbel_tradicional": {
        "product": "TOPBEL TRADICIONAL 50UN",
        "model_path": os.path.join(MODELS_DIR, "topbel_tradicional_rf_model.pkl"),
        "info_path": os.path.join(MODELS_DIR, "topbel_tradicional_rf_model_info.json"),
    },
}

flag_predictions_cache: Dict[Tuple[str, int, int], Dict[str, Any]] = {}

lasso_model = None
ridge_model = None
bombom_model = None

app = FastAPI(title="DemandAI")
# Monta diretório /static para servir CSS/JS/imagens
STATIC_DIR = os.path.abspath(os.path.join(BASE_DIR, "../static"))
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

TEMPLATES_DIR = os.path.abspath(os.path.join(BASE_DIR, "../templates"))
templates = Jinja2Templates(directory=TEMPLATES_DIR)
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
        # Garantir que o valor exibido na página seja inteiro
        raw_pred = result.get("prediction") if isinstance(result, dict) else result
        if isinstance(raw_pred, (int, float)):
            try:
                prediction = int(round(raw_pred))
            except Exception:
                prediction = raw_pred
        else:
            prediction = raw_pred
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
    model_type: str  # 'topbel_lasso', 'topbel_ridge', 'bombom_lasso', 'rf_bombom', 'rf_teta_bel', 'rf_topbel_leite', 'rf_topbel_tradicional'
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
    spec = RF_MODEL_SPECS.get(model_type)
    if spec:
        return spec['product']
    raise HTTPException(status_code=400, detail="Modelo não suportado.")


def _load_flag_predictions() -> None:
    global flag_predictions_cache
    if not os.path.exists(FLAG_PREDICTIONS_PATH):
        flag_predictions_cache = {}
        return
    try:
        df_flags = pd.read_csv(FLAG_PREDICTIONS_PATH)
        df_flags.columns = [str(col).strip() for col in df_flags.columns]
    except Exception as exc:
        logging.warning("Falha ao ler predicted_flags_h1.csv: %s", exc)
        flag_predictions_cache = {}
        return

    cache: Dict[Tuple[str, int, int], Dict[str, Any]] = {}
    required_cols = {
        'product', 'target_year', 'target_month',
        'campaign_prediction', 'seasonality_prediction',
        'campaign_probability', 'seasonality_probability'
    }
    missing_cols = required_cols.difference(df_flags.columns)
    if missing_cols:
        logging.warning("Arquivo de flags não possui colunas %s", missing_cols)
    for _, row in df_flags.iterrows():
        try:
            product_name = str(row.get('product') or row.get('slug') or '').strip()
            if not product_name:
                continue
            key = (product_name, int(row['target_year']), int(row['target_month']))
            cache[key] = {
                'campaign_prediction': int(row.get('campaign_prediction', 0)),
                'seasonality_prediction': int(row.get('seasonality_prediction', 0)),
                'campaign_probability': float(row.get('campaign_probability', 0.0)),
                'seasonality_probability': float(row.get('seasonality_probability', 0.0)),
            }
        except Exception:
            continue
    flag_predictions_cache = cache


def get_predicted_flags(product: str, target_year: int, target_month: int) -> Optional[Dict[str, Any]]:
    if not flag_predictions_cache:
        _load_flag_predictions()
    return flag_predictions_cache.get((product, target_year, target_month))


def _load_feature_columns(info_path: str) -> List[str]:
    if not info_path or not os.path.exists(info_path):
        return DEFAULT_RF_FEATURES
    try:
        with open(info_path, "r", encoding="utf-8") as fp:
            data = json.load(fp)
        features = data.get("features") or data.get("feature_columns")
        if isinstance(features, list) and features:
            return features
    except Exception as exc:
        logging.warning("Falha ao ler feature columns de %s: %s", info_path, exc)
    return DEFAULT_RF_FEATURES

def build_history_from_dataset(
    model_type: str,
    year: int,
    month: int,
    campaign: Optional[int],
    seasonality: Optional[int],
) -> Tuple[pd.DataFrame, int, int]:
    try:
        df = pd.read_csv(DATASET_PATH, sep=',', encoding='latin1')
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao ler dataset.csv: {e}")

    # Padronizar nomes de colunas independentemente de maiúsculas/acentos/espacos
    df.columns = [_normalize_column_name(col) for col in df.columns]
    rename_map = {
        'produto': 'product',
        'ano': 'year',
        'mes': 'month',
        'campanha': 'campaign',
        'sazonalidade': 'seasonality',
        'pesol': 'pesol',
        'quantidade': 'quantity',
    }
    df = df.rename(columns=rename_map)

    # Garantir que as colunas necessárias existem
    required_cols = {'product', 'year', 'month', 'campaign', 'seasonality', 'quantity'}
    missing = required_cols - set(df.columns)
    if missing:
        # Fallback: aplicar mapeamento por posição quando headers estiverem corrompidos
        ordered_cols = ['product', 'year', 'month', 'campaign', 'seasonality', 'pesol', 'quantity']
        if len(df.columns) >= len(ordered_cols):
            df = df.iloc[:, :len(ordered_cols)].copy()
            df.columns = ordered_cols
            missing = required_cols - set(df.columns)
        if missing:
            raise HTTPException(status_code=500, detail=f"Colunas ausentes no dataset: {sorted(missing)}")

    # Remover coluna opcional PESOL e linhas vazias
    if 'pesol' in df.columns:
        df = df.drop(columns=['pesol'])
    df = df.dropna(subset=list(required_cols))

    # Reordenar colunas conforme esperado e normalizar espaços no nome do produto
    df = df[['product', 'year', 'month', 'campaign', 'seasonality', 'quantity']]
    df['product'] = df['product'].astype(str).str.strip()

    numeric_cols = ['year', 'month', 'campaign', 'seasonality', 'quantity']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=numeric_cols)
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
    last_observed = df_hist.iloc[-1]
    inferred_campaign = int(campaign if campaign is not None else last_observed['campaign'])
    inferred_seasonality = int(seasonality if seasonality is not None else last_observed['seasonality'])

    # Adicionar linha do alvo com quantity=0
    target_row = {
        'product': product_name,
        'year': year,
        'month': month,
        'campaign': inferred_campaign,
        'seasonality': inferred_seasonality,
        'quantity': 0.0
    }
    df_hist = pd.concat([df_hist, pd.DataFrame([target_row])], ignore_index=True)
    return df_hist[['year','month','campaign','seasonality','quantity']], inferred_campaign, inferred_seasonality


def _append_prediction_log(entry: dict) -> None:
    """Append a prediction entry to CSV log (creates file with header if missing)."""
    os.makedirs(DATA_DIR, exist_ok=True)
    write_header = not os.path.exists(PREDICTIONS_LOG_PATH)
    fieldnames = [
        "timestamp",
        "model_type",
        "year",
        "month",
        "campaign",
        "seasonality",
        "prediction"
    ]
    try:
        with open(PREDICTIONS_LOG_PATH, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow({k: entry.get(k) for k in fieldnames})
    except Exception:
        # Logging falhou não deve impedir resposta principal
        pass


def _read_recent_predictions(limit: int = 10) -> List[dict]:
    """Return the last `limit` predictions from CSV log."""
    if not os.path.exists(PREDICTIONS_LOG_PATH):
        return []
    try:
        with open(PREDICTIONS_LOG_PATH, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            buffer = deque(maxlen=limit)
            for row in reader:
                # converter tipos básicos quando possível
                try:
                    row['year'] = int(row.get('year', 0))
                    row['month'] = int(row.get('month', 0))
                    row['campaign'] = int(row.get('campaign', 0))
                    row['seasonality'] = int(row.get('seasonality', 0))
                    row['prediction'] = int(row.get('prediction', 0))
                except Exception:
                    pass
                # map model_type code to human-readable product name when possible
                try:
                    row_product = _product_name_from_model(row.get('model_type', ''))
                except Exception:
                    row_product = row.get('model_type')
                row['product'] = row_product
                buffer.append(row)
            return list(buffer)
    except Exception:
        return []


@app.get("/predictions")
def recent_predictions(limit: int = 10):
    """Return recent user prediction logs."""
    preds = _read_recent_predictions(limit)
    return JSONResponse({"user_predictions": preds})

@app.on_event("startup")
def load_models():
    global lasso_model, ridge_model, bombom_model

    # Modelos Lasso/Ridge permanecem desativados; mantidos para compatibilidade
    # lasso_model = joblib.load(LASSO_MODEL_PATH) if os.path.exists(LASSO_MODEL_PATH) else None
    # ridge_model = joblib.load(RIDGE_MODEL_PATH) if os.path.exists(RIDGE_MODEL_PATH) else None
    # bombom_model = joblib.load(BOMBOM_MODEL_PATH) if os.path.exists(BOMBOM_MODEL_PATH) else None

    for model_key, spec in RF_MODEL_SPECS.items():
        model_path = spec.get('model_path')
        info_path = spec.get('info_path')

        if model_path and os.path.exists(model_path):
            try:
                spec['model'] = joblib.load(model_path)
            except Exception as exc:
                logging.error("Falha ao carregar modelo %s: %s", model_key, exc)
                spec['model'] = None
        else:
            logging.warning("Modelo %s não encontrado em %s", model_key, model_path)
            spec['model'] = None

        spec['feature_columns'] = _load_feature_columns(info_path)

    _load_flag_predictions()


# Função para preparar features para modelos Random Forest
def prepare_rf_features(
    df_hist: pd.DataFrame,
    target_year: int,
    target_month: int,
    target_campaign: int,
    target_seasonality: int,
    feature_columns: List[str],
) -> pd.DataFrame:
    """
    Prepara features para modelos Random Forest a partir do histórico.
    Os modelos RF esperam as mesmas 76 features que foram usadas no treinamento.
    """
    # Garantir que o histórico está ordenado
    df_hist = df_hist.sort_values(['year', 'month']).reset_index(drop=True)
    
    # Criar linha para o mês alvo
    target_row = {
        'year': target_year,
        'month': target_month,
        'campaign': target_campaign,
        'seasonality': target_seasonality,
        'quantity': 0.0  # Será preenchida pela predição
    }
    
    # Adicionar a linha alvo ao histórico
    df_complete = pd.concat([df_hist, pd.DataFrame([target_row])], ignore_index=True).reset_index(drop=True)
    
    mock_model_data = {'feature_columns': feature_columns or DEFAULT_RF_FEATURES}

    X = compute_features_from_history(df_complete, mock_model_data)
    missing_cols = [col for col in mock_model_data['feature_columns'] if col not in X.columns]
    for col in missing_cols:
        X[col] = 0.0
    return X[mock_model_data['feature_columns']]


# Helper: cap prediction based on historical maxima to avoid unrealistic extremes
def cap_prediction(pred_value: float, df_hist: pd.DataFrame) -> float:
    try:
        # Exclude the last row if it's the target row with quantity=0
        hist = df_hist.copy()
        if len(hist) > 1:
            hist_vals = hist.iloc[:-1]['quantity']
        else:
            hist_vals = hist['quantity']
        max_hist = float(hist_vals.max()) if not hist_vals.empty else 0.0
        # If no history, don't cap aggressively; keep original
        if max_hist <= 0:
            return float(pred_value)
        # Allow some growth but limit extreme predictions (e.g., 1.5x historical max)
        cap = max_hist * 1.5
        # Never return negative
        capped = float(min(max(pred_value, 0.0), cap))
        return capped
    except Exception:
        return float(pred_value)

# Novo endpoint que aceita histórico
@app.post("/predict")
def predict(input: PredictionInput):
    # Selecionar modelo
    model_data = None
    is_rf_model = False
    rf_feature_columns: Optional[List[str]] = None
    
    if input.model_type == 'topbel_lasso':
        model_data = lasso_model
    elif input.model_type == 'topbel_ridge':
        model_data = ridge_model
    elif input.model_type == 'bombom_lasso':
        model_data = bombom_model
    else:
        rf_spec = RF_MODEL_SPECS.get(input.model_type)
        if rf_spec:
            model_data = rf_spec.get('model')
            rf_feature_columns = rf_spec.get('feature_columns') or DEFAULT_RF_FEATURES
            is_rf_model = True
        else:
            raise HTTPException(status_code=400, detail="Modelo não suportado.")
    
    if model_data is None:
        raise HTTPException(status_code=500, detail="Modelo não carregado.")

    product_name = _product_name_from_model(input.model_type)
    applied_campaign: Optional[int] = input.campaign if input.campaign is not None else None
    applied_seasonality: Optional[int] = input.seasonality if input.seasonality is not None else None
    flag_source = "user"

    # Obter histórico: usar o fornecido ou montar a partir do dataset
    if input.history is not None and len(input.history) > 0:
        df_hist = pd.DataFrame([r.dict() for r in input.history])
        if not {'year','month','campaign','seasonality','quantity'}.issubset(df_hist.columns):
            raise HTTPException(status_code=400, detail="Histórico precisa conter year, month, campaign, seasonality e quantity")
        applied_campaign = int(df_hist.iloc[-1]['campaign'])
        applied_seasonality = int(df_hist.iloc[-1]['seasonality'])
        flag_source = "history"
    else:
        # Validar parâmetros do mês alvo
        missing = [name for name, val in {
            'year': input.year,
            'month': input.month,
            'campaign': input.campaign,
            'seasonality': input.seasonality
        }.items() if val is None]
        if missing:
            # campaign/seasonality podem ser preenchidos pelas flags preditas
            missing = [m for m in missing if m not in {'campaign', 'seasonality'}]
        if missing:
            raise HTTPException(status_code=400, detail=f"Parâmetros ausentes para montar histórico: {', '.join(missing)}")

        target_year = int(input.year)
        target_month = int(input.month)
        predicted_flags = get_predicted_flags(product_name, target_year, target_month)
        if predicted_flags:
            if applied_campaign is None:
                applied_campaign = predicted_flags['campaign_prediction']
                flag_source = "predicted"
            if applied_seasonality is None:
                applied_seasonality = predicted_flags['seasonality_prediction']
                flag_source = "predicted"

        df_hist, applied_campaign, applied_seasonality = build_history_from_dataset(
            input.model_type,
            int(input.year),
            int(input.month),
            applied_campaign,
            applied_seasonality,
        )
        if flag_source == "user" and (input.campaign is None or input.seasonality is None):
            flag_source = "history"
    if df_hist is None or len(df_hist) < 2:
        raise HTTPException(status_code=400, detail="Histórico insuficiente para calcular features compostas.")

    # Calcular features e fazer predição
    if is_rf_model:
        # Modelos Random Forest não precisam de scaling e usam estrutura diferente
        X = prepare_rf_features(
            df_hist[:-1] if len(df_hist) > 1 else df_hist,  # Histórico sem a linha alvo
            int(input.year) if input.year else df_hist.iloc[-1]['year'],
            int(input.month) if input.month else df_hist.iloc[-1]['month'],
            int(applied_campaign if applied_campaign is not None else df_hist.iloc[-1]['campaign']),
            int(applied_seasonality if applied_seasonality is not None else df_hist.iloc[-1]['seasonality']),
            rf_feature_columns or DEFAULT_RF_FEATURES,
        )
        pred = model_data.predict(X)
    else:
        # Modelos lineares (Lasso/Ridge) usam o fluxo original
        X = compute_features_from_history(df_hist, model_data)
        X_scaled = model_data['scaler'].transform(X)
        pred = model_data['model'].predict(X_scaled)
    
    pred = float(np.maximum(pred, 0)[0])
    pred_capped = cap_prediction(pred, df_hist)
    pred_int = int(round(pred_capped))

    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "model_type": input.model_type,
        "year": input.year if input.year is not None else int(df_hist.iloc[-1]['year']),
        "month": input.month if input.month is not None else int(df_hist.iloc[-1]['month']),
        "campaign": int(applied_campaign if applied_campaign is not None else df_hist.iloc[-1]['campaign']),
        "seasonality": int(applied_seasonality if applied_seasonality is not None else df_hist.iloc[-1]['seasonality']),
        "prediction": pred_int
    }
    _append_prediction_log(log_entry)
    user_predictions = _read_recent_predictions(limit=10)

    return {
        "prediction": pred_int,
        "model_used": input.model_type,
        "applied_flags": {
            "campaign": int(applied_campaign if applied_campaign is not None else df_hist.iloc[-1]['campaign']),
            "seasonality": int(applied_seasonality if applied_seasonality is not None else df_hist.iloc[-1]['seasonality'])
        },
        "flag_source": flag_source,
        "user_predictions": user_predictions
    }

# Endpoint de debug: inspeciona features e contribuições do modelo linear
@app.post("/_debug/inspect")
def debug_inspect(input: PredictionInput):
    # Reutiliza o fluxo de seleção de modelo e histórico
    model_data = None
    is_rf_model = False
    
    # if input.model_type == 'topbel_lasso':
    #     model_data = lasso_model
    # elif input.model_type == 'topbel_ridge':
    #     model_data = ridge_model
    # elif input.model_type == 'bombom_lasso':
    #     model_data = bombom_model
    
    rf_spec = RF_MODEL_SPECS.get(input.model_type)
    if rf_spec:
        model_data = rf_spec.get('model')
        is_rf_model = True
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
        df_hist, _, _ = build_history_from_dataset(
            input.model_type,
            int(input.year),
            int(input.month),
            input.campaign,
            input.seasonality,
        )
    
    # Processar de acordo com o tipo de modelo
    if is_rf_model:
        rf_spec = RF_MODEL_SPECS.get(input.model_type)
        feature_columns = rf_spec.get('feature_columns') if rf_spec else DEFAULT_RF_FEATURES
        # Para modelos Random Forest, usamos feature importance em vez de coeficientes lineares
        X = prepare_rf_features(
            df_hist[:-1] if len(df_hist) > 1 else df_hist,
            int(input.year) if input.year else df_hist.iloc[-1]['year'],
            int(input.month) if input.month else df_hist.iloc[-1]['month'],
            int(input.campaign) if input.campaign else int(df_hist.iloc[-1]['campaign']),
            int(input.seasonality) if input.seasonality else int(df_hist.iloc[-1]['seasonality']),
            feature_columns or DEFAULT_RF_FEATURES,
        )
        
        pred_raw = float(model_data.predict(X)[0])
        feature_values = X.iloc[0].to_dict()
        
        # Para Random Forest, mostramos a importância das features em vez de contribuições lineares
        feature_importance = getattr(model_data, 'feature_importances_', None)
        top_important = []
        if feature_importance is not None:
            importance_pairs = list(zip(X.columns, feature_importance))
            top_important = sorted(importance_pairs, key=lambda x: x[1], reverse=True)[:15]
        
        return {
            "model": input.model_type,
            "model_type": "RandomForest",
            "target": df_hist.iloc[-1][['year','month','campaign','seasonality']].to_dict(),
            "feature_values_preview": {k: feature_values[k] for k in list(feature_values)[:20]},
            # Return prediction as integer (rounded). If None, keep None.
            "prediction": None if pred_raw is None else int(round(pred_raw)),
            # Clip negative predictions to zero and return as integer
            "prediction_after_clip": None if pred_raw is None else int(round(max(pred_raw, 0.0))),
            "top_feature_importance": top_important,
            "note": "Random Forest models use feature importance instead of linear coefficients"
        }
    else:
        # Código original para modelos lineares
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
            "model_type": "Linear",
            "target": df_hist.iloc[-1][['year','month','campaign','seasonality']].to_dict(),
            "missing_columns": missing_cols,
            "extra_columns": extra_cols,
            "feature_values": feature_values,
            "scaled_values_preview": {k: scaled_values[k] for k in list(scaled_values)[:20]},
            "prediction_linear_raw": pred_raw,
            # Also provide a unified "prediction" key as integer for consistency
            "prediction": None if pred_raw is None else int(round(pred_raw)),
            "prediction_after_clip": None if pred_raw is None else int(round(max(pred_raw, 0.0))),
            "top_positive_contribs": top_pos,
            "top_negative_contribs": top_neg
        }

@app.get("/")
def root():
    return {"message": "DemandAI FastAPI está rodando! Use o endpoint /predict."}

# Carregar todos os modelos na inicialização
load_models()
