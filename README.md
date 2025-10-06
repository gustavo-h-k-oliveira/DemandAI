# DemandAI

Plataforma de predição de demanda composta por uma API FastAPI (Python) e um frontend React com Vite/Tailwind.

## Visão geral

- **Backend**: expõe endpoints de previsão, inspeção e formulário rápido (`src/app.py`).
- **Frontend** (`frontend/`): interface em React/Vite para consumir o endpoint `/predict`.
- **Modelos**: artefatos `.pkl` residentes em `models/`, com análises em `models/*_analysis.png`.

## Backend

### Execução local (sem Docker)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn src.app:app --reload --host 0.0.0.0 --port 8000
```

A API ficará acessível em `http://localhost:8000`.

### Docker

```bash
# Build manual
docker build -t demandai .
docker run -p 8000:8000 demandai

# Ou via Compose (sobe apenas o backend por enquanto)
docker compose up --build
```

> ⚠️ O arquivo `docker-compose.yml` atual só provisiona o backend FastAPI. Para subir o frontend pelo Compose, crie um novo serviço baseado em Node/Vite ou sirva o build estático em outro container.

## Frontend

Pré-requisitos: Node.js 18+ e npm.

```bash
cd frontend
npm install

# Desenvolvimento (Vite) em http://localhost:5173
npm run dev

# Build de produção em frontend/dist
npm run build
```

Por padrão, o Vite proxy-a as requisições para `http://localhost:8000/predict`. Garanta que o backend esteja rodando antes de iniciar o frontend.

## Endpoints principais da API

- `GET /` — status básico
- `GET /form` — formulário HTML simples para teste
- `POST /predict` — previsão com `model_type`, `year`, `month`, `campaign`, `seasonality` e opcional `history`
- `POST /_debug/inspect` — inspeção de features e contribuições do modelo

## Artefatos necessários

- `data/dataset.csv`
- `models/bombom_moranguete_rf_model.pkl`
- `models/teta_bel_rf_model.pkl`
- `models/topbel_leite_condensado_rf_model.pkl`
- `models/topbel_tradicional_rf_model.pkl`
