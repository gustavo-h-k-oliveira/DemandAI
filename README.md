# DemandAI

Plataforma de predição de demanda com FastAPI e modelos scikit-learn.

## Como rodar localmente (sem Docker)

- Requisitos: Python 3.11+
- Instalar dependências: `pip install -r requirements.txt`
- Rodar a API: `uvicorn app:app --reload`

## Como rodar com Docker

- Build da imagem: `docker build -t demandai .`
- Subir o container: `docker run -p 8000:8000 demandai`
- Via Compose: `docker compose up --build`

## Endpoints principais

- `GET /` — status básico
- `GET /form` — formulário HTML simples para teste
- `POST /predict` — previsão com `model_type`, `year`, `month`, `campaign`, `seasonality` e opcional `history`
- `POST /_debug/inspect` — inspeção de features e contribuições do modelo

Certifique-se de que os artefatos de modelo estejam em `models/` e que `dataset.csv` exista para o modo de histórico automático.
