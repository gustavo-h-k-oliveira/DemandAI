# Base Python 3.12
FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1
WORKDIR /app

# Dependências do sistema (opcional, úteis em alguns casos)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates \
  && rm -rf /var/lib/apt/lists/*

# Instalar dependências primeiro (cache)
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Ambiente de desenvolvimento (hot reload)
FROM base AS dev
EXPOSE 8000
# Código montado via bind pelo compose
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# Produção
FROM base AS prod
COPY . .
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]