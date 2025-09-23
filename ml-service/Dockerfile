FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_DIR=/models
WORKDIR /app
COPY requirements-ml.txt .
RUN pip install -r requirements-ml.txt
COPY . .
# Copie os artefatos de modelo para a imagem
COPY models/ /models/
EXPOSE 9000
CMD ["uvicorn", "ml.main:app", "--host", "0.0.0.0", "--port", "9000"]