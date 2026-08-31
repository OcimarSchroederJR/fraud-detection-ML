FROM python:3.12-slim

WORKDIR /app

# libgomp1: runtime do OpenMP exigido pela lib nativa do LightGBM
# (sem isso, joblib.load falha com "libgomp.so.1: cannot open shared object file")
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-serving.txt .
RUN pip install --no-cache-dir -r requirements-serving.txt

COPY src/ src/
COPY config/ config/

EXPOSE 8000

CMD ["uvicorn", "src.serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
