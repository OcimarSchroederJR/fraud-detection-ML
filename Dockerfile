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

# roda como usuário sem privilégios em vez de root
RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# readiness: o container só é considerado saudável quando o modelo carregou
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/ready').status==200 else 1)"]

CMD ["uvicorn", "src.serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
