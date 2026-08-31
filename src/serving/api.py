"""Serviço mínimo de inferência (passo 11 do guia).

Expõe o modelo treinado por trás de uma rota de API, recebendo os
atributos de uma transação e devolvendo a probabilidade de fraude e a
decisão sob o limiar escolhido em config.yaml. Mantido deliberadamente
leve, com o mínimo de dependências, já que é o código que rodaria em
produção sob restrição de latência.
"""

import os
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.ingestion.load_data import load_config

app = FastAPI(title="Fraud Detection API")

CONFIG = load_config()
MODEL_PATH = Path(os.environ.get("MODEL_PATH", "models/model.joblib"))
DECISION_THRESHOLD = float(os.environ.get("DECISION_THRESHOLD", 0.5))

_model = None


def get_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise HTTPException(
                status_code=503,
                detail=f"Modelo não encontrado em '{MODEL_PATH}'. Treine e salve um modelo antes de servir.",
            )
        _model = joblib.load(MODEL_PATH)
    return _model


class Transaction(BaseModel):
    Time: float
    Amount: float = Field(ge=0)
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float


class PredictionResponse(BaseModel):
    fraud_probability: float
    is_fraud: bool
    threshold: float


@app.get("/health")
def health():
    return {"status": "ok"}


# O modelo é treinado com as colunas na ordem do CSV original
# (Time, V1..V28, Amount), diferente da ordem de declaração do
# schema Pydantic acima — por isso a ordem é explícita aqui.
FEATURE_ORDER = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]


@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: Transaction):
    model = get_model()
    features = [[getattr(transaction, name) for name in FEATURE_ORDER]]
    fraud_probability = float(model.predict_proba(features)[0][1])
    return PredictionResponse(
        fraud_probability=fraud_probability,
        is_fraud=fraud_probability >= DECISION_THRESHOLD,
        threshold=DECISION_THRESHOLD,
    )
