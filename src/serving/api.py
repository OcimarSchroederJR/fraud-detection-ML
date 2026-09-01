"""Serviço mínimo de inferência (passo 11 do guia).

Expõe o modelo treinado por trás de uma rota de API, recebendo os
atributos de uma transação e devolvendo a probabilidade de fraude e a
decisão sob o limiar escolhido. Mantido deliberadamente leve, com o
mínimo de dependências, já que é o código que rodaria em produção sob
restrição de latência.

O limiar de decisão é lido de ``models/model_metadata.json``
(``decision_threshold``, escolhido no conjunto de validação por custo
esperado durante o treino). A variável de ambiente ``DECISION_THRESHOLD``,
se definida, tem precedência.
"""

import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.features import FEATURE_ORDER, feature_frame

logger = logging.getLogger("fraud_detection.api")

MODEL_PATH = Path(os.environ.get("MODEL_PATH", "models/model.joblib"))
METADATA_PATH = Path(os.environ.get("MODEL_METADATA_PATH", "models/model_metadata.json"))
DEFAULT_THRESHOLD = 0.5

_model = None
_threshold = None


def _load_threshold() -> float:
    env_value = os.environ.get("DECISION_THRESHOLD")
    if env_value is not None:
        return float(env_value)
    try:
        metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
        return float(metadata.get("decision_threshold", DEFAULT_THRESHOLD))
    except (OSError, ValueError):
        return DEFAULT_THRESHOLD


def get_threshold() -> float:
    global _threshold
    if _threshold is None:
        _threshold = _load_threshold()
    return _threshold


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


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Carrega o modelo no startup (fail-fast) em vez de na primeira request."""
    if MODEL_PATH.exists():
        try:
            get_model()
            logger.info("Modelo carregado de %s (limiar=%.4f)", MODEL_PATH, get_threshold())
        except Exception:
            # startup não deve derrubar o processo silenciosamente; o erro fica no log
            logger.exception("Falha ao carregar o modelo no startup")
    else:
        logger.warning("Modelo ausente em %s; /predict responderá 503 até um modelo estar disponível", MODEL_PATH)
    yield


app = FastAPI(title="Fraud Detection API", lifespan=lifespan)


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
    """Liveness: o processo está de pé."""
    return {"status": "ok"}


@app.get("/ready")
def ready():
    """Readiness: o modelo está carregado e a rota /predict pode responder."""
    if _model is None and not MODEL_PATH.exists():
        raise HTTPException(status_code=503, detail="Modelo ainda não disponível.")
    return {"status": "ready", "model_loaded": _model is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: Transaction):
    model = get_model()
    threshold = get_threshold()
    features = feature_frame({name: getattr(transaction, name) for name in FEATURE_ORDER})
    fraud_probability = float(model.predict_proba(features)[0][1])
    return PredictionResponse(
        fraud_probability=fraud_probability,
        is_fraud=fraud_probability >= threshold,
        threshold=threshold,
    )
