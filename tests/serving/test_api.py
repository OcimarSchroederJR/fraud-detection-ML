from fastapi.testclient import TestClient

from src.serving import api

EXPECTED_FEATURE_ORDER = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]


class _StubModel:
    """Modelo falso que verifica a ordem das colunas recebidas."""

    def predict_proba(self, features):
        received = features[0]
        assert len(received) == len(EXPECTED_FEATURE_ORDER)
        return [[0.1, 0.9]]


def _sample_payload():
    payload = {"Time": 0.0, "Amount": 149.62}
    payload.update({f"V{i}": float(i) for i in range(1, 29)})
    return payload


def test_predict_returns_fraud_probability(monkeypatch):
    monkeypatch.setattr(api, "_model", _StubModel())
    client = TestClient(api.app)

    response = client.post("/predict", json=_sample_payload())

    assert response.status_code == 200
    body = response.json()
    assert body["fraud_probability"] == 0.9
    assert body["is_fraud"] is True


def test_predict_sends_features_in_dataset_column_order(monkeypatch):
    """O modelo é treinado com colunas na ordem Time, V1..V28, Amount
    (ordem do CSV original). A API precisa enviar os atributos nessa
    mesma ordem, e não na ordem de declaração do schema Pydantic.
    """
    captured = {}

    class _OrderCheckingModel:
        def predict_proba(self, features):
            captured["features"] = features[0]
            return [[0.5, 0.5]]

    monkeypatch.setattr(api, "_model", _OrderCheckingModel())
    client = TestClient(api.app)

    payload = _sample_payload()
    client.post("/predict", json=payload)

    expected = [payload[name] for name in EXPECTED_FEATURE_ORDER]
    assert captured["features"] == expected
