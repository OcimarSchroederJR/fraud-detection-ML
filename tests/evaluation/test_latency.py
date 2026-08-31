import numpy as np

from src.evaluation.latency import compare_models_latency, measure_single_transaction_latency


class _DummyModel:
    def predict_proba(self, X):
        return np.tile([0.9, 0.1], (len(X), 1))


def test_measure_single_transaction_latency_returns_expected_keys():
    X = np.random.default_rng(0).normal(size=(10, 5))
    result = measure_single_transaction_latency(_DummyModel(), X)

    assert set(result.keys()) == {"mean_ms", "median_ms", "p95_ms", "p99_ms"}
    assert all(value >= 0 for value in result.values())


def test_compare_models_latency_returns_one_entry_per_model():
    X = np.random.default_rng(0).normal(size=(5, 3))
    models = {"modelo_a": _DummyModel(), "modelo_b": _DummyModel()}

    result = compare_models_latency(models, X)

    assert set(result.keys()) == {"modelo_a", "modelo_b"}
