import numpy as np
import pandas as pd

from src.evaluation.run_comparison import evaluate_isolation_forest, evaluate_supervised
from src.train.models import build_lightgbm

_CONFIG = {
    "model": {
        "lightgbm": {"n_estimators": 15, "learning_rate": 0.2, "num_leaves": 7, "random_state": 42},
        "isolation_forest": {"n_estimators": 20, "contamination": 0.05, "random_state": 42},
    },
    "cost_model": {"false_negative_cost": 100.0, "false_positive_cost": 5.0},
}

_EXPECTED_KEYS = {
    "pr_auc",
    "threshold",
    "precision_at_threshold",
    "recall_at_threshold",
    "expected_cost",
    "latency_p95_ms",
    "latency_p99_ms",
}


def _toy_data(n=300, n_features=4, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, n_features)), columns=[f"f{i}" for i in range(n_features)])
    y = pd.Series((X["f0"] + rng.normal(scale=0.1, size=n) > 1.5).astype(int))
    return X, y


def test_evaluate_supervised_returns_expected_keys():
    X_train, y_train = _toy_data(seed=0)
    X_val, y_val = _toy_data(seed=1)
    X_test, y_test = _toy_data(seed=2)

    model = build_lightgbm(_CONFIG)
    model.fit(X_train, y_train)

    metrics = evaluate_supervised(model, X_val, y_val, X_test, y_test, _CONFIG["cost_model"], X_test.head(20))

    assert set(metrics.keys()) == _EXPECTED_KEYS
    assert 0.0 <= metrics["pr_auc"] <= 1.0


def test_evaluate_supervised_threshold_comes_from_validation_not_test():
    """O limiar escolhido não deve depender do conteúdo do teste: dois
    testes diferentes, com a mesma validação, devem produzir o mesmo
    limiar."""
    X_train, y_train = _toy_data(seed=0)
    X_val, y_val = _toy_data(seed=1)
    X_test_a, y_test_a = _toy_data(seed=2)
    X_test_b, y_test_b = _toy_data(seed=3)

    model = build_lightgbm(_CONFIG)
    model.fit(X_train, y_train)

    metrics_a = evaluate_supervised(model, X_val, y_val, X_test_a, y_test_a, _CONFIG["cost_model"], X_test_a.head(20))
    metrics_b = evaluate_supervised(model, X_val, y_val, X_test_b, y_test_b, _CONFIG["cost_model"], X_test_b.head(20))

    assert metrics_a["threshold"] == metrics_b["threshold"]


def test_evaluate_isolation_forest_returns_expected_keys():
    X_train, y_train = _toy_data(seed=0)
    X_val, y_val = _toy_data(seed=1)
    X_test, y_test = _toy_data(seed=2)
    X_legit = X_train[y_train == 0]

    metrics = evaluate_isolation_forest(X_legit, X_val, y_val, X_test, y_test, _CONFIG, X_test.head(20))

    assert set(metrics.keys()) == _EXPECTED_KEYS
    assert 0.0 <= metrics["pr_auc"] <= 1.0
