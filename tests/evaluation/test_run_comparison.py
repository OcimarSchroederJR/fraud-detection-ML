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


def _toy_data(n=300, n_features=4):
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(n, n_features)), columns=[f"f{i}" for i in range(n_features)])
    y = pd.Series((X["f0"] + rng.normal(scale=0.1, size=n) > 1.5).astype(int))
    return X, y


def test_evaluate_supervised_returns_expected_keys():
    X, y = _toy_data()
    model = build_lightgbm(_CONFIG)
    model.fit(X, y)

    metrics = evaluate_supervised(model, X, y, _CONFIG["cost_model"], X.head(20))

    expected_keys = {
        "pr_auc",
        "threshold",
        "precision_at_threshold",
        "recall_at_threshold",
        "expected_cost",
        "latency_p95_ms",
        "latency_p99_ms",
    }
    assert set(metrics.keys()) == expected_keys
    assert 0.0 <= metrics["pr_auc"] <= 1.0


def test_evaluate_isolation_forest_returns_expected_keys():
    X, y = _toy_data()
    X_legit = X[y == 0]

    metrics = evaluate_isolation_forest(X_legit, X, y, _CONFIG, X.head(20))

    expected_keys = {
        "pr_auc",
        "threshold",
        "precision_at_threshold",
        "recall_at_threshold",
        "expected_cost",
        "latency_p95_ms",
        "latency_p99_ms",
    }
    assert set(metrics.keys()) == expected_keys
    assert 0.0 <= metrics["pr_auc"] <= 1.0
