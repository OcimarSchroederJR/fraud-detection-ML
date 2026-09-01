from lightgbm import LGBMClassifier
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.train.models import build_isolation_forest, build_lightgbm, build_logistic_regression

_CONFIG = {
    "model": {
        "logistic_regression": {"class_weight": "balanced", "max_iter": 100, "random_state": 42},
        "lightgbm": {"n_estimators": 10, "learning_rate": 0.1, "num_leaves": 7, "random_state": 42},
        "isolation_forest": {"n_estimators": 10, "contamination": 0.01, "random_state": 42},
    }
}


def test_build_logistic_regression_returns_scaled_pipeline():
    model = build_logistic_regression(_CONFIG)
    assert isinstance(model, Pipeline)
    assert isinstance(model.named_steps["scaler"], StandardScaler)

    clf = model.named_steps["clf"]
    assert isinstance(clf, LogisticRegression)
    assert clf.class_weight == "balanced"
    assert clf.random_state == 42


def test_build_lightgbm_returns_configured_estimator():
    model = build_lightgbm(_CONFIG)
    assert isinstance(model, LGBMClassifier)
    assert model.n_estimators == 10


def test_build_isolation_forest_returns_configured_estimator():
    model = build_isolation_forest(_CONFIG)
    assert isinstance(model, IsolationForest)
    assert model.contamination == 0.01
