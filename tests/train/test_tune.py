import numpy as np
import optuna
import pandas as pd

from src.train.tune import run_study

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _toy_dataset(n=150):
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        {
            "a": rng.normal(size=n),
            "b": rng.normal(size=n),
        }
    )
    y = (X["a"] + 0.1 * rng.normal(size=n) > 0).astype(int)
    return X, y


def test_run_study_returns_best_params_within_search_space():
    X, y = _toy_dataset()
    study = run_study(X, y, n_trials=3, n_splits=2)

    assert study.best_value is not None
    best_params = study.best_params
    assert 50 <= best_params["n_estimators"] <= 500
    assert 0.01 <= best_params["learning_rate"] <= 0.3
    assert 7 <= best_params["num_leaves"] <= 63
