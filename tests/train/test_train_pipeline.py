import numpy as np
import pandas as pd
import pytest

from src.train import train_pipeline


def _synthetic_transactions(n=600, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.08).astype(int)
    df = pd.DataFrame({"Time": np.arange(n, dtype=float)})
    for i in range(1, 29):
        df[f"V{i}"] = rng.normal(size=n) + y * rng.normal(0.5, 0.1)
    df["Amount"] = rng.uniform(1, 200, size=n)
    df["Class"] = y.astype("int64")
    return df


def _fake_config(tmp_path):
    return {
        "paths": {"model_output_dir": str(tmp_path / "models")},
        "split": {"time_column": "Time", "test_size": 0.2, "target_column": "Class"},
        "balancing": {
            "strategy": "class_weight",
            "smote": {"k_neighbors": 3, "random_state": 42},
        },
        "model": {
            "logistic_regression": {"class_weight": "balanced", "max_iter": 100, "random_state": 42},
            "lightgbm": {"n_estimators": 20, "learning_rate": 0.2, "num_leaves": 7, "random_state": 42},
            "isolation_forest": {"n_estimators": 20, "contamination": 0.05, "random_state": 42},
        },
        "cost_model": {"false_negative_cost": 100.0, "false_positive_cost": 5.0},
        "mlflow": {"tracking_uri": (tmp_path / "mlruns").as_uri(), "experiment_name": "test"},
    }


@pytest.fixture(autouse=True)
def _patch_data_and_config(tmp_path, monkeypatch):
    df = _synthetic_transactions()
    monkeypatch.setattr(train_pipeline, "load_raw_transactions", lambda *a, **k: df)
    monkeypatch.setattr(train_pipeline, "load_config", lambda *a, **k: _fake_config(tmp_path))
    return tmp_path


def test_run_without_tuning_saves_model_and_metadata_without_tuning_info(tmp_path):
    metrics = train_pipeline.run(model_name="lightgbm", tune=False)

    assert "pr_auc" in metrics
    model_dir = tmp_path / "models"
    assert (model_dir / "model.joblib").exists()

    import json

    metadata = json.loads((model_dir / "model_metadata.json").read_text(encoding="utf-8"))
    assert metadata["hyperparameter_tuning"] is None


def test_run_with_tuning_records_tuned_params_in_metadata(tmp_path):
    metrics = train_pipeline.run(model_name="lightgbm", tune=True, n_trials=2)

    assert "pr_auc" in metrics
    import json

    metadata = json.loads((tmp_path / "models" / "model_metadata.json").read_text(encoding="utf-8"))
    tuning = metadata["hyperparameter_tuning"]
    assert tuning is not None
    assert tuning["n_trials"] == 2
    assert set(tuning["tuned_params"].keys()) == {"n_estimators", "learning_rate", "num_leaves", "min_child_samples"}
