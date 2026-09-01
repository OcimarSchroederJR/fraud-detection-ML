"""Pipeline de treino ponta a ponta: carrega os dados, valida o schema,
faz a divisão temporal, aplica a estratégia de balanceamento escolhida,
treina o modelo, avalia e registra tudo no MLflow.

Executável como script isolado:
    python -m src.train.train_pipeline
"""

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import joblib
import mlflow
import sklearn

from src.balancing.strategies import apply_smote, class_weight_dict
from src.evaluation.metrics import best_threshold_by_cost, expected_cost, pr_auc
from src.features import FEATURE_ORDER, TARGET_COLUMN
from src.ingestion.load_data import load_config, load_raw_transactions
from src.preprocessing.schema import validate_transactions
from src.preprocessing.split import temporal_train_val_test_split
from src.train.models import build_lightgbm, build_logistic_regression
from src.train.tune import run_study

FEATURE_COLUMNS_TEMPLATE = FEATURE_ORDER


def _current_git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2], text=True
        ).strip()
    except Exception:
        return None


def run(config_path: Path | None = None, model_name: str = "lightgbm", tune: bool = False, n_trials: int = 15):
    config = load_config(config_path) if config_path else load_config()

    df = load_raw_transactions(config_path) if config_path else load_raw_transactions()
    df = validate_transactions(df)

    validation_size = config["split"].get("validation_size", 0.2)
    train_df, val_df, test_df = temporal_train_val_test_split(
        df,
        time_column=config["split"]["time_column"],
        test_size=config["split"]["test_size"],
        validation_size=validation_size,
    )

    X_train, y_train = train_df[FEATURE_COLUMNS_TEMPLATE], train_df[TARGET_COLUMN]
    X_val, y_val = val_df[FEATURE_COLUMNS_TEMPLATE], val_df[TARGET_COLUMN]
    X_test, y_test = test_df[FEATURE_COLUMNS_TEMPLATE], test_df[TARGET_COLUMN]

    tuning_info = None
    if tune and model_name == "lightgbm":
        study = run_study(X_train, y_train, n_trials=n_trials, n_splits=3)
        tuning_info = {
            "n_trials": n_trials,
            "best_cv_pr_auc": study.best_value,
            "tuned_params": study.best_params,
        }

    def build_lgbm():
        model = build_lightgbm(config)
        if tuning_info:
            model.set_params(**tuning_info["tuned_params"])
        return model

    strategy = config["balancing"]["strategy"]
    if strategy == "smote":
        X_train, y_train = apply_smote(
            X_train,
            y_train,
            k_neighbors=config["balancing"]["smote"]["k_neighbors"],
            random_state=config["balancing"]["smote"]["random_state"],
        )
        model = build_lgbm() if model_name == "lightgbm" else build_logistic_regression(config)
    elif strategy == "class_weight":
        weights = class_weight_dict(y_train)
        if model_name == "lightgbm":
            model = build_lgbm()
            model.set_params(class_weight=weights)
        else:
            model = build_logistic_regression(config)
    else:
        raise ValueError(f"Estratégia de balanceamento desconhecida: '{strategy}'")

    model.fit(X_train, y_train)

    fn_cost = config["cost_model"]["false_negative_cost"]
    fp_cost = config["cost_model"]["false_positive_cost"]

    # O limiar de decisão é escolhido no conjunto de validação (minimizando
    # o custo esperado), nunca no teste — usar o teste para essa escolha
    # vazaria informação da avaliação final para uma decisão de modelo.
    val_scores = model.predict_proba(X_val)[:, 1]
    decision_threshold, val_expected_cost = best_threshold_by_cost(
        y_val, val_scores, false_negative_cost=fn_cost, false_positive_cost=fp_cost
    )
    decision_threshold = float(decision_threshold)

    y_scores = model.predict_proba(X_test)[:, 1]
    metrics = {
        "pr_auc": pr_auc(y_test, y_scores),
        "expected_cost": expected_cost(
            y_test,
            (y_scores >= decision_threshold).astype(int),
            false_negative_cost=fn_cost,
            false_positive_cost=fp_cost,
        ),
        "decision_threshold": decision_threshold,
        "validation_expected_cost": float(val_expected_cost),
    }

    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])
    with mlflow.start_run():
        mlflow.log_param("model", model_name)
        mlflow.log_param("balancing_strategy", strategy)
        mlflow.log_param("hyperparameter_tuning", tune)
        if tuning_info:
            mlflow.log_params({f"tuned_{k}": v for k, v in tuning_info["tuned_params"].items()})
            mlflow.log_metric("tuning_best_cv_pr_auc", tuning_info["best_cv_pr_auc"])
        mlflow.log_metrics(metrics)

    model_output_dir = Path(config["paths"]["model_output_dir"])
    model_output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_output_dir / "model.joblib")

    versions = {"python": sys.version.split()[0], "scikit_learn": sklearn.__version__}
    if model_name == "lightgbm":
        import lightgbm

        versions["lightgbm"] = lightgbm.__version__

    metadata = {
        "model_name": model_name,
        "balancing_strategy": strategy,
        "feature_order": FEATURE_COLUMNS_TEMPLATE,
        "decision_threshold": decision_threshold,
        "hyperparameters": {k: str(v) for k, v in model.get_params().items()},
        "hyperparameter_tuning": tuning_info,
        "split": {
            "type": "temporal",
            "time_column": config["split"]["time_column"],
            "test_size": config["split"]["test_size"],
            "validation_size": validation_size,
            "threshold_used_for_expected_cost": decision_threshold,
            "threshold_selection": "custo esperado mínimo no conjunto de validação",
        },
        "cost_model": {"false_negative_cost": fn_cost, "false_positive_cost": fp_cost},
        "metrics_on_test": metrics,
        "dataset": {
            "source": "Kaggle - mlg-ulb/creditcardfraud",
            "n_rows": len(df),
            "n_fraud": int(df[TARGET_COLUMN].sum()),
        },
        "trained_at_utc": datetime.now(UTC).isoformat(),
        "git_commit": _current_git_commit(),
        "library_versions": versions,
    }
    with open(model_output_dir / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2, default=str)

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Treina o modelo de detecção de fraude.")
    parser.add_argument("--model", default="lightgbm", choices=["lightgbm", "logistic_regression"])
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Ajusta os hiperparâmetros do LightGBM com Optuna antes do treino final (passo 10 do guia).",
    )
    parser.add_argument("--n-trials", type=int, default=15, help="Número de tentativas do Optuna quando --tune é usado.")
    args = parser.parse_args()
    resulting_metrics = run(model_name=args.model, tune=args.tune, n_trials=args.n_trials)
    print(resulting_metrics)
