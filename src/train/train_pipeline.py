"""Pipeline de treino ponta a ponta: carrega os dados, valida o schema,
faz a divisão temporal, aplica a estratégia de balanceamento escolhida,
treina o modelo, avalia e registra tudo no MLflow.

Executável como script isolado:
    python -m src.train.train_pipeline
"""

import argparse
from pathlib import Path

import joblib
import mlflow

from src.balancing.strategies import apply_smote, class_weight_dict
from src.evaluation.metrics import expected_cost, pr_auc
from src.ingestion.load_data import load_config, load_raw_transactions
from src.preprocessing.schema import validate_transactions
from src.preprocessing.split import temporal_split
from src.train.models import build_lightgbm, build_logistic_regression

FEATURE_COLUMNS_TEMPLATE = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
TARGET_COLUMN = "Class"


def run(config_path: Path = None, model_name: str = "lightgbm"):
    config = load_config(config_path) if config_path else load_config()

    df = load_raw_transactions(config_path) if config_path else load_raw_transactions()
    df = validate_transactions(df)

    train_df, test_df = temporal_split(
        df,
        time_column=config["split"]["time_column"],
        test_size=config["split"]["test_size"],
    )

    X_train, y_train = train_df[FEATURE_COLUMNS_TEMPLATE], train_df[TARGET_COLUMN]
    X_test, y_test = test_df[FEATURE_COLUMNS_TEMPLATE], test_df[TARGET_COLUMN]

    strategy = config["balancing"]["strategy"]
    if strategy == "smote":
        X_train, y_train = apply_smote(
            X_train,
            y_train,
            k_neighbors=config["balancing"]["smote"]["k_neighbors"],
            random_state=config["balancing"]["smote"]["random_state"],
        )
        model = build_lightgbm(config) if model_name == "lightgbm" else build_logistic_regression(config)
    elif strategy == "class_weight":
        weights = class_weight_dict(y_train)
        if model_name == "lightgbm":
            model = build_lightgbm(config)
            model.set_params(class_weight=weights)
        else:
            model = build_logistic_regression(config)
    else:
        raise ValueError(f"Estratégia de balanceamento desconhecida: '{strategy}'")

    model.fit(X_train, y_train)

    y_scores = model.predict_proba(X_test)[:, 1]
    metrics = {
        "pr_auc": pr_auc(y_test, y_scores),
        "expected_cost": expected_cost(
            y_test,
            (y_scores >= 0.5).astype(int),
            false_negative_cost=config["cost_model"]["false_negative_cost"],
            false_positive_cost=config["cost_model"]["false_positive_cost"],
        ),
    }

    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])
    with mlflow.start_run():
        mlflow.log_param("model", model_name)
        mlflow.log_param("balancing_strategy", strategy)
        mlflow.log_metrics(metrics)

    model_output_dir = Path(config["paths"]["model_output_dir"])
    model_output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_output_dir / "model.joblib")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Treina o modelo de detecção de fraude.")
    parser.add_argument("--model", default="lightgbm", choices=["lightgbm", "logistic_regression"])
    args = parser.parse_args()
    resulting_metrics = run(model_name=args.model)
    print(resulting_metrics)
