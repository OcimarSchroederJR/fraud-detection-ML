"""Script executável que produz a matriz de comparação do passo 8:
3 modelos (regressão logística, LightGBM, Isolation Forest) sob as
estratégias de balanceamento do passo 6, nas divisões temporal e
aleatória do passo 7.

O limiar de decisão que minimiza o custo esperado (passo 9) é escolhido
sobre um conjunto de validação separado do teste — nunca sobre o
próprio teste, o que vazaria informação da avaliação final para uma
decisão de modelo. PR-AUC e as métricas finais são sempre calculadas
sobre o teste, usando o limiar já fixado na validação.

Uso:
    python -m src.evaluation.run_comparison
"""

from pathlib import Path

import pandas as pd

from src.balancing.strategies import apply_smote, class_weight_dict
from src.evaluation.latency import measure_single_transaction_latency
from src.evaluation.metrics import best_threshold_by_cost, expected_cost, pr_auc
from src.ingestion.load_data import load_config, load_raw_transactions
from src.preprocessing.schema import validate_transactions
from src.preprocessing.split import random_train_val_test_split, temporal_train_val_test_split
from src.train.models import build_isolation_forest, build_lightgbm, build_logistic_regression

FEATURE_COLUMNS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
TARGET_COLUMN = "Class"


def _fit_supervised(model, X_train, y_train, strategy, config):
    if strategy == "smote":
        X_bal, y_bal = apply_smote(
            X_train,
            y_train,
            k_neighbors=config["balancing"]["smote"]["k_neighbors"],
            random_state=config["balancing"]["smote"]["random_state"],
        )
        model.fit(X_bal, y_bal)
    elif strategy == "class_weight":
        weights = class_weight_dict(y_train)
        if "class_weight" in model.get_params():
            model.set_params(class_weight=weights)
        model.fit(X_train, y_train)
    else:
        raise ValueError(f"Estratégia desconhecida: {strategy}")
    return model


def _precision_recall_at_threshold(y_true, y_pred):
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return precision, recall


def _score_and_evaluate(y_val, y_val_scores, y_test, y_test_scores, cost_config, model, latency_sample, predict_fn=None):
    threshold, _ = best_threshold_by_cost(
        y_val,
        y_val_scores,
        false_negative_cost=cost_config["false_negative_cost"],
        false_positive_cost=cost_config["false_positive_cost"],
    )
    y_pred = (y_test_scores >= threshold).astype(int)
    precision, recall = _precision_recall_at_threshold(y_test, y_pred)
    latency = measure_single_transaction_latency(model, latency_sample, predict_fn=predict_fn)

    return {
        "pr_auc": pr_auc(y_test, y_test_scores),
        "threshold": float(threshold),
        "precision_at_threshold": precision,
        "recall_at_threshold": recall,
        "expected_cost": expected_cost(y_test, y_pred, cost_config["false_negative_cost"], cost_config["false_positive_cost"]),
        "latency_p95_ms": latency["p95_ms"],
        "latency_p99_ms": latency["p99_ms"],
    }


def evaluate_supervised(model, X_val, y_val, X_test, y_test, cost_config, latency_sample):
    y_val_scores = model.predict_proba(X_val)[:, 1]
    y_test_scores = model.predict_proba(X_test)[:, 1]
    return _score_and_evaluate(y_val, y_val_scores, y_test, y_test_scores, cost_config, model, latency_sample)


def evaluate_isolation_forest(X_train_legit, X_val, y_val, X_test, y_test, config, latency_sample):
    """Treina só sobre transações legítimas e trata o score de anomalia
    invertido como proxy da probabilidade de fraude."""
    model = build_isolation_forest(config)
    model.fit(X_train_legit)

    y_val_scores = -model.decision_function(X_val)
    y_test_scores = -model.decision_function(X_test)

    return _score_and_evaluate(
        y_val,
        y_val_scores,
        y_test,
        y_test_scores,
        config["cost_model"],
        model,
        latency_sample,
        predict_fn=lambda m, row: m.decision_function(row),
    )


def run(config_path=None, latency_sample_size: int = 500) -> pd.DataFrame:
    config = load_config(config_path) if config_path else load_config()
    df = load_raw_transactions(config_path) if config_path else load_raw_transactions()
    df = validate_transactions(df)

    split_kwargs = {
        "test_size": config["split"]["test_size"],
        "validation_size": config["split"]["validation_size"],
    }
    splits = {
        "temporal": temporal_train_val_test_split(df, time_column=config["split"]["time_column"], **split_kwargs),
        "aleatoria": random_train_val_test_split(df, target_column=config["split"]["target_column"], **split_kwargs),
    }

    model_builders = {
        "logistic_regression": build_logistic_regression,
        "lightgbm": build_lightgbm,
    }
    strategies = ["class_weight", "smote"]

    rows = []
    for split_name, (train_df, val_df, test_df) in splits.items():
        X_train, y_train = train_df[FEATURE_COLUMNS], train_df[TARGET_COLUMN]
        X_val, y_val = val_df[FEATURE_COLUMNS], val_df[TARGET_COLUMN]
        X_test, y_test = test_df[FEATURE_COLUMNS], test_df[TARGET_COLUMN]
        latency_sample = X_test.sample(n=min(latency_sample_size, len(X_test)), random_state=42)

        for model_name, builder in model_builders.items():
            for strategy in strategies:
                model = builder(config)
                model = _fit_supervised(model, X_train, y_train, strategy, config)
                metrics = evaluate_supervised(model, X_val, y_val, X_test, y_test, config["cost_model"], latency_sample)
                rows.append({"split": split_name, "model": model_name, "balancing": strategy, **metrics})

        X_train_legit = X_train[y_train == 0]
        iso_metrics = evaluate_isolation_forest(X_train_legit, X_val, y_val, X_test, y_test, config, latency_sample)
        rows.append(
            {"split": split_name, "model": "isolation_forest", "balancing": "treinado só com legítimas", **iso_metrics}
        )

    return pd.DataFrame(rows)


if __name__ == "__main__":
    results_df = run()
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    results_df.to_csv(output_dir / "comparacao_modelos.csv", index=False)
    print(results_df.to_string(index=False))
