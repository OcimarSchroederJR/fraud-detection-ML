"""Script executável que treina o LightGBM (ponderação de classe, split
temporal) sobre o dataset real e calcula o ranking de importância via
SHAP (passo 14 do guia).

Uso:
    python -m src.evaluation.run_shap_report
"""

from pathlib import Path

from src.balancing.strategies import class_weight_dict
from src.evaluation.interpretability import compute_shap_values, mean_absolute_shap_importance
from src.ingestion.load_data import load_config, load_raw_transactions
from src.preprocessing.schema import validate_transactions
from src.preprocessing.split import temporal_split
from src.train.models import build_lightgbm

FEATURE_COLUMNS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
TARGET_COLUMN = "Class"


def run(config_path=None, shap_sample_size: int = 5000):
    config = load_config(config_path) if config_path else load_config()
    df = load_raw_transactions(config_path) if config_path else load_raw_transactions()
    df = validate_transactions(df)

    train_df, test_df = temporal_split(
        df, time_column=config["split"]["time_column"], test_size=config["split"]["test_size"]
    )
    X_train, y_train = train_df[FEATURE_COLUMNS], train_df[TARGET_COLUMN]
    X_test = test_df[FEATURE_COLUMNS]

    model = build_lightgbm(config)
    model.set_params(class_weight=class_weight_dict(y_train))
    model.fit(X_train, y_train)

    X_sample = X_test.sample(n=min(shap_sample_size, len(X_test)), random_state=42)
    shap_values = compute_shap_values(model, X_sample)
    importance = mean_absolute_shap_importance(shap_values, X_sample.columns)
    return importance


if __name__ == "__main__":
    importance = run()
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    importance.to_csv(output_dir / "shap_importance.csv", header=["mean_abs_shap"])
    print(importance.to_string())
