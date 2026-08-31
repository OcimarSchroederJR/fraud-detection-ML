"""Carregamento do dataset de transações a partir do CSV bruto."""

from pathlib import Path

import pandas as pd
import yaml

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "config.yaml"


def load_config(config_path: Path = CONFIG_PATH) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_raw_transactions(config_path: Path = CONFIG_PATH) -> pd.DataFrame:
    """Lê o CSV bruto de transações definido em config.yaml (paths.raw_data)."""
    config = load_config(config_path)
    raw_path = Path(config["paths"]["raw_data"])
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Dataset bruto não encontrado em '{raw_path}'. "
            "Baixe o Credit Card Fraud Detection do Kaggle e salve-o nesse caminho."
        )
    return pd.read_csv(raw_path)
