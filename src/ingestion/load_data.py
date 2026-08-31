"""Carregamento do dataset de transações a partir do CSV bruto."""

from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


def load_config(config_path: Path = CONFIG_PATH) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_raw_transactions(config_path: Path = CONFIG_PATH) -> pd.DataFrame:
    """Lê o CSV bruto de transações definido em config.yaml (paths.raw_data).

    Caminhos relativos em config.yaml são resolvidos a partir da raiz do
    projeto, não do diretório de trabalho atual — importante para que
    isso funcione tanto rodando scripts a partir da raiz quanto a partir
    de notebooks em notebooks/.
    """
    config = load_config(config_path)
    raw_path = Path(config["paths"]["raw_data"])
    if not raw_path.is_absolute():
        raw_path = PROJECT_ROOT / raw_path
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Dataset bruto não encontrado em '{raw_path}'. "
            "Baixe o Credit Card Fraud Detection do Kaggle e salve-o nesse caminho."
        )
    return pd.read_csv(raw_path)
