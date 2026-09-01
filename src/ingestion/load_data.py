"""Carregamento do dataset de transações a partir do CSV bruto."""

import logging
from pathlib import Path

import pandas as pd
import yaml

logger = logging.getLogger("fraud_detection.ingestion")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


def load_config(config_path: Path = CONFIG_PATH) -> dict:
    with open(config_path, encoding="utf-8") as f:
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
    df = pd.read_csv(raw_path)
    _check_dataset_integrity(df, config.get("data", {}))
    return df


def _check_dataset_integrity(df: pd.DataFrame, data_config: dict) -> None:
    """Avisa (sem interromper) se o CSV não bate com o dataset esperado.

    Os metadados do modelo alegam "condições exatas do treino"; um alerta
    aqui torna evidente quando alguém está treinando sobre um arquivo
    diferente do dataset público de referência.
    """
    expected_rows = data_config.get("expected_n_rows")
    if expected_rows is not None and len(df) != expected_rows:
        logger.warning(
            "Dataset com %d linhas, esperado %d — pode não ser o Credit Card Fraud Detection de referência.",
            len(df),
            expected_rows,
        )
    expected_fraud = data_config.get("expected_n_fraud")
    if expected_fraud is not None and "Class" in df and int(df["Class"].sum()) != expected_fraud:
        logger.warning(
            "Dataset com %d fraudes, esperado %d.", int(df["Class"].sum()), expected_fraud
        )
