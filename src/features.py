"""Ordem canônica das features do modelo, em um único lugar.

O modelo é treinado com as colunas na ordem do CSV original do dataset
(Time, V1..V28, Amount). Essa ordem precisa ser idêntica no treino, na
API de serving e no dashboard — por isso vive aqui e é importada pelos
três, em vez de ser redeclarada em cada arquivo.
"""

from __future__ import annotations

import pandas as pd

FEATURE_ORDER: list[str] = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
TARGET_COLUMN = "Class"


def feature_frame(record: dict) -> pd.DataFrame:
    """Constrói um DataFrame de uma linha com as colunas na ordem canônica.

    Passar um DataFrame com nomes de coluna (em vez de uma lista crua) ao
    ``predict``/``predict_proba`` evita o aviso "X does not have valid
    feature names" do scikit-learn/LightGBM e torna a inferência robusta
    a reordenação acidental dos atributos.
    """
    return pd.DataFrame([[record[name] for name in FEATURE_ORDER]], columns=FEATURE_ORDER)
