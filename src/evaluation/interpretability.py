"""Interpretabilidade via SHAP (passo 14 do guia).

Usa o TreeExplainer, otimizado para modelos baseados em árvore como
LightGBM e XGBoost, o que torna o cálculo consideravelmente mais
rápido do que a versão genérica da biblioteca shap.
"""

import numpy as np
import pandas as pd
import shap


def compute_shap_values(model, X: pd.DataFrame):
    """Calcula os valores de SHAP para um modelo baseado em árvore.

    Retorna a matriz de valores de SHAP para a classe positiva (fraude).
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Para classificação binária, alguns modelos retornam uma lista
    # [valores_classe_0, valores_classe_1]; outros retornam já a matriz
    # da classe positiva.
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    return shap_values


def mean_absolute_shap_importance(shap_values, feature_names) -> pd.Series:
    """Ranking de importância das features pela média do valor absoluto de SHAP."""
    importance = np.abs(shap_values).mean(axis=0)
    return pd.Series(importance, index=feature_names).sort_values(ascending=False)
