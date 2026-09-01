"""Construção dos três modelos comparados no projeto (passo 8 do guia):
regressão logística (baseline interpretável), LightGBM (gradient boosting)
e Isolation Forest (detecção de anomalia, treinado só com transações
legítimas).
"""

from lightgbm import LGBMClassifier
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Nome do passo classificador dentro do Pipeline da regressão logística.
# Usado por quem precisa ajustar hiperparâmetros do estimador final
# (ex.: class_weight) via set_params("clf__...").
LOGREG_CLASSIFIER_STEP = "clf"


def build_logistic_regression(config: dict) -> Pipeline:
    """Regressão logística com padronização das features.

    `Time` e `Amount` estão em escala muito diferente das componentes
    PCA `V1..V28`; sem padronização a regressão logística converge mal e
    os coeficientes ficam dominados pela escala, não pelo sinal.
    """
    params = config["model"]["logistic_regression"]
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (LOGREG_CLASSIFIER_STEP, LogisticRegression(**params)),
        ]
    )


def build_lightgbm(config: dict) -> LGBMClassifier:
    params = config["model"]["lightgbm"]
    return LGBMClassifier(**params)


def build_isolation_forest(config: dict) -> IsolationForest:
    params = config["model"]["isolation_forest"]
    return IsolationForest(**params)
