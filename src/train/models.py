"""Construção dos três modelos comparados no projeto (passo 8 do guia):
regressão logística (baseline interpretável), LightGBM (gradient boosting)
e Isolation Forest (detecção de anomalia, treinado só com transações
legítimas).
"""

from lightgbm import LGBMClassifier
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression


def build_logistic_regression(config: dict) -> LogisticRegression:
    params = config["model"]["logistic_regression"]
    return LogisticRegression(**params)


def build_lightgbm(config: dict) -> LGBMClassifier:
    params = config["model"]["lightgbm"]
    return LGBMClassifier(**params)


def build_isolation_forest(config: dict) -> IsolationForest:
    params = config["model"]["isolation_forest"]
    return IsolationForest(**params)
