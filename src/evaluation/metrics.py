"""Métricas de avaliação apropriadas para classes extremamente
desbalanceadas (passo 9 do guia): PR-AUC como métrica principal e
custo esperado por transação para escolha do limiar de decisão.

Acurácia é deliberadamente evitada aqui: prever sempre a classe
majoritária já produziria acurácia acima de 99% sem detectar uma
única fraude.
"""

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve


def pr_auc(y_true, y_scores) -> float:
    """Área sob a curva de precisão e recall."""
    return average_precision_score(y_true, y_scores)


def expected_cost(y_true, y_pred, false_negative_cost: float, false_positive_cost: float) -> float:
    """Custo médio esperado por transação, dado um limiar já aplicado.

    false_negative_cost: custo estimado de uma fraude não detectada.
    false_positive_cost: custo estimado de bloquear/sinalizar uma
    transação legítima indevidamente.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    false_negatives = np.sum((y_true == 1) & (y_pred == 0))
    false_positives = np.sum((y_true == 0) & (y_pred == 1))

    total_cost = false_negatives * false_negative_cost + false_positives * false_positive_cost
    return total_cost / len(y_true)


def best_threshold_by_cost(y_true, y_scores, false_negative_cost: float, false_positive_cost: float):
    """Varre os limiares da curva PR e retorna o que minimiza o custo esperado."""
    _, _, thresholds = precision_recall_curve(y_true, y_scores)
    best_threshold = 0.5
    best_cost = float("inf")

    for threshold in thresholds:
        y_pred = (np.asarray(y_scores) >= threshold).astype(int)
        cost = expected_cost(y_true, y_pred, false_negative_cost, false_positive_cost)
        if cost < best_cost:
            best_cost = cost
            best_threshold = threshold

    return best_threshold, best_cost
