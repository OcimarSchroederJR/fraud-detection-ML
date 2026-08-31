import numpy as np

from src.evaluation.metrics import best_threshold_by_cost, expected_cost, pr_auc


def test_pr_auc_perfect_classifier_is_one():
    y_true = [0, 0, 0, 1, 1]
    y_scores = [0.1, 0.2, 0.3, 0.9, 0.95]
    assert pr_auc(y_true, y_scores) == 1.0


def test_expected_cost_counts_false_negatives_and_positives():
    y_true = [0, 1, 0, 1]
    y_pred = [0, 0, 1, 1]  # 1 falso negativo (idx 1), 1 falso positivo (idx 2)
    cost = expected_cost(y_true, y_pred, false_negative_cost=100.0, false_positive_cost=10.0)
    assert cost == (100.0 + 10.0) / 4


def test_expected_cost_is_zero_for_perfect_predictions():
    y_true = [0, 1, 0, 1]
    y_pred = [0, 1, 0, 1]
    cost = expected_cost(y_true, y_pred, false_negative_cost=100.0, false_positive_cost=10.0)
    assert cost == 0.0


def test_best_threshold_by_cost_favors_higher_recall_when_fn_cost_dominates():
    rng = np.random.default_rng(0)
    y_true = np.array([0] * 95 + [1] * 5)
    # scores mais altos para a classe 1, com alguma sobreposição
    y_scores = np.concatenate([rng.uniform(0, 0.6, 95), rng.uniform(0.4, 1.0, 5)])

    threshold, cost = best_threshold_by_cost(
        y_true, y_scores, false_negative_cost=1000.0, false_positive_cost=1.0
    )
    assert 0.0 <= threshold <= 1.0
    assert cost >= 0.0
