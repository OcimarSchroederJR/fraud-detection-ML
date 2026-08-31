import pandas as pd

from src.balancing.strategies import apply_smote, class_weight_dict


def test_class_weight_dict_gives_higher_weight_to_minority_class():
    y_train = pd.Series([0] * 90 + [1] * 10)
    weights = class_weight_dict(y_train)
    assert weights[1] > weights[0]


def test_apply_smote_balances_classes():
    rng_state = 42
    X_train = pd.DataFrame(
        {"a": list(range(100)), "b": list(range(100, 200))}
    )
    y_train = pd.Series([0] * 90 + [1] * 10)

    X_resampled, y_resampled = apply_smote(X_train, y_train, k_neighbors=3, random_state=rng_state)

    counts = y_resampled.value_counts()
    assert counts[0] == counts[1]
    assert len(X_resampled) == len(y_resampled)
