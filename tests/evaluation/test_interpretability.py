import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from src.evaluation.interpretability import compute_shap_values, mean_absolute_shap_importance


def _train_toy_model():
    rng = np.random.default_rng(42)
    n = 200
    X = pd.DataFrame(
        {
            "informative": rng.normal(size=n),
            "noise": rng.normal(size=n),
        }
    )
    # y depende fortemente de "informative" e não de "noise"
    y = (X["informative"] > 0).astype(int)

    model = LGBMClassifier(n_estimators=20, num_leaves=3, min_child_samples=5, verbosity=-1)
    model.fit(X, y)
    return model, X


def test_compute_shap_values_shape_matches_input():
    model, X = _train_toy_model()
    shap_values = compute_shap_values(model, X)
    assert shap_values.shape == X.shape


def test_mean_absolute_shap_importance_ranks_informative_feature_first():
    model, X = _train_toy_model()
    shap_values = compute_shap_values(model, X)

    importance = mean_absolute_shap_importance(shap_values, X.columns)

    assert importance.index[0] == "informative"
