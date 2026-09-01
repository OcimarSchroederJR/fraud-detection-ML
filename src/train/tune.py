"""Busca de hiperparâmetros do LightGBM via otimização bayesiana (passo 10).

Usa Optuna em vez de busca em grade porque o espaço de hiperparâmetros
do gradient boosting é grande demais para uma busca exaustiva caber no
orçamento de tempo de uma disciplina.
"""

import optuna
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score


def _objective(trial: optuna.Trial, X, y, n_splits: int) -> float:
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 7, 63),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        # a busca precisa refletir o tratamento de desbalanceamento usado no
        # treino final; sem isto os hiperparâmetros são otimizados para um
        # cenário (sem ponderação) diferente do que será de fato treinado
        "class_weight": "balanced",
        "verbosity": -1,
        "random_state": 42,
    }
    model = LGBMClassifier(**params)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, scoring="average_precision", cv=cv)
    return scores.mean()


def run_study(
    X, y, n_trials: int = 30, n_splits: int = 3, direction: str = "maximize", seed: int = 42
) -> optuna.Study:
    """Roda a busca bayesiana maximizando PR-AUC via validação cruzada.

    O sampler é semeado para que a busca seja reprodutível — os metadados
    do modelo alegam reprodutibilidade do treino, o que só vale se a
    escolha de hiperparâmetros também for determinística.
    """
    study = optuna.create_study(direction=direction, sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(lambda trial: _objective(trial, X, y, n_splits), n_trials=n_trials)
    return study
