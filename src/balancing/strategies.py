"""Estratégias de balanceamento de classes (passo 6 do guia).

Três abordagens comparadas: ponderação de classe, SMOTE (aplicado
apenas sobre o treino) e detecção de anomalia via Isolation Forest.
"""

from imblearn.over_sampling import SMOTE


def apply_smote(X_train, y_train, k_neighbors: int = 5, random_state: int = 42):
    """Sobreamostra a classe minoritária apenas no conjunto de treino.

    Nunca deve ser aplicado ao conjunto de teste: isso vazaria exemplos
    sintéticos derivados de fraudes reais do teste para o treino.
    """
    smote = SMOTE(k_neighbors=k_neighbors, random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    return X_resampled, y_resampled


def class_weight_dict(y_train) -> dict:
    """Calcula pesos de classe inversamente proporcionais à frequência."""
    n_samples = len(y_train)
    n_classes = y_train.nunique()
    counts = y_train.value_counts()
    return {cls: n_samples / (n_classes * count) for cls, count in counts.items()}
