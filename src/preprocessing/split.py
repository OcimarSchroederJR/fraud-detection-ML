"""Divisão dos dados em treino e teste: temporal e aleatória.

O guia do projeto pede as duas divisões para comparação: a temporal
(mais antigas para treino, mais recentes para teste) reproduz o cenário
real de um sistema em produção; a aleatória serve como referência.
"""

import pandas as pd
from sklearn.model_selection import train_test_split


def temporal_split(df: pd.DataFrame, time_column: str = "Time", test_size: float = 0.2):
    """Ordena por tempo e reserva a fração mais recente para teste."""
    df_sorted = df.sort_values(time_column).reset_index(drop=True)
    cutoff = int(len(df_sorted) * (1 - test_size))
    train_df = df_sorted.iloc[:cutoff].copy()
    test_df = df_sorted.iloc[cutoff:].copy()
    return train_df, test_df


def random_split(df: pd.DataFrame, target_column: str = "Class", test_size: float = 0.2, random_state: int = 42):
    """Divisão aleatória estratificada pela classe, usada como comparação."""
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df[target_column],
        random_state=random_state,
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)
