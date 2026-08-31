import numpy as np
import pandas as pd

from src.preprocessing.split import (
    random_split,
    random_train_val_test_split,
    temporal_split,
    temporal_train_val_test_split,
)


def _sample_df(n=100):
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "Time": np.arange(n, dtype=float),
            "Amount": rng.uniform(1, 100, size=n),
            "Class": rng.choice([0, 1], size=n, p=[0.9, 0.1]),
        }
    )


def test_temporal_split_orders_by_time():
    df = _sample_df()
    train_df, test_df = temporal_split(df, time_column="Time", test_size=0.2)
    assert train_df["Time"].max() < test_df["Time"].min()
    assert len(train_df) + len(test_df) == len(df)


def test_temporal_split_respects_test_size():
    df = _sample_df(100)
    train_df, test_df = temporal_split(df, test_size=0.3)
    assert len(test_df) == 30
    assert len(train_df) == 70


def test_random_split_preserves_total_rows():
    df = _sample_df()
    train_df, test_df = random_split(df, test_size=0.25)
    assert len(train_df) + len(test_df) == len(df)


def test_temporal_train_val_test_split_orders_all_three_parts():
    df = _sample_df(200)
    train_df, val_df, test_df = temporal_train_val_test_split(df, test_size=0.2, validation_size=0.2)

    assert len(train_df) + len(val_df) + len(test_df) == len(df)
    assert train_df["Time"].max() < val_df["Time"].min()
    assert val_df["Time"].max() < test_df["Time"].min()


def test_random_train_val_test_split_preserves_total_rows():
    df = _sample_df(200)
    train_df, val_df, test_df = random_train_val_test_split(df, test_size=0.2, validation_size=0.2)

    assert len(train_df) + len(val_df) + len(test_df) == len(df)
    # Amount é único por linha (uniforme contínuo): nenhuma transação se repete entre as partes
    assert set(train_df["Amount"]).isdisjoint(val_df["Amount"])
    assert set(train_df["Amount"]).isdisjoint(test_df["Amount"])
    assert set(val_df["Amount"]).isdisjoint(test_df["Amount"])
