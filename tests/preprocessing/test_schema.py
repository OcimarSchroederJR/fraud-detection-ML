import pandas as pd
import pandera.errors
import pytest

from src.preprocessing.schema import validate_transactions


def _valid_row(**overrides):
    row = {"Time": 0.0, "Amount": 10.0, "Class": 0}
    row.update({f"V{i}": 0.1 for i in range(1, 29)})
    row.update(overrides)
    return row


def test_validate_transactions_accepts_valid_data():
    df = pd.DataFrame([_valid_row()])
    validated = validate_transactions(df)
    assert len(validated) == 1


def test_validate_transactions_rejects_negative_amount():
    df = pd.DataFrame([_valid_row(Amount=-5.0)])
    with pytest.raises(pandera.errors.SchemaError):
        validate_transactions(df)


def test_validate_transactions_rejects_missing_column():
    df = pd.DataFrame([_valid_row()]).drop(columns=["V1"])
    with pytest.raises(pandera.errors.SchemaError):
        validate_transactions(df)
