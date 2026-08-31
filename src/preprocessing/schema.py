"""Validação de schema dos dados de entrada com pandera.

Garante, antes de qualquer transação ser processada pelo pipeline, que
todas as colunas esperadas estão presentes e que o valor da transação
nunca é negativo.
"""

from pandera import Check, Column, DataFrameSchema

_V_COLUMNS = {f"V{i}": Column(float, nullable=False) for i in range(1, 29)}

transaction_schema = DataFrameSchema(
    {
        "Time": Column(float, Check.ge(0), nullable=False),
        **_V_COLUMNS,
        "Amount": Column(float, Check.ge(0), nullable=False),
        "Class": Column(int, Check.isin([0, 1]), nullable=False),
    },
    strict=False,
)


def validate_transactions(df):
    """Valida um DataFrame de transações contra o schema esperado.

    Lança pandera.errors.SchemaError se alguma verificação falhar.
    """
    return transaction_schema.validate(df)
