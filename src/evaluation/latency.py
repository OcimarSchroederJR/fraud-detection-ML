"""Medição de latência de inferência (passo 11 do guia).

Mede o tempo de predição transação a transação, e não sobre um lote
grande de uma vez, já que a experiência real de autorização de cartão
é uma transação por vez.
"""

import time

import numpy as np


def measure_single_transaction_latency(model, X, n_repeats: int = 1) -> dict:
    """Mede a latência de model.predict_proba uma linha por vez.

    Retorna um dicionário com média, mediana, p95 e p99, em milissegundos.
    """
    X_values = np.asarray(X)
    latencies_ms = []

    for _ in range(n_repeats):
        for i in range(len(X_values)):
            row = X_values[i : i + 1]
            start = time.perf_counter()
            model.predict_proba(row)
            elapsed_ms = (time.perf_counter() - start) * 1000
            latencies_ms.append(elapsed_ms)

    latencies_ms = np.array(latencies_ms)
    return {
        "mean_ms": float(np.mean(latencies_ms)),
        "median_ms": float(np.median(latencies_ms)),
        "p95_ms": float(np.percentile(latencies_ms, 95)),
        "p99_ms": float(np.percentile(latencies_ms, 99)),
    }


def compare_models_latency(models: dict, X, n_repeats: int = 1) -> dict:
    """Compara a latência de inferência transação a transação entre modelos.

    `models` é um dicionário {nome: modelo_treinado}.
    """
    return {name: measure_single_transaction_latency(model, X, n_repeats) for name, model in models.items()}
