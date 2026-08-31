"""Medição de latência de inferência (passo 11 do guia).

Mede o tempo de predição transação a transação, e não sobre um lote
grande de uma vez, já que a experiência real de autorização de cartão
é uma transação por vez.
"""

import time

import numpy as np


def measure_single_transaction_latency(model, X, n_repeats: int = 1, predict_fn=None) -> dict:
    """Mede a latência de inferência uma linha por vez.

    `predict_fn` recebe (model, row) e por padrão chama
    `model.predict_proba(row)`. Modelos sem predict_proba (como
    IsolationForest) podem passar uma função própria, por exemplo
    `lambda m, row: m.decision_function(row)`.

    Retorna um dicionário com média, mediana, p95 e p99, em milissegundos.
    """
    if predict_fn is None:
        predict_fn = lambda m, row: m.predict_proba(row)

    X_values = np.asarray(X)
    latencies_ms = []

    for _ in range(n_repeats):
        for i in range(len(X_values)):
            row = X_values[i : i + 1]
            start = time.perf_counter()
            predict_fn(model, row)
            elapsed_ms = (time.perf_counter() - start) * 1000
            latencies_ms.append(elapsed_ms)

    latencies_ms = np.array(latencies_ms)
    return {
        "mean_ms": float(np.mean(latencies_ms)),
        "median_ms": float(np.median(latencies_ms)),
        "p95_ms": float(np.percentile(latencies_ms, 95)),
        "p99_ms": float(np.percentile(latencies_ms, 99)),
    }


def compare_models_latency(models: dict, X, n_repeats: int = 1, predict_fns: dict = None) -> dict:
    """Compara a latência de inferência transação a transação entre modelos.

    `models` é um dicionário {nome: modelo_treinado}. `predict_fns`,
    opcional, é um dicionário {nome: função} para modelos que não
    expõem predict_proba (ver measure_single_transaction_latency).
    """
    predict_fns = predict_fns or {}
    return {
        name: measure_single_transaction_latency(model, X, n_repeats, predict_fns.get(name))
        for name, model in models.items()
    }
