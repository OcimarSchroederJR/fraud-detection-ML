import pandas as pd

from src.evaluation.generate_report_charts import _load_results, plot_latency, plot_pr_auc


def _fake_results_csv(tmp_path):
    df = pd.DataFrame(
        [
            {"split": "temporal", "model": "logistic_regression", "balancing": "class_weight", "pr_auc": 0.74, "latency_p95_ms": 0.25},
            {"split": "temporal", "model": "lightgbm", "balancing": "smote", "pr_auc": 0.79, "latency_p95_ms": 1.20},
            {"split": "temporal", "model": "isolation_forest", "balancing": "treinado só com legítimas", "pr_auc": 0.03, "latency_p95_ms": 5.66},
            {"split": "aleatoria", "model": "logistic_regression", "balancing": "class_weight", "pr_auc": 0.71, "latency_p95_ms": 0.17},
            {"split": "aleatoria", "model": "lightgbm", "balancing": "smote", "pr_auc": 0.86, "latency_p95_ms": 1.53},
            {"split": "aleatoria", "model": "isolation_forest", "balancing": "treinado só com legítimas", "pr_auc": 0.13, "latency_p95_ms": 5.91},
        ]
    )
    path = tmp_path / "comparacao_modelos.csv"
    df.to_csv(path, index=False)
    return path


def test_plot_pr_auc_and_latency_produce_image_files(tmp_path):
    csv_path = _fake_results_csv(tmp_path)
    df = _load_results(csv_path)

    pr_auc_path = tmp_path / "pr_auc.png"
    latency_path = tmp_path / "latencia.png"

    plot_pr_auc(df, pr_auc_path)
    plot_latency(df, latency_path)

    assert pr_auc_path.exists() and pr_auc_path.stat().st_size > 0
    assert latency_path.exists() and latency_path.stat().st_size > 0
