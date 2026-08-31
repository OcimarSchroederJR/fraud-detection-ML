"""Gera as figuras estáticas usadas no README a partir dos resultados
já calculados em results/comparacao_modelos.csv (produzido por
`python -m src.evaluation.run_comparison`).

Uso:
    python -m src.evaluation.generate_report_charts
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_CSV = PROJECT_ROOT / "results" / "comparacao_modelos.csv"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"


def _load_results() -> pd.DataFrame:
    df = pd.read_csv(RESULTS_CSV)
    df["combinacao"] = df["model"] + "\n" + df["balancing"]
    return df


def plot_pr_auc(df: pd.DataFrame, output_path: Path):
    pivot = df.pivot(index="combinacao", columns="split", values="pr_auc")
    ax = pivot.plot(kind="bar", figsize=(9, 5), rot=30)
    ax.set_ylabel("PR-AUC")
    ax.set_xlabel("")
    ax.set_title("PR-AUC por modelo, estratégia de balanceamento e divisão dos dados")
    ax.legend(title="Divisão")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_latency(df: pd.DataFrame, output_path: Path):
    pivot = df.pivot(index="combinacao", columns="split", values="latency_p95_ms")
    ax = pivot.plot(kind="bar", figsize=(9, 5), rot=30, logy=True)
    ax.set_ylabel("Latência p95 (ms, escala log)")
    ax.set_xlabel("")
    ax.set_title("Latência de inferência transação a transação, por modelo")
    ax.legend(title="Divisão")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main():
    if not RESULTS_CSV.exists():
        raise FileNotFoundError(
            f"'{RESULTS_CSV}' não encontrado. Rode `python -m src.evaluation.run_comparison` primeiro."
        )
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df = _load_results()
    plot_pr_auc(df, FIGURES_DIR / "pr_auc_comparacao.png")
    plot_latency(df, FIGURES_DIR / "latencia_comparacao.png")
    print(f"Figuras salvas em {FIGURES_DIR}")


if __name__ == "__main__":
    main()
