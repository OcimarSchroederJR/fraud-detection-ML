"""Gera as figuras estáticas usadas no README a partir dos resultados
já calculados em results/comparacao_modelos.csv (produzido por
`python -m src.evaluation.run_comparison`).

As figuras usam barras horizontais ordenadas por valor, com o número
escrito direto ao lado de cada barra — pensadas para serem entendidas
em poucos segundos por alguém sem contexto do projeto, não só por
quem já sabe o que é PR-AUC.

Uso:
    python -m src.evaluation.generate_report_charts
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_CSV = PROJECT_ROOT / "results" / "comparacao_modelos.csv"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"

# cores fixas por modelo (paleta categórica: mesma cor em todo gráfico
# do projeto, nunca reatribuída por posição ou ranking)
MODEL_COLORS = {
    "logistic_regression": "#2a78d6",
    "lightgbm": "#eb6834",
    "isolation_forest": "#1baf7a",
}
MODEL_LABELS = {
    "logistic_regression": "Regressão logística",
    "lightgbm": "LightGBM",
    "isolation_forest": "Isolation Forest",
}
BALANCING_LABELS = {
    "class_weight": "ponderação de classe",
    "smote": "SMOTE",
    "treinado só com legítimas": "só transações legítimas",
}
SPLIT_LABELS = {
    "temporal": "Divisão temporal (mais realista)",
    "aleatoria": "Divisão aleatória",
}

INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"


def _load_results(csv_path: Path = RESULTS_CSV) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["rotulo"] = df["model"].map(MODEL_LABELS) + " (" + df["balancing"].map(BALANCING_LABELS) + ")"
    df["cor"] = df["model"].map(MODEL_COLORS)
    return df


def _style_axis(ax):
    ax.set_facecolor(SURFACE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color(INK_MUTED)
    ax.tick_params(colors=INK_SECONDARY, labelsize=10)
    ax.xaxis.grid(True, color=GRID, linewidth=1)
    ax.set_axisbelow(True)


def _legend(fig):
    handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in MODEL_COLORS.values()]
    fig.legend(
        handles,
        MODEL_LABELS.values(),
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
        fontsize=10,
        labelcolor=INK_SECONDARY,
    )


def _horizontal_bars(ax, split_df, value_col, value_fmt, ascending, xlim=None):
    split_df = split_df.sort_values(value_col, ascending=ascending)
    y_pos = range(len(split_df))
    ax.barh(y_pos, split_df[value_col], color=split_df["cor"], height=0.6)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(split_df["rotulo"], color=INK, fontsize=10)

    x_max = xlim[1] if xlim else split_df[value_col].max()
    for y, value in zip(y_pos, split_df[value_col], strict=True):
        ax.text(
            value + x_max * 0.02,
            y,
            value_fmt(value),
            va="center",
            ha="left",
            color=INK,
            fontsize=9.5,
            fontweight="bold",
        )
    if xlim:
        ax.set_xlim(*xlim)
    else:
        ax.set_xlim(0, x_max * 1.2)


def plot_pr_auc(df: pd.DataFrame, output_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), facecolor=SURFACE)
    fig.suptitle(
        "Quão bem cada modelo separa fraude de transação legítima",
        fontsize=13,
        fontweight="bold",
        color=INK,
    )
    fig.text(0.5, 0.90, "PR-AUC, de 0 a 1 — quanto maior, melhor", ha="center", fontsize=10, color=INK_SECONDARY)

    for ax, (split_key, split_label) in zip(axes, SPLIT_LABELS.items(), strict=True):
        split_df = df[df["split"] == split_key]
        _horizontal_bars(ax, split_df, "pr_auc", lambda v: f"{v:.2f}", ascending=True, xlim=(0, 1.05))
        ax.set_title(split_label, fontsize=11, color=INK_SECONDARY, pad=10)
        _style_axis(ax)

    _legend(fig)
    plt.tight_layout(rect=(0, 0.06, 0.98, 0.86))
    plt.savefig(output_path, dpi=150, facecolor=SURFACE)
    plt.close()


def plot_latency(df: pd.DataFrame, output_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), facecolor=SURFACE)
    fig.suptitle(
        "Tempo de resposta ao avaliar uma transação",
        fontsize=13,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.5, 0.90, "Milissegundos por transação — quanto menor, mais rápido", ha="center", fontsize=10, color=INK_SECONDARY
    )

    shared_xlim = (0, df["latency_p95_ms"].max() * 1.4)
    for ax, (split_key, split_label) in zip(axes, SPLIT_LABELS.items(), strict=True):
        split_df = df[df["split"] == split_key]
        _horizontal_bars(ax, split_df, "latency_p95_ms", lambda v: f"{v:.2f} ms", ascending=False, xlim=shared_xlim)
        ax.set_title(split_label, fontsize=11, color=INK_SECONDARY, pad=10)
        ax.set_xlabel("")
        _style_axis(ax)

    _legend(fig)
    plt.tight_layout(rect=(0, 0.06, 0.98, 0.86))
    plt.savefig(output_path, dpi=150, facecolor=SURFACE)
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
