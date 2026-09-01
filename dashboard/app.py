"""Dashboard interativo do projeto (Streamlit).

Não faz parte do código de serving (`src/serving/`, que é mantido leve
para produção). Este dashboard é uma camada de demonstração, separada
de propósito: usa dependências mais pesadas (streamlit) que nunca
deveriam rodar atrás de uma rota de autorização de cartão em produção.

Uso:
    streamlit run dashboard/app.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json

import joblib
import pandas as pd
import streamlit as st

from src.features import FEATURE_ORDER, feature_frame
from src.ingestion.load_data import load_config, load_raw_transactions

st.set_page_config(page_title="Detecção de fraude", page_icon="💳", layout="wide")


@st.cache_resource
def get_config():
    return load_config()


@st.cache_resource
def get_model(model_path: str):
    path = Path(model_path)
    if not path.exists():
        return None
    return joblib.load(path)


@st.cache_data
def get_sample_transactions(n: int = 200):
    try:
        df = load_raw_transactions()
    except FileNotFoundError:
        return None
    return df.sample(n=min(n, len(df)), random_state=None)


@st.cache_data
def get_default_threshold(model_dir: str) -> float:
    """Limiar escolhido no treino (custo esperado mínimo na validação),
    lido dos metadados do modelo; 0.5 se não houver metadados."""
    path = Path(model_dir) / "model_metadata.json"
    if not path.exists():
        return 0.5
    try:
        return float(json.loads(path.read_text(encoding="utf-8")).get("decision_threshold", 0.5))
    except (OSError, ValueError):
        return 0.5


@st.cache_data
def get_comparison_results():
    path = PROJECT_ROOT / "results" / "comparacao_modelos.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def _init_feature_state():
    if "features" not in st.session_state:
        st.session_state["features"] = {name: 0.0 for name in FEATURE_ORDER}


def main():
    st.title("💳 Detecção de fraude em transações de cartão")
    st.caption(
        "Demonstração interativa do modelo treinado no projeto "
        "[fraud-detection-ML](https://github.com/OcimarSchroederJR/fraud-detection-ML). "
        "As 28 componentes V1..V28 vêm anonimizadas por PCA no dataset original."
    )

    config = get_config()
    model_path = PROJECT_ROOT / config["paths"]["model_output_dir"] / "model.joblib"
    model = get_model(str(model_path))

    _init_feature_state()

    col_form, col_result = st.columns([2, 1])

    with col_form:
        st.subheader("Transação")

        sample_df = get_sample_transactions()
        if sample_df is not None:
            if st.button("🎲 Carregar transação aleatória do dataset"):
                row = sample_df.sample(n=1).iloc[0]
                for name in FEATURE_ORDER:
                    st.session_state["features"][name] = float(row[name])
                st.session_state["ground_truth"] = int(row["Class"])
        else:
            st.info(
                "Dataset não encontrado em `data/raw/creditcard.csv` — preencha os "
                "campos manualmente ou baixe o dataset para sortear transações reais."
            )

        amount = st.number_input(
            "Amount (valor da transação)", min_value=0.0, value=st.session_state["features"]["Amount"], step=1.0
        )
        time_value = st.number_input(
            "Time (segundos desde a primeira transação da base)",
            min_value=0.0,
            value=st.session_state["features"]["Time"],
            step=1.0,
        )
        st.session_state["features"]["Amount"] = amount
        st.session_state["features"]["Time"] = time_value

        with st.expander("Componentes PCA (V1..V28)"):
            v_cols = st.columns(4)
            for i in range(1, 29):
                name = f"V{i}"
                with v_cols[(i - 1) % 4]:
                    st.session_state["features"][name] = st.number_input(
                        name, value=st.session_state["features"][name], key=f"input_{name}", format="%.4f"
                    )

        if "ground_truth" in st.session_state:
            rotulo = "FRAUDE" if st.session_state["ground_truth"] == 1 else "legítima"
            st.caption(f"Rótulo real desta transação no dataset: **{rotulo}** (só para fins de demonstração).")

    with col_result:
        st.subheader("Previsão")

        if model is None:
            st.warning(
                "Nenhum modelo treinado encontrado em "
                f"`{model_path.relative_to(PROJECT_ROOT)}`. Rode "
                "`python -m src.train.train_pipeline` antes de usar o dashboard."
            )
        else:
            default_threshold = get_default_threshold(str(PROJECT_ROOT / config["paths"]["model_output_dir"]))
            threshold = st.slider(
                "Limiar de decisão", min_value=0.0, max_value=1.0, value=default_threshold, step=0.01
            )
            features = feature_frame(st.session_state["features"])
            fraud_probability = float(model.predict_proba(features)[0][1])

            st.metric("Probabilidade de fraude", f"{fraud_probability:.2%}")
            if fraud_probability >= threshold:
                st.error("Decisão: sinalizar como possível fraude")
            else:
                st.success("Decisão: transação legítima")

    st.divider()
    st.subheader("Resultados do experimento (passo 8 do guia)")

    results_df = get_comparison_results()
    if results_df is None:
        st.info(
            "Nenhum resultado encontrado em `results/comparacao_modelos.csv`. Rode "
            "`python -m src.evaluation.run_comparison` para gerá-lo."
        )
    else:
        results_df = results_df.copy()
        results_df["combinacao"] = results_df["model"] + " / " + results_df["balancing"]

        col_pr, col_lat = st.columns(2)
        with col_pr:
            st.caption("PR-AUC por modelo, balanceamento e divisão")
            pr_pivot = results_df.pivot(index="combinacao", columns="split", values="pr_auc")
            st.bar_chart(pr_pivot)
        with col_lat:
            st.caption("Latência p95 (ms) por modelo, balanceamento e divisão")
            lat_pivot = results_df.pivot(index="combinacao", columns="split", values="latency_p95_ms")
            st.bar_chart(lat_pivot)

        st.dataframe(results_df.drop(columns="combinacao"), use_container_width=True)


if __name__ == "__main__":
    main()
