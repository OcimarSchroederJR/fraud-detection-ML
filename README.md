# fraud-detection-ML

[![Testes](https://github.com/OcimarSchroederJR/fraud-detection-ML/actions/workflows/tests.yml/badge.svg)](https://github.com/OcimarSchroederJR/fraud-detection-ML/actions/workflows/tests.yml)
[![Cobertura](https://img.shields.io/badge/cobertura-79%25-green)](#como-rodar)
[![Lint](https://img.shields.io/badge/lint-ruff-blueviolet)](ruff.toml)

Detecção de fraude em transações de cartão de crédito, com engenharia de pipeline para dados extremamente desbalanceados.

## Sobre o projeto

Projeto acadêmico/pessoal de Machine Learning que usa o dataset público **Credit Card Fraud Detection** (ULB / Worldline, disponível no Kaggle), contendo 284.807 transações de cartões europeus realizadas ao longo de dois dias de setembro de 2013, das quais apenas 492 (≈0,172%) são fraudulentas.

O foco do projeto não é apenas treinar um classificador, mas explorar as decisões de engenharia exigidas por um problema de classe extremamente desbalanceada e por um cenário de inferência em tempo real:

- Comparação de estratégias de balanceamento (ponderação de classe, SMOTE, detecção de anomalia via Isolation Forest).
- Divisão temporal dos dados, além da divisão aleatória padrão.
- Métricas apropriadas para classes raras (PR-AUC, custo esperado por transação) em vez de acurácia.
- Medição de latência de inferência transação a transação.
- Serviço de inferência leve via FastAPI, empacotado com Docker.
- Interpretabilidade via SHAP.

## Resultados

Comparação real entre os 3 modelos e as 2 estratégias de balanceamento, nas divisões temporal e aleatória dos dados (números completos e discussão em [`docs/relatorio_final.md`](docs/relatorio_final.md)):

![PR-AUC por modelo, balanceamento e divisão](results/figures/pr_auc_comparacao.png)

![Latência p95 por modelo](results/figures/latencia_comparacao.png)

O Isolation Forest (detecção de anomalia, sem rótulos de fraude no treino) fica muito atrás dos modelos supervisionados em PR-AUC e é o mais lento dos três — evidência de que, neste dataset, fraude tem estrutura própria que a detecção de anomalia não-supervisionada não captura bem.

## Modelo treinado

Um modelo LightGBM já treinado (PR-AUC 0,81 na divisão temporal) vem versionado em [`models/model.joblib`](models/model.joblib), com as condições exatas do treino documentadas em [`models/model_metadata.json`](models/model_metadata.json) (hiperparâmetros, estratégia de balanceamento, divisão usada, métricas no teste, commit e versões das bibliotecas). Isso permite rodar a API e o dashboard sem precisar baixar o dataset nem treinar nada primeiro — só quem quiser reproduzir o treino ou testar outra estratégia precisa de `data/raw/creditcard.csv`.

## Limiar de decisão

O modelo devolve uma probabilidade; a decisão fraude/legítima depende do limiar aplicado a ela. O limiar não é fixado em 0,5: durante o treino ele é escolhido no **conjunto de validação** (separado do teste) como o valor que minimiza o custo esperado por transação, dadas as premissas de custo em [`config/config.yaml`](config/config.yaml) (`false_negative_cost`, `false_positive_cost`). O valor escolhido é gravado em `models/model_metadata.json` (`decision_threshold`) e é o que a API e o dashboard usam por padrão. Na API, a variável de ambiente `DECISION_THRESHOLD` tem precedência, se definida.

## API de inferência

```bash
uvicorn src.serving.api:app --reload
```

O modelo é carregado no startup. `GET /health` é liveness (processo de pé); `GET /ready` é readiness (modelo carregado). Exemplo de chamada:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"Time":0,"Amount":149.62,"V1":-1.36,"V2":-0.07,"V3":2.54,"V4":1.38,"V5":-0.34,
       "V6":0.46,"V7":0.24,"V8":0.10,"V9":0.36,"V10":0.09,"V11":-0.55,"V12":-0.62,
       "V13":-0.99,"V14":-0.31,"V15":1.47,"V16":-0.47,"V17":0.21,"V18":0.03,"V19":0.40,
       "V20":0.25,"V21":-0.02,"V22":0.28,"V23":-0.11,"V24":0.07,"V25":0.13,"V26":-0.19,
       "V27":0.13,"V28":-0.02}'
```

```json
{"fraud_probability": 0.0123, "is_fraud": false, "threshold": 0.47}
```

## Dashboard interativo

Um dashboard em Streamlit permite testar o modelo já treinado com uma transação real (sorteada do dataset, se disponível) ou preenchida manualmente, além de visualizar a tabela de resultados acima interativamente:

```bash
streamlit run dashboard/app.py
```

## Estrutura do repositório

```
dashboard/        # dashboard interativo (Streamlit), separado do serving de produção
data/
  raw/            # dados brutos (não versionados)
  processed/      # dados processados (não versionados)
notebooks/        # notebooks de exploração
src/
  ingestion/      # carregamento e leitura dos dados
  preprocessing/  # limpeza e transformação
  balancing/      # estratégias de balanceamento de classes
  train/          # treino dos modelos
  evaluation/     # métricas e avaliação
  serving/        # código de inferência (API)
tests/            # testes automatizados, espelhando src/ e dashboard/
config/           # hiperparâmetros e caminhos
results/          # métricas e figuras geradas pelos scripts de avaliação
docs/             # guia do projeto, relatório final e discussões
```

## Fonte de dados

Dataset: [Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud) (ULB Machine Learning Group / Worldline), licença Database Contents License. Necessário apenas para retreinar o modelo ou rodar o notebook de análise exploratória — baixe o CSV e salve-o em `data/raw/creditcard.csv`.

## Como rodar

```bash
python -m venv .venv
source .venv/Scripts/activate  # Windows (Git Bash)
pip install -r requirements.txt

# testes automatizados
pytest

# testes com relatório de cobertura
pytest --cov=src --cov-report=term-missing

# lint
ruff check .

# retreina o modelo do zero (opcional — já vem um treinado em models/, requer data/raw/creditcard.csv)
python -m src.train.train_pipeline --model lightgbm

# idem, ajustando hiperparâmetros do LightGBM com Optuna antes do treino final (passo 10)
python -m src.train.train_pipeline --model lightgbm --tune --n-trials 15

# matriz de comparação entre modelos e estratégias de balanceamento (passo 8)
python -m src.evaluation.run_comparison

# ranking de importância via SHAP (passo 14)
python -m src.evaluation.run_shap_report

# regenera as figuras de comparação usadas neste README
python -m src.evaluation.generate_report_charts

# serviço de inferência
uvicorn src.serving.api:app --reload

# dashboard interativo
streamlit run dashboard/app.py
```

## Status

- [x] Estrutura do projeto (Passo 4)
- [x] Validação de schema e divisão temporal/aleatória (Passos 7 e 13)
- [x] Estratégias de balanceamento: ponderação de classe e SMOTE (Passo 6)
- [x] Modelos: regressão logística, LightGBM e Isolation Forest (Passo 8)
- [x] Métricas para classes desbalanceadas: PR-AUC e custo esperado por transação (Passo 9)
- [x] Busca de hiperparâmetros com Optuna, conectada ao treino final via `--tune` (Passo 10)
- [x] Medição de latência de inferência transação a transação (Passo 11)
- [x] Serviço de inferência (FastAPI) e empacotamento (Docker), build e execução testados de ponta a ponta no CI (Passo 11)
- [x] Notebook de análise exploratória (Passo 5)
- [x] Interpretabilidade via SHAP (Passo 14)
- [x] Discussão de monitoramento e deriva de conceito, sem implementação completa (Passo 12) — [`docs/monitoramento_deriva_conceito.md`](docs/monitoramento_deriva_conceito.md)
- [x] Matriz de comparação entre modelos e estratégias de balanceamento sobre o dataset real (Passo 8) — [`results/comparacao_modelos.csv`](results/comparacao_modelos.csv)
- [x] Relatório final, com resultados reais preenchidos (Passo 16) — [`docs/relatorio_final.md`](docs/relatorio_final.md)
- [x] CI no GitHub Actions rodando a suíte de testes a cada push/PR
- [x] Dashboard interativo em Streamlit — [`dashboard/app.py`](dashboard/app.py)
- [x] Modelo treinado versionado com metadados de reprodutibilidade — [`models/model_metadata.json`](models/model_metadata.json)
- [x] Lint (ruff) e cobertura de testes no CI
