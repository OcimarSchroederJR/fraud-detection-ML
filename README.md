# fraud-detection-ML

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

## Estrutura do repositório

```
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
tests/            # testes automatizados, espelhando src/
config/           # hiperparâmetros e caminhos
```

## Fonte de dados

Dataset: [Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud) (ULB Machine Learning Group / Worldline), licença Database Contents License.

## Status

Projeto em desenvolvimento inicial.
