# Detecção de fraude em transações de cartão de crédito com engenharia de pipeline para dados extremamente desbalanceados

> Rascunho estrutural do relatório final (Passo 16 do guia). As seções de Introdução, Revisão de Literatura e Dados e Método já estão preenchidas com o que foi de fato decidido e implementado no projeto. As seções de Resultados, Discussão e Conclusão dependem de rodar o pipeline de treino sobre o dataset real (`data/raw/creditcard.csv`, baixado do Kaggle) e estão marcadas como **TODO** — não devem ser preenchidas com números inventados.

## 1. Introdução

A pergunta de pesquisa deste projeto é: dado o histórico de transações de cartão de crédito, é possível construir um classificador que identifique transações fraudulentas com uma taxa de detecção alta o suficiente para ser útil operacionalmente, mantendo uma taxa de falsos positivos baixa o suficiente para não gerar atrito excessivo com clientes legítimos, e que consiga fazer isso dentro de um orçamento de latência compatível com autorização de transação em tempo real?

Essa formulação contém três pressões concorrentes — recall sobre a classe minoritária, precisão sobre a mesma classe, e tempo de inferência — e não apenas uma métrica única de classificação a maximizar. O restante do relatório trata esse compromisso de três lados como o critério central de avaliação, em vez de ancorar as conclusões em um único número de F1 ou acurácia.

## 2. Revisão de literatura

O próprio grupo que publicou o dataset usado neste projeto (ULB Machine Learning Group, em parceria com a Worldline) também publicou a pesquisa mais diretamente relevante: Dal Pozzolo et al. (2014) discutem lições práticas de detecção de fraude sob a perspectiva de quem opera esses sistemas, incluindo deriva de conceito e o atraso na rotulagem de fraudes confirmadas — pontos retomados na discussão de monitoramento deste projeto (ver [`docs/monitoramento_deriva_conceito.md`](monitoramento_deriva_conceito.md)). Dal Pozzolo et al. (2015) tratam de como calibrar probabilidades de saída quando o treino é feito sobre uma versão balanceada artificialmente da classe majoritária, um problema técnico relevante para a comparação entre as estratégias de balanceamento deste projeto.

Chawla et al. (2002) introduziram o SMOTE, técnica de sobreamostragem sintética da classe minoritária usada como uma das três estratégias de balanceamento comparadas aqui (`src/balancing/strategies.py`). Liu, Ting e Zhou (2008) introduziram o Isolation Forest, tratado neste projeto como uma terceira via de detecção de anomalia, alternativa à classificação supervisionada tradicional (`src/train/models.py`).

## 3. Dados e método

### 3.1 Dados

Dataset Credit Card Fraud Detection (ULB Machine Learning Group / Worldline, Kaggle, licença Database Contents License): 284.807 transações de cartão europeu ao longo de dois dias de setembro de 2013, das quais 492 (≈0,172%) são fraudulentas. 28 das variáveis (`V1`..`V28`) já vêm transformadas por PCA por motivo de confidencialidade; `Time` e `Amount` permanecem em sua forma original. O dataset já vem limpo, sem valores faltantes — uma limitação documentada na Seção 5, já que um dataset de produção real dificilmente chega tão organizado.

### 3.2 Engenharia do pipeline

O repositório separa explicitamente o código de treino (offline) do código de inferência (`src/serving/`), que foi mantido deliberadamente leve — poucas dependências, sem MLflow/Optuna/SHAP — já que é o código que hipoteticamente rodaria em produção sob restrição de latência (ver `Dockerfile`). Cada etapa do pipeline (`ingestion`, `preprocessing`, `balancing`, `train`, `evaluation`, `serving`) é um módulo Python testável isoladamente, com testes automatizados espelhando essa mesma divisão em `tests/` e validação de schema de entrada via `pandera` (`src/preprocessing/schema.py`).

### 3.3 Divisão dos dados

Os dados são divididos tanto temporalmente (transações mais antigas para treino, mais recentes para teste, via `src/preprocessing/split.temporal_split`) quanto aleatoriamente (`random_split`, estratificado pela classe), para permitir comparar as duas abordagens. A divisão temporal reproduz a situação real de um sistema em produção, que sempre prevê sobre transações futuras a partir de um modelo treinado sobre o passado.

### 3.4 Estratégias de balanceamento

Três estratégias são comparadas (`src/balancing/`), cada uma com uma suposição implícita diferente sobre a natureza do problema:

- **Ponderação de classe**: assume que o modelo consegue aprender a fronteira de decisão apenas ajustando o custo do erro, sem alterar os dados.
- **SMOTE**: assume que a vizinhança geométrica de uma fraude no espaço de atributos também representa um padrão de fraude plausível — suposição que nem sempre se sustenta em alta dimensionalidade. Aplicado exclusivamente sobre o conjunto de treino, nunca sobre o teste.
- **Isolation Forest**: trata fraude como detecção de anomalia, não como uma classe com estrutura interna própria, treinando apenas sobre transações legítimas.

### 3.5 Modelos

Regressão logística com ponderação de classe como baseline interpretável, LightGBM como modelo de gradient boosting (padrão de fato da indústria para dados tabulares), e Isolation Forest como terceira via não supervisionada (`src/train/models.py`). Os hiperparâmetros do LightGBM são ajustados via otimização bayesiana com Optuna (`src/train/tune.py`), maximizando PR-AUC por validação cruzada.

### 3.6 Métricas

Acurácia é evitada por ser praticamente inútil sob desbalanceamento extremo. A métrica principal é a área sob a curva de precisão e recall (PR-AUC), complementada por uma métrica de custo esperado por transação (`src/evaluation/metrics.py`), que atribui um custo estimado a falsos negativos (fraude não detectada) e a falsos positivos (transação legítima bloqueada indevidamente), usada para escolher o limiar de decisão que minimiza o custo total — em vez do limiar padrão de 0,5. As premissas de custo usadas (`config/config.yaml`, seção `cost_model`) são estimativas ilustrativas, não valores de mercado verificados, e devem ser declaradas como tal na versão final deste relatório.

### 3.7 Latência

O tempo de inferência é medido transação a transação (`src/evaluation/latency.py`), não em lote, para refletir a experiência real de autorização de cartão, e comparado entre os três modelos.

## 4. Resultados

**TODO** — preencher após rodar `python -m src.train.train_pipeline` sobre `data/raw/creditcard.csv` real. Organizar como uma tabela única cruzando modelo × estratégia de balanceamento × métrica (PR-AUC, precisão e recall no limiar escolhido, custo esperado por transação, latência p95/p99), para as divisões temporal e aleatória.

## 5. Discussão

**TODO** — discutir, com os números da Seção 4: (a) a diferença de desempenho entre a divisão temporal e a aleatória; (b) qual estratégia de balanceamento generalizou melhor e se isso confirma ou contradiz as suposições implícitas listadas na Seção 3.4; (c) os valores de SHAP (`src/evaluation/interpretability.py`) sobre o modelo LightGBM, verificando se `Amount` e `Time` — as únicas variáveis interpretáveis — aparecem entre as mais relevantes; (d) diferença de custo computacional entre os modelos frente à diferença de desempenho preditivo.

Independentemente dos números, os seguintes pontos qualitativos já estão estabelecidos e devem constar nesta seção:

- A anonimização por PCA resolve a questão de privacidade dos dados, mas impede interpretação de negócio direta sobre o que o modelo aprendeu, além do que os valores de SHAP permitem inferir indiretamente.
- A escolha do limiar de decisão (Seção 3.6) não é uma decisão puramente técnica: falsos positivos recaem de forma desigual sobre clientes que dependem mais do uso do cartão no dia a dia, o que é uma decisão de produto com consequências distributivas reais, não apenas um parâmetro a otimizar.

## 6. Limitações

O dataset cobre apenas dois dias de um único mercado europeu em 2013, o que limita fortemente a validade externa de qualquer conclusão sobre padrões atuais de fraude — tanto o comportamento de consumo quanto as táticas de fraude mudam substancialmente ao longo de mais de uma década. A anonimização por componentes principais impede qualquer interpretação de negócio direta além da inferida via SHAP. O monitoramento de deriva de conceito em produção foi discutido mas não implementado, pelas razões detalhadas em [`docs/monitoramento_deriva_conceito.md`](monitoramento_deriva_conceito.md).

## 7. Conclusão

**TODO** — resumir, à luz dos resultados da Seção 4, se a pergunta de pesquisa da Seção 1 foi respondida: existe uma combinação de modelo e estratégia de balanceamento que atinge um recall operacionalmente útil, com falsos positivos e latência dentro de limites aceitáveis? Apontar próximos passos (ex.: implementação do monitoramento de deriva de conceito, validação em um segundo dataset como o IEEE-CIS Fraud Detection).

## Referências

CHAWLA, Nitesh V.; BOWYER, Kevin W.; HALL, Lawrence O.; KEGELMEYER, W. Philip. SMOTE: synthetic minority over-sampling technique. *Journal of Artificial Intelligence Research*, v. 16, 2002.

DAL POZZOLO, Andrea; CAELEN, Olivier; LE BORGNE, Yann-Aël; WATERSCHOOT, Serge; BONTEMPI, Gianluca. Learned lessons in credit card fraud detection from a practitioner perspective. *Expert Systems with Applications*, v. 41, n. 10, 2014.

DAL POZZOLO, Andrea; CAELEN, Olivier; JOHNSON, Reid A.; BONTEMPI, Gianluca. Calibrating probability with undersampling for unbalanced classification. In: *IEEE Symposium on Computational Intelligence and Data Mining*, 2015.

LIU, Fei Tony; TING, Kai Ming; ZHOU, Zhi-Hua. Isolation forest. In: *IEEE International Conference on Data Mining*, 2008.
