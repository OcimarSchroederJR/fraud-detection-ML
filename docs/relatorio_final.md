# Detecção de fraude em transações de cartão de crédito com engenharia de pipeline para dados extremamente desbalanceados

> Relatório final (Passo 16 do guia), preenchido com os resultados reais de `python -m src.evaluation.run_comparison` e `python -m src.evaluation.run_shap_report` sobre o dataset completo (`data/raw/creditcard.csv`, baixado do Kaggle). Os dados brutos por trás das tabelas estão em [`results/`](../results/). Os números refletem a versão do experimento com escolha de limiar sobre validação (Seção 3.3) — uma primeira versão que escolhia o limiar sobre o próprio teste foi identificada e corrigida antes deste relatório.

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

Os dados são divididos tanto temporalmente (transações mais antigas para treino, mais recentes para teste) quanto aleatoriamente (estratificado pela classe), para permitir comparar as duas abordagens. A divisão temporal reproduz a situação real de um sistema em produção, que sempre prevê sobre transações futuras a partir de um modelo treinado sobre o passado.

Cada uma dessas divisões é, na verdade, em três partes — treino, validação e teste (`temporal_train_val_test_split` e `random_train_val_test_split` em `src/preprocessing/split.py`) — e não apenas duas. A validação existe exclusivamente para a escolha do limiar de custo mínimo (Seção 3.6): calculá-la sobre o próprio teste vazaria informação da avaliação final para uma decisão de modelo, inflando artificialmente as métricas reportadas. Essa separação foi introduzida depois de uma primeira versão do experimento que cometia exatamente esse vazamento — os números da Seção 4 já refletem a versão corrigida.

### 3.4 Estratégias de balanceamento

Três estratégias são comparadas (`src/balancing/`), cada uma com uma suposição implícita diferente sobre a natureza do problema:

- **Ponderação de classe**: assume que o modelo consegue aprender a fronteira de decisão apenas ajustando o custo do erro, sem alterar os dados.
- **SMOTE**: assume que a vizinhança geométrica de uma fraude no espaço de atributos também representa um padrão de fraude plausível — suposição que nem sempre se sustenta em alta dimensionalidade. Aplicado exclusivamente sobre o conjunto de treino, nunca sobre o teste.
- **Isolation Forest**: trata fraude como detecção de anomalia, não como uma classe com estrutura interna própria, treinando apenas sobre transações legítimas.

### 3.5 Modelos

Regressão logística com ponderação de classe como baseline interpretável, LightGBM como modelo de gradient boosting (padrão de fato da indústria para dados tabulares), e Isolation Forest como terceira via não supervisionada (`src/train/models.py`). Os hiperparâmetros do LightGBM são ajustados via otimização bayesiana com Optuna (`src/train/tune.py`), maximizando PR-AUC por validação cruzada.

### 3.6 Métricas

Acurácia é evitada por ser praticamente inútil sob desbalanceamento extremo. A métrica principal é a área sob a curva de precisão e recall (PR-AUC), complementada por uma métrica de custo esperado por transação (`src/evaluation/metrics.py`), que atribui um custo estimado a falsos negativos (fraude não detectada) e a falsos positivos (transação legítima bloqueada indevidamente). O limiar de decisão que minimiza esse custo — em vez do limiar padrão de 0,5 — é escolhido sobre o conjunto de validação da Seção 3.3, nunca sobre o teste, e só então aplicado ao teste para calcular a precisão, o recall e o custo final reportados na Seção 4. As premissas de custo usadas (`config/config.yaml`, seção `cost_model`: falso negativo = 120, falso positivo = 5) são estimativas ilustrativas, não valores de mercado verificados.

### 3.7 Latência

O tempo de inferência é medido transação a transação (`src/evaluation/latency.py`), não em lote, para refletir a experiência real de autorização de cartão, e comparado entre os três modelos.

## 4. Resultados

Resultados obtidos rodando `python -m src.evaluation.run_comparison` sobre o dataset real (`data/raw/creditcard.csv`, 284.807 transações), com as premissas de custo ilustrativas de `config/config.yaml` (falso negativo = 120, falso positivo = 5). Tabela completa em [`results/comparacao_modelos.csv`](../results/comparacao_modelos.csv). O limiar já reflete o valor que minimiza o custo esperado (Seção 3.6), escolhido na validação; precisão, recall, PR-AUC e custo são todos calculados sobre o teste, nunca visto nessa escolha.

O conjunto de teste tem 56.962 transações em ambas as divisões, com 75 fraudes na divisão temporal (prevalência 0,132%) e 98 na aleatória (prevalência 0,172%) — ou seja, um classificador aleatório teria PR-AUC próximo desses valores de prevalência, não de zero.

| Split | Modelo | Balanceamento | PR-AUC | Limiar | Precisão | Recall | Custo esperado/transação | Latência p95 (ms) | Latência p99 (ms) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Temporal | Regressão logística | ponderação de classe | 0,740 | 0,959 | 0,458 | 0,800 | 0,0378 | 0,25 | 0,47 |
| Temporal | Regressão logística | SMOTE | 0,787 | 0,954 | 0,571 | 0,800 | 0,0356 | 0,19 | 0,31 |
| Temporal | LightGBM | ponderação de classe | 0,798 | 0,0002 | 0,725 | 0,773 | 0,0377 | 1,19 | 2,03 |
| Temporal | LightGBM | SMOTE | 0,792 | 0,0006 | 0,465 | 0,787 | 0,0397 | 1,20 | 1,80 |
| Temporal | Isolation Forest | treinado só com legítimas | 0,031 | 0,109 | 0,000 | 0,000 | 0,1581 | 5,66 | 7,38 |
| Aleatória | Regressão logística | ponderação de classe | 0,708 | 1,000 | 0,648 | 0,847 | 0,0356 | 0,17 | 0,24 |
| Aleatória | Regressão logística | SMOTE | 0,737 | 0,999 | 0,678 | 0,837 | 0,0371 | 0,19 | 0,26 |
| Aleatória | LightGBM | ponderação de classe | 0,861 | 0,00002 | 0,352 | 0,888 | 0,0372 | 1,25 | 2,00 |
| Aleatória | LightGBM | SMOTE | 0,863 | 0,0050 | 0,494 | 0,888 | 0,0310 | 1,53 | 2,41 |
| Aleatória | Isolation Forest | treinado só com legítimas | 0,125 | -0,074 | 0,085 | 0,694 | 0,1272 | 5,91 | 6,38 |

Interpretabilidade (SHAP, LightGBM, split temporal, amostra de 5.000 transações de teste — `results/shap_importance.csv`, via `python -m src.evaluation.run_shap_report`): as 3 variáveis mais importantes por média do valor absoluto de SHAP são `V4` (0,758), `V1` (0,494) e `V8` (0,492). `Amount`, a única variável monetária interpretável, aparece em 9º lugar de 30 (0,284); `Time` aparece em 12º (0,264) — nem entre as mais relevantes, nem irrelevantes. (Esse ranking não depende do limiar, então não muda com a correção metodológica acima.)

## 5. Discussão

**Divisão temporal vs. aleatória**: o LightGBM manteve PR-AUC parecido entre as divisões (0,798 temporal vs. 0,861 aleatória, uma queda de ~7%), mas a regressão logística caiu de forma mais visível na aleatória em relação à temporal para a variante SMOTE (0,787 temporal vs. 0,737 aleatória) — o oposto do que se esperaria se a divisão aleatória fosse sistematicamente mais fácil. Isso sugere que, neste dataset de apenas dois dias, qual fração específica das 492 fraudes cai em cada partição pesa tanto quanto qualquer efeito de deriva temporal real. A divisão temporal continua sendo a escolha metodologicamente correta para simular produção, mas os dados não mostram aqui uma vantagem artificial clara da divisão aleatória.

**Estratégias de balanceamento**: os resultados não mostram mais uma vantagem consistente de uma estratégia sobre a outra. Para a regressão logística, SMOTE superou ponderação de classe nas duas divisões (0,787 vs. 0,740 na temporal; 0,737 vs. 0,708 na aleatória) — um modelo linear parece se beneficiar de ver exemplos sintéticos explícitos da classe minoritária. Para o LightGBM, o resultado é ambíguo: ponderação de classe venceu por pouco na temporal (0,798 vs. 0,792), SMOTE venceu por pouco na aleatória (0,863 vs. 0,861) — diferenças pequenas o suficiente para não sustentar uma recomendação forte de uma estratégia sobre a outra para modelos de árvore neste dataset. O padrão mais robusto é que a regressão logística se beneficia mais de SMOTE do que o LightGBM, consistente com a suposição da Seção 3.4 de que modelos baseados em fronteiras geométricas simples (lineares) tiram mais proveito de rebalancear o espaço de atributos do que modelos que já particionam esse espaço de forma não-linear.

**Isolation Forest como terceira via**: o resultado ficou ainda mais desfavorável ao Isolation Forest depois da correção metodológica. Na divisão temporal, o limiar de custo mínimo escolhido na validação simplesmente não sinalizou nenhuma transação como fraude no teste (precisão e recall iguais a zero) — ou seja, dado o quão fraco é o poder de separação do score de anomalia e as premissas de custo adotadas, o modelo "ótimo" é não alertar ninguém, o que naturalmente iguala o custo esperado ao custo de nunca detectar as 75 fraudes do teste (75 × 120 / 56.962 = 0,158, exatamente o valor da tabela). Isso é um resultado mais honesto — e mais didático — do que o PR-AUC baixo isolado: mostra concretamente que, sob as premissas de custo adotadas, o Isolation Forest não teria utilidade operacional real neste dataset, por pior que fosse calibrado o limiar. Confirma, com ainda mais força do que antes, a suposição da Seção 3.4 de que fraude aqui tem estrutura própria que a detecção de anomalia não-supervisionada não captura.

**Interpretabilidade**: nem `Amount` nem `Time` dominam o ranking de SHAP, mas ambas aparecem em posições medianas (9ª e 12ª de 30) em vez de no fim da lista — sugerindo que o valor da transação e o momento em que ela ocorre carregam algum sinal preditivo, mas o essencial da decisão do modelo vem das componentes anonimizadas (`V4`, `V1`, `V8`, `V14`, `V12`), que não podem ser interpretadas em termos de negócio.

**Latência**: a regressão logística é a mais rápida (~0,17–0,25 ms p95), o LightGBM é 5 a 7x mais lento (~1,2–1,5 ms p95) e o Isolation Forest é o mais lento de todos (~5,7–5,9 ms p95), mais de 20x a latência da regressão logística. Essa diferença de custo computacional é maior do que a diferença de desempenho preditivo entre regressão logística e LightGBM na divisão temporal (0,787 vs. 0,798, uma diferença de PR-AUC pequena), exatamente o ponto que o guia antecipa como contraintuitivo para quem vem de uma formação puramente de ciência de dados: em um cenário de latência crítica, a regressão logística com SMOTE pode ser uma escolha de produto mais defensável do que o LightGBM, apesar do PR-AUC ligeiramente menor.

**Custo esperado por transação**: sob as premissas ilustrativas de custo, o menor custo esperado entre os modelos supervisionados na divisão aleatória foi do LightGBM com SMOTE (0,0310/transação); na temporal, foi da regressão logística com SMOTE (0,0356/transação) — mas todas as oito combinações supervisionadas ficam num intervalo estreito (0,031 a 0,040), muito menor que o custo do Isolation Forest (0,127–0,158, de 3 a 5x pior). Isso reforça que, entre os modelos supervisionados, a escolha de balanceamento pesa pouco no custo final; a decisão que realmente importa é usar ou não um classificador supervisionado.

Independentemente dos números, os seguintes pontos qualitativos permanecem válidos:

- A anonimização por PCA resolve a questão de privacidade dos dados, mas impede interpretação de negócio direta sobre o que o modelo aprendeu, além do que os valores de SHAP permitem inferir indiretamente — e mesmo esses valores apontam para componentes sem significado de negócio conhecido.
- A escolha do limiar de decisão (Seção 3.6) não é uma decisão puramente técnica: falsos positivos recaem de forma desigual sobre clientes que dependem mais do uso do cartão no dia a dia, o que é uma decisão de produto com consequências distributivas reais, não apenas um parâmetro a otimizar. Os limiares ótimos encontrados variam por ordens de grandeza entre modelos (de 0,00002 a 1,0), o que por si só mostra por que usar cegamente 0,5 seria inadequado — e por que essa escolha precisa ser feita com uma validação separada do teste, como discutido na Seção 3.3.

## 6. Limitações

O dataset cobre apenas dois dias de um único mercado europeu em 2013, o que limita fortemente a validade externa de qualquer conclusão sobre padrões atuais de fraude — tanto o comportamento de consumo quanto as táticas de fraude mudam substancialmente ao longo de mais de uma década. A anonimização por componentes principais impede qualquer interpretação de negócio direta além da inferida via SHAP. O monitoramento de deriva de conceito em produção foi discutido mas não implementado, pelas razões detalhadas em [`docs/monitoramento_deriva_conceito.md`](monitoramento_deriva_conceito.md).

## 7. Conclusão

A pergunta de pesquisa da Seção 1 pede um classificador que equilibre recall alto, precisão suficiente para não gerar atrito excessivo, e latência compatível com autorização em tempo real. Nenhuma combinação testada maximiza as três dimensões simultaneamente, mas os resultados apontam dois candidatos plausíveis, dependendo de qual restrição pesa mais:

- Se o orçamento de latência é apertado (sub-milissegundo), a **regressão logística com SMOTE** entrega PR-AUC 0,787 na divisão temporal com recall de 0,80 e latência p95 de ~0,19 ms — o melhor equilíbrio entre desempenho e velocidade encontrado.
- Se uma latência de ~1,2 ms é aceitável, o **LightGBM** entrega o melhor PR-AUC (0,798–0,863 dependendo da divisão e do balanceamento) e, na divisão aleatória, também o menor custo esperado por transação sob as premissas ilustrativas adotadas.

O Isolation Forest, apesar de não exigir rótulos de fraude no treino, ficou muito atrás dos modelos supervisionados em todas as métricas — na divisão temporal, o limiar de custo mínimo escolhido de forma honesta (sobre validação, não sobre teste) resultou em recall zero, ou seja, o modelo não teria utilidade operacional alguma sob as premissas de custo adotadas. A suposição de que fraude é só "anomalia" não se sustentou empiricamente aqui; o Isolation Forest não é recomendado como abordagem principal para este dataset, embora continue relevante como um sinal complementar de baixo custo de manutenção em cenários onde rótulos de fraude não estão disponíveis.

Um resultado inesperado desta rodada de experimentos foi que a divisão temporal não se mostrou uniformemente mais difícil que a aleatória — para a regressão logística ela foi, na verdade, mais fácil. Isso indica que, nesta janela de apenas dois dias, o resultado é sensível a qual subconjunto específico das 492 fraudes cai em cada partição, mais do que a um efeito sistemático de deriva de conceito. Isso não invalida a divisão temporal como prática recomendada — ela continua sendo a que reproduz corretamente o cenário de produção — mas é um lembrete de que, com poucos exemplos positivos, a variância entre partições pode ser tão grande quanto o efeito que se está tentando medir.

Próximos passos naturais, fora do escopo desta entrega: (a) implementar de fato o monitoramento de deriva de conceito discutido em [`docs/monitoramento_deriva_conceito.md`](monitoramento_deriva_conceito.md), inclusive com o Evidently; (b) validar as mesmas conclusões metodológicas (divisão temporal, comparação de balanceamento, custo por transação) no dataset IEEE-CIS Fraud Detection, mais próximo de um cenário real de engenharia de dados; (c) explorar um ensemble ou uma etapa de recalibração de probabilidade (Dal Pozzolo et al., 2015) para reduzir a lacuna entre o modelo treinado sob balanceamento artificial e o comportamento real em produção.

## Referências

CHAWLA, Nitesh V.; BOWYER, Kevin W.; HALL, Lawrence O.; KEGELMEYER, W. Philip. SMOTE: synthetic minority over-sampling technique. *Journal of Artificial Intelligence Research*, v. 16, 2002.

DAL POZZOLO, Andrea; CAELEN, Olivier; LE BORGNE, Yann-Aël; WATERSCHOOT, Serge; BONTEMPI, Gianluca. Learned lessons in credit card fraud detection from a practitioner perspective. *Expert Systems with Applications*, v. 41, n. 10, 2014.

DAL POZZOLO, Andrea; CAELEN, Olivier; JOHNSON, Reid A.; BONTEMPI, Gianluca. Calibrating probability with undersampling for unbalanced classification. In: *IEEE Symposium on Computational Intelligence and Data Mining*, 2015.

LIU, Fei Tony; TING, Kai Ming; ZHOU, Zhi-Hua. Isolation forest. In: *IEEE International Conference on Data Mining*, 2008.
