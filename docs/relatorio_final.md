# Detecção de fraude em transações de cartão de crédito com engenharia de pipeline para dados extremamente desbalanceados

> Relatório final (Passo 16 do guia), preenchido com os resultados reais de `python -m src.evaluation.run_comparison` e `python -m src.evaluation.run_shap_report` sobre o dataset completo (`data/raw/creditcard.csv`, baixado do Kaggle). Os dados brutos por trás das tabelas estão em [`results/`](../results/).

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

Resultados obtidos rodando `python -m src.evaluation.run_comparison` sobre o dataset real (`data/raw/creditcard.csv`, 284.807 transações), com as premissas de custo ilustrativas de `config/config.yaml` (falso negativo = 120, falso positivo = 5). Tabela completa em [`results/comparacao_modelos.csv`](../results/comparacao_modelos.csv); cada célula de limiar já reflete o limiar que minimiza o custo esperado (Seção 3.6), não o padrão de 0,5.

O conjunto de teste tem 56.962 transações em ambas as divisões, com 75 fraudes na divisão temporal (prevalência 0,132%) e 98 na aleatória (prevalência 0,172%) — ou seja, um classificador aleatório teria PR-AUC próximo desses valores de prevalência, não de zero.

| Split | Modelo | Balanceamento | PR-AUC | Limiar | Precisão | Recall | Custo esperado/transação | Latência p95 (ms) | Latência p99 (ms) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Temporal | Regressão logística | ponderação de classe | 0,737 | 0,919 | 0,443 | 0,827 | 0,0342 | 0,14 | 0,16 |
| Temporal | Regressão logística | SMOTE | 0,806 | 0,973 | 0,714 | 0,800 | 0,0337 | 0,13 | 0,15 |
| Temporal | LightGBM | ponderação de classe | 0,809 | 0,0007 | 0,549 | 0,827 | 0,0319 | 1,28 | 1,35 |
| Temporal | LightGBM | SMOTE | 0,802 | 0,777 | 0,879 | 0,773 | 0,0365 | 1,48 | 2,25 |
| Temporal | Isolation Forest | treinado só com legítimas | 0,034 | 0,510 | 0,062 | 0,453 | 0,1317 | 4,39 | 4,94 |
| Aleatória | Regressão logística | ponderação de classe | 0,726 | 0,982 | 0,475 | 0,888 | 0,0316 | 0,14 | 0,16 |
| Aleatória | Regressão logística | SMOTE | 0,730 | 0,954 | 0,512 | 0,878 | 0,0325 | 0,13 | 0,14 |
| Aleatória | LightGBM | ponderação de classe | 0,858 | 0,0018 | 0,726 | 0,867 | 0,0302 | 1,27 | 1,30 |
| Aleatória | LightGBM | SMOTE | 0,843 | 0,108 | 0,694 | 0,857 | 0,0327 | 1,50 | 2,21 |
| Aleatória | Isolation Forest | treinado só com legítimas | 0,128 | 0,543 | 0,092 | 0,694 | 0,1220 | 4,09 | 4,81 |

Interpretabilidade (SHAP, LightGBM, split temporal, amostra de 5.000 transações de teste — `results/shap_importance.csv`, via `python -m src.evaluation.run_shap_report`): as 3 variáveis mais importantes por média do valor absoluto de SHAP são `V4` (0,758), `V1` (0,494) e `V8` (0,492). `Amount`, a única variável monetária interpretável, aparece em 9º lugar de 30 (0,284); `Time` aparece em 12º (0,264) — nem entre as mais relevantes, nem irrelevantes.

## 5. Discussão

**Divisão temporal vs. aleatória**: o LightGBM com ponderação de classe caiu de PR-AUC 0,858 (aleatória) para 0,809 (temporal), uma queda de ~6%. A regressão logística com SMOTE caiu de 0,730 para 0,806 — na verdade *subiu* na divisão temporal, o que sugere que o efeito de qual fração específica das 492 fraudes cai no teste pesa mais do que um efeito sistemático de deriva dentro dessa janela de só dois dias. Ainda assim, o padrão dominante (LightGBM, os dois melhores resultados de PR-AUC do experimento) é de queda sob a divisão temporal, consistente com a expectativa do guia de que a divisão aleatória tende a ser artificialmente otimista.

**Estratégias de balanceamento**: LightGBM com ponderação de classe teve o melhor PR-AUC nas duas divisões (0,809 e 0,858), superando SMOTE (0,802 e 0,843) — uma diferença pequena, mas consistente nas duas divisões. Isso é compatível com a suposição da Seção 3.4 de que a ponderação de classe, por não alterar a geometria dos dados, generaliza de forma pelo menos tão robusta quanto o SMOTE neste dataset. Já para a regressão logística, SMOTE venceu com folga na divisão temporal (0,806 vs. 0,737) — um modelo linear parece se beneficiar mais de ver exemplos sintéticos explícitos da classe minoritária do que de um ajuste de custo. Não há uma estratégia que domine para os dois modelos, o que já é, em si, um resultado relevante: a escolha de balanceamento não pode ser feita independentemente da escolha de modelo.

**Isolation Forest como terceira via**: PR-AUC de 0,034 (temporal) e 0,128 (aleatória) — muito acima do nível de um classificador aleatório (≈0,0013 e ≈0,0017, a prevalência real de cada conjunto de teste), mas seis a oito vezes pior que qualquer combinação supervisionada. Isso confirma, com dados, a suposição levantada na Seção 3.4: fraude neste dataset não é bem descrita apenas como "um padrão raro e diferente do normal" — ela tem estrutura interna própria que um classificador supervisionado consegue aprender e a detecção de anomalia não-supervisionada não captura tão bem. Vale notar que o Isolation Forest teve o recall mais baixo entre os modelos na divisão temporal (0,453) combinado com a pior precisão (0,062), o pior dos dois mundos.

**Interpretabilidade**: nem `Amount` nem `Time` dominam o ranking de SHAP, mas ambas aparecem em posições medianas (9ª e 12ª de 30) em vez de no fim da lista — sugerindo que o valor da transação e o momento em que ela ocorre carregam algum sinal preditivo, mas o essencial da decisão do modelo vem das componentes anonimizadas (`V4`, `V1`, `V8`, `V14`, `V12`), que não podem ser interpretadas em termos de negócio.

**Latência**: a regressão logística é a mais rápida (~0,13–0,14 ms p95), o LightGBM é cerca de 10x mais lento (~1,3–1,5 ms p95) e o Isolation Forest é o mais lento de todos (~4,1–4,4 ms p95), mais de 30x a latência da regressão logística. Essa diferença de custo computacional é maior do que a diferença de desempenho preditivo entre regressão logística e LightGBM (poucos pontos de PR-AUC), exatamente o ponto que o guia antecipa como contraintuitivo para quem vem de uma formação puramente de ciência de dados: em um cenário de latência crítica, a regressão logística com SMOTE (PR-AUC 0,806 na divisão temporal, a 0,13 ms) pode ser uma escolha de produto mais defensável do que o LightGBM (PR-AUC 0,809, a 1,28 ms), apesar do PR-AUC quase idêntico.

**Custo esperado por transação**: sob as premissas ilustrativas de custo (falso negativo = 120, falso positivo = 5), o menor custo esperado entre os modelos supervisionados foi do LightGBM com ponderação de classe na divisão aleatória (0,0302/transação) e na temporal (0,0319/transação) — mas as diferenças entre as quatro combinações supervisionadas são pequenas (0,030 a 0,037), muito menores que a diferença de custo do Isolation Forest (0,122–0,132, cerca de 4x pior). Isso reforça que, entre os modelos supervisionados, a escolha de balanceamento pesa menos no resultado final do que a escolha entre usar ou não um classificador supervisionado.

Independentemente dos números, os seguintes pontos qualitativos permanecem válidos:

- A anonimização por PCA resolve a questão de privacidade dos dados, mas impede interpretação de negócio direta sobre o que o modelo aprendeu, além do que os valores de SHAP permitem inferir indiretamente — e mesmo esses valores apontam para componentes sem significado de negócio conhecido.
- A escolha do limiar de decisão (Seção 3.6) não é uma decisão puramente técnica: falsos positivos recaem de forma desigual sobre clientes que dependem mais do uso do cartão no dia a dia, o que é uma decisão de produto com consequências distributivas reais, não apenas um parâmetro a otimizar. Os limiares ótimos encontrados variam bastante entre modelos (de 0,0007 a 0,98), o que por si só mostra por que usar cegamente 0,5 seria inadequado.

## 6. Limitações

O dataset cobre apenas dois dias de um único mercado europeu em 2013, o que limita fortemente a validade externa de qualquer conclusão sobre padrões atuais de fraude — tanto o comportamento de consumo quanto as táticas de fraude mudam substancialmente ao longo de mais de uma década. A anonimização por componentes principais impede qualquer interpretação de negócio direta além da inferida via SHAP. O monitoramento de deriva de conceito em produção foi discutido mas não implementado, pelas razões detalhadas em [`docs/monitoramento_deriva_conceito.md`](monitoramento_deriva_conceito.md).

## 7. Conclusão

A pergunta de pesquisa da Seção 1 pede um classificador que equilibre recall alto, precisão suficiente para não gerar atrito excessivo, e latência compatível com autorização em tempo real. Nenhuma combinação testada maximiza as três dimensões simultaneamente, mas os resultados apontam dois candidatos plausíveis, dependendo de qual restrição pesa mais:

- Se o orçamento de latência é apertado (ex.: sub-milissegundo), a **regressão logística com SMOTE** entrega PR-AUC 0,806 na divisão temporal com recall de 0,80 e latência p95 de 0,13 ms — o melhor equilíbrio entre desempenho e velocidade encontrado.
- Se a latência de ~1,3 ms é aceitável, o **LightGBM com ponderação de classe** entrega o melhor PR-AUC (0,809 na divisão temporal, 0,858 na aleatória) e o menor custo esperado por transação sob as premissas ilustrativas adotadas.

O Isolation Forest, apesar de não exigir rótulos de fraude no treino, ficou muito atrás dos modelos supervisionados em todas as métricas, e não é recomendado como abordagem principal para este dataset — a suposição de que fraude é só "anomalia" não se sustentou empiricamente aqui, embora o modelo continue relevante como um sinal complementar de baixo custo de manutenção (não precisa de rótulos atualizados).

Como a divisão temporal produziu resultados sistematicamente piores ou iguais aos da divisão aleatória para o melhor modelo (LightGBM), é razoável esperar alguma degradação adicional em produção real, onde a diferença temporal entre treino e inferência é muito maior do que os poucos minutos ou horas cobertos pela divisão de teste deste dataset de dois dias.

Próximos passos naturais, fora do escopo desta entrega: (a) implementar de fato o monitoramento de deriva de conceito discutido em [`docs/monitoramento_deriva_conceito.md`](monitoramento_deriva_conceito.md), inclusive com o Evidently; (b) validar as mesmas conclusões metodológicas (divisão temporal, comparação de balanceamento, custo por transação) no dataset IEEE-CIS Fraud Detection, mais próximo de um cenário real de engenharia de dados; (c) explorar um ensemble ou uma etapa de recalibração de probabilidade (Dal Pozzolo et al., 2015) para reduzir a lacuna entre o modelo treinado sob balanceamento artificial e o comportamento real em produção.

## Referências

CHAWLA, Nitesh V.; BOWYER, Kevin W.; HALL, Lawrence O.; KEGELMEYER, W. Philip. SMOTE: synthetic minority over-sampling technique. *Journal of Artificial Intelligence Research*, v. 16, 2002.

DAL POZZOLO, Andrea; CAELEN, Olivier; LE BORGNE, Yann-Aël; WATERSCHOOT, Serge; BONTEMPI, Gianluca. Learned lessons in credit card fraud detection from a practitioner perspective. *Expert Systems with Applications*, v. 41, n. 10, 2014.

DAL POZZOLO, Andrea; CAELEN, Olivier; JOHNSON, Reid A.; BONTEMPI, Gianluca. Calibrating probability with undersampling for unbalanced classification. In: *IEEE Symposium on Computational Intelligence and Data Mining*, 2015.

LIU, Fei Tony; TING, Kai Ming; ZHOU, Zhi-Hua. Isolation forest. In: *IEEE International Conference on Data Mining*, 2008.
