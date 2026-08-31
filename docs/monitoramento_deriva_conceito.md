# Monitoramento e deriva de conceito (Passo 12)

Este documento discute como o monitoramento de deriva de conceito seria feito em produção, sem implementar um sistema de monitoramento completo — o próprio guia do projeto trata isso como fora do escopo de uma disciplina, mas exige que a discussão exista.

## O problema

Dal Pozzolo et al. (2014) apontam dois problemas práticos que qualquer sistema de detecção de fraude em produção enfrenta e que este projeto, por rodar sobre uma base histórica estática de dois dias, não é capaz de reproduzir:

1. **Deriva de conceito**: o padrão de comportamento fraudulento muda ao longo do tempo, à medida que fraudadores adaptam suas táticas e o comportamento legítimo dos usuários também evolui. Um modelo treinado sobre dados de um período pode degradar silenciosamente à medida que esse padrão se afasta do que foi visto no treino.
2. **Atraso na rotulagem**: a confirmação de que uma transação era de fato fraude costuma chegar dias ou semanas depois da transação em si (via contestação do cliente ou investigação). Isso significa que, na prática, não é possível calcular recall "em tempo real" sobre produção — o rótulo verdadeiro chega atrasado.

## Abordagem de monitoramento proposta

A abordagem mais comum, e a que seria adotada aqui, é monitorar a **distribuição das variáveis de entrada** ao longo do tempo e compará-la contra a distribuição observada no conjunto de treino, sem depender do rótulo de fraude:

- Para cada variável de entrada (`Amount`, `Time` transformado em hora do dia, e as componentes `V1`..`V28`), calcular periodicamente uma métrica de distância entre a distribuição da janela mais recente de produção e a distribuição do conjunto de treino (por exemplo, Population Stability Index ou distância de Kolmogorov-Smirnov).
- Definir um limiar de alerta para essa distância. Quando ultrapassado, sinalizar a necessidade de investigação e, possivelmente, retreinamento.
- Monitorar também a distribuição dos **scores de saída do modelo** (a probabilidade de fraude prevista), não só das entradas: uma mudança na distribuição dos scores pode indicar deriva mesmo quando nenhuma variável isolada mudou de forma óbvia.
- Quando os rótulos atrasados finalmente chegam, recalcular as métricas do [Passo 9](../guia_projeto_ml_deteccao_fraude.md) (PR-AUC, custo esperado) sobre essa janela retroativa, para validar se a degradação de distribuição realmente correspondeu a uma degradação de desempenho.

## Ferramentas

Ferramentas de código aberto como o [Evidently](https://github.com/evidentlyai/evidently) automatizam exatamente esse tipo de comparação de distribuição entre um conjunto de referência (treino) e um conjunto atual (produção), gerando relatórios de drift por variável. A integração do Evidently ao pipeline deste projeto fica registrada aqui como trabalho futuro, fora do escopo de tempo desta entrega.

## Por que isso não é implementado neste projeto

O dataset usado cobre apenas uma janela de dois dias (setembro de 2013), sem uma segunda janela temporal distante o suficiente para simular deriva de conceito real. Qualquer implementação de monitoramento aqui seria testada contra dados sintéticos ou contra uma re-amostragem artificial da mesma base, o que teria pouco valor demonstrativo. A decisão de projeto foi documentar a abordagem em profundidade (este documento) em vez de implementar um mecanismo que não teria como ser validado com os dados disponíveis.
