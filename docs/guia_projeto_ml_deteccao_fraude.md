# Guia de Projeto Acadêmico de Machine Learning
## Detecção de fraude em transações de cartão de crédito, com engenharia de pipeline para dados extremamente desbalanceados

## Sobre a escolha do tema

Detecção de fraude é um dos casos de uso mais estudados e mais implantados em produção na indústria financeira, e por um motivo simples: o problema é genuinamente difícil de um jeito que um caso didático de sala de aula raramente reproduz. As fraudes representam uma fração ínfima das transações, os custos de errar para os dois lados não são simétricos, o modelo precisa responder em milissegundos, e o adversário se adapta ao longo do tempo. Isso empurra o projeto para decisões de engenharia que uma tarefa de classificação equilibrada nunca exige.

O projeto usa um dataset real, não sintético, publicado pelo grupo de aprendizado de máquina da Université Libre de Bruxelles em parceria com a processadora de pagamentos Worldline, disponível publicamente no Kaggle sob o nome Credit Card Fraud Detection. Ele contém 284807 transações de cartão de crédito realizadas por portadores europeus ao longo de dois dias de setembro de 2013, das quais apenas 492 são fraudulentas, uma proporção de cerca de 0,172% da base inteira. Por motivo de confidencialidade, 28 das variáveis já vêm transformadas por análise de componentes principais e são identificadas apenas como V1 até V28, enquanto duas variáveis permanecem em sua forma original: o tempo decorrido em segundos desde a primeira transação da base e o valor monetário da transação. Essa característica de anonimização já embutida no dataset é, ela mesma, um ponto relevante para a discussão metodológica do projeto, porque limita a engenharia de atributos de domínio ao mesmo tempo em que resolve de saída qualquer questão de privacidade sobre os dados usados.

## Passo 1. Definição do problema e da pergunta de pesquisa

A pergunta central é: dado o histórico de transações, é possível construir um classificador que identifique transações fraudulentas com uma taxa de detecção alta o suficiente para ser útil operacionalmente, mantendo uma taxa de falsos positivos baixa o suficiente para não gerar atrito excessivo com clientes legítimos, e que consiga fazer isso dentro de um orçamo de latência compatível com autorização de transação em tempo real?

Note que essa formulação já contém três pressões concorrentes, não apenas uma métrica de acurácia a maximizar: recall sobre a classe minoritária, precisão sobre a mesma classe, e tempo de inferência. Grande parte do valor do projeto está em mostrar que você entende esse compromisso de três lados, e não apenas em reportar um número de F1 no final.

## Passo 2. Revisão de literatura

O próprio grupo que publicou o dataset também publicou a pesquisa mais diretamente relevante para o seu projeto. Dal Pozzolo, Caelen, Le Borgne, Waterschoot e Bontempi, em artigo de 2014 na revista Expert Systems with Applications, discutem lições práticas de detecção de fraude sob a perspectiva de quem efetivamente opera esses sistemas, incluindo o problema de deriva de conceito, que é a mudança do padrão de comportamento fraudulento ao longo do tempo, e o problema de que a rotulagem de fraudes confirmadas costuma chegar com atraso em relação ao momento da transação, o que complica a validação de qualquer modelo em produção.

Dal Pozzolo, Caelen, Johnson e Bontempi, em artigo de 2015 apresentado no IEEE Symposium on Computational Intelligence and Data Mining, tratam especificamente de como calibrar probabilidades de saída de um classificador quando o treino é feito sobre uma versão subamostrada da classe majoritária, um problema técnico direto que você provavelmente vai encontrar ao aplicar as estratégias de balanceamento do passo 6.

Duas referências adicionais cobrem os fundamentos técnicos que sustentam esse tipo de projeto. Chawla, Bowyer, Hall e Kegelmeyer, no artigo de 2002 no Journal of Artificial Intelligence Research, introduziram a técnica SMOTE de sobreamostragem sintética da classe minoritária, hoje o método de balanceamento mais citado na literatura de aprendizado com classes desbalanceadas. Liu, Ting e Zhou, em artigo de 2008 apresentado na IEEE International Conference on Data Mining, introduziram o Isolation Forest, um algoritmo de detecção de anomalias que serve como abordagem alternativa ao classificador supervisionado tradicional e que vale a pena comparar no seu projeto, já que fraude pode ser tratada tanto como classificação supervisionada quanto como detecção de anomalia não supervisionada.

## Passo 3. Fonte de dados e escopo

Use o dataset Credit Card Fraud Detection do grupo de aprendizado de máquina da ULB, disponível no Kaggle sob licença Database Contents License. Ele já vem limpo, sem valores faltantes, o que é conveniente para o escopo de uma disciplina, mas documente essa limitação no relatório: um dataset de produção real dificilmente chega tão organizado, e parte do trabalho de engenharia de dados que você faria em um cenário real já foi feito por quem publicou a base.

Se quiser um passo adicional de complexidade e tiver tempo disponível no cronograma da disciplina, o dataset IEEE-CIS Fraud Detection, publicado pela Vesta Corporation em uma competição do Kaggle, oferece um cenário bem mais próximo de um caso real de engenharia de dados, com cerca de 590 mil transações, taxa de fraude em torno de 3,5%, e uma combinação de variáveis numéricas e categóricas não anonimizadas, incluindo informações de dispositivo e domínio de email, que exige junção entre duas tabelas distintas e um esforço de limpeza e engenharia de atributos muito mais substancial. Trate essa segunda opção como uma extensão do projeto principal, não como o ponto de partida, porque o volume e a complexidade de limpeza tornam o escopo mais arriscado dentro do prazo de uma disciplina.

## Passo 4. Estrutura do projeto como software

Organize o repositório separando dados brutos, dados processados, notebooks de exploração, código fonte modular dividido em ingestão, pré processamento, balanceamento, treino, avaliação e serving, uma pasta de testes automatizados espelhando essa mesma divisão, e uma pasta de configuração com hiperparâmetros e caminhos definidos fora do código. Fixe as versões das bibliotecas em um arquivo de ambiente desde o primeiro commit e trate cada etapa do pipeline como um script executável de forma isolada, não apenas como células de notebook.

Como o projeto tem uma dimensão de serving em tempo real, que será tratada no passo 11, vale a pena já prever no desenho do repositório uma separação clara entre o código de treino, que roda offline, e o código de inferência, que precisa ser leve e ter o mínimo de dependências possível, já que esse segundo código é o que hipoteticamente rodaria em produção sob restrição de latência.

## Passo 5. Análise exploratória com foco no desbalanceamento

Comece confirmando e visualizando a proporção real entre as duas classes, porque essa proporção vai guiar todas as decisões metodológicas seguintes. Examine a distribuição do valor da transação separadamente para transações fraudulentas e legítimas, já que fraudes costumam se concentrar em faixas de valor específicas, mesmo em um dataset anonimizado como este. Examine também a distribuição da variável de tempo, verificando se a taxa de fraude varia ao longo das duas janelas de vinte e quatro horas cobertas pelo dataset, algo plausível dado que padrões de consumo e de tentativa de fraude tendem a ter um componente horário.

Como as variáveis V1 até V28 são componentes principais sem significado direto, uma matriz de correlação entre elas tende a ser pouco informativa por construção, já que componentes principais são desenhados para serem não correlacionados entre si. Nesse caso, é mais produtivo observar a separação das distribuições de cada componente entre as duas classes, procurando quais componentes mostram maior diferença entre a distribuição de transações fraudulentas e legítimas, o que já antecipa quais variáveis provavelmente vão pesar mais no modelo.

## Passo 6. Estratégias para lidar com o desbalanceamento de classes

Compare pelo menos três abordagens diferentes para o desbalanceamento, porque a comparação entre elas é uma parte central do valor metodológico do projeto. A primeira é ponderação de classe diretamente no treino do modelo, atribuindo peso maior aos erros cometidos sobre a classe minoritária, uma abordagem que não altera os dados e é a mais simples de implementar. A segunda é sobreamostragem sintética via SMOTE, criando exemplos artificiais da classe minoritária no espaço de atributos, aplicada exclusivamente sobre o conjunto de treino para não vazar informação sintética para o conjunto de teste. A terceira é tratar o problema como detecção de anomalia em vez de classificação supervisionada, treinando um Isolation Forest apenas sobre transações legítimas e avaliando sua capacidade de sinalizar as transações fraudulentas como anômalas.

Documente explicitamente, para cada abordagem, o que ela assume implicitamente sobre a natureza do problema. Ponderação de classe assume que o modelo consegue aprender a fronteira de decisão apenas ajustando o custo do erro. SMOTE assume que a vizinhança geométrica de uma fraude no espaço de atributos também representa um padrão de fraude plausível, uma suposição que nem sempre se sustenta em dados de alta dimensionalidade. Detecção de anomalia assume que fraude é fundamentalmente um padrão raro e diferente do comportamento normal, e não uma classe com sua própria estrutura interna, uma suposição que também merece ser questionada explicitamente no relatório.

## Passo 7. Divisão temporal dos dados

Como o dataset preserva a variável de tempo decorrido, use essa informação para fazer uma divisão temporal em vez de uma divisão aleatória: separe as transações mais antigas para treino e as transações mais recentes, dentro da mesma janela de dois dias, para teste. Essa escolha reproduz a situação real de um sistema de detecção de fraude em produção, que sempre prevê sobre transações futuras a partir de um modelo treinado sobre o passado, e evita o vazamento sutil de informação temporal que uma divisão aleatória pode introduzir quando o comportamento de fraude tem qualquer componente de tendência ao longo do tempo. Reporte o desempenho nas duas divisões, temporal e aleatória, e discuta a diferença entre elas no relatório, seguindo a mesma lógica de rigor metodológico que se aplicaria a qualquer dataset com estrutura temporal.

## Passo 8. Modelagem: baseline e modelos avançados

Treine uma regressão logística com ponderação de classe como baseline interpretável. Em seguida, treine um modelo de gradient boosting sobre árvores, como XGBoost ou LightGBM, que é hoje o padrão de fato da indústria para problemas de classificação sobre dados tabulares como este, incluindo a maior parte dos sistemas reais de detecção de fraude usados por processadoras de pagamento. Treine também o Isolation Forest do passo 6 como uma terceira via, não supervisionada, para comparação.

Avalie os três modelos sob as duas estratégias de balanceamento mais promissoras identificadas no passo 6, o que já produz uma matriz de comparação relativamente rica de resultados, e é justamente essa matriz de comparação, não um único número final, que deve ancorar a seção de resultados do relatório.

## Passo 9. Métricas de avaliação apropriadas para classes extremamente desbalanceadas

Acurácia é uma métrica praticamente inútil aqui, já que prever sempre a classe majoritária já produziria uma acurácia superior a 99% sem detectar uma única fraude. Use a área sob a curva de precisão e recall como métrica principal de comparação entre modelos, porque ela é muito mais informativa do que a área sob a curva ROC quando a classe positiva é rara. Reporte também precisão e recall em um ponto de operação específico, ou seja, em um limiar de decisão concreto, e não apenas as curvas agregadas, porque um sistema real de detecção de fraude precisa escolher um limiar único para operar.

Para tornar essa escolha de limiar tangível, construa uma métrica de custo esperado por transação, atribuindo um custo estimado a um falso negativo, o valor médio de uma fraude não detectada, e um custo estimado a um falso positivo, o custo operacional e de relacionamento de bloquear ou sinalizar indevidamente uma transação legítima. Compare o custo total esperado sob diferentes limiares de decisão e escolha o limiar que minimiza esse custo, em vez de usar o limiar padrão de 0,5, e documente explicitamente as premissas usadas para estimar cada um dos dois custos, deixando claro que são estimativas ilustrativas e não valores de mercado verificados.

## Passo 10. Engenharia de experimentos e ajuste de hiperparâmetros

Registre cada execução de treino em uma ferramenta de rastreamento de experimentos como o MLflow, incluindo a estratégia de balanceamento usada, os hiperparâmetros do modelo, e o conjunto completo de métricas do passo 9, não apenas uma métrica isolada. Use uma ferramenta de otimização bayesiana como o Optuna para o ajuste de hiperparâmetros do modelo de gradient boosting, que tem um espaço de hiperparâmetros grande o suficiente para tornar uma busca em grade cara demais computacionalmente dentro do prazo de uma disciplina.

## Passo 11. Considerações de latência e serving em tempo real

Meça o tempo de inferência de cada modelo treinado sobre um lote de transações individuais, simulando a situação de autorização de uma única transação por vez, e não apenas sobre um lote grande de uma vez, já que a experiência de uso real de um sistema de autorização de cartão é transação a transação. Compare esse tempo entre a regressão logística, o modelo de gradient boosting e o Isolation Forest, porque a diferença de custo computacional entre esses modelos é normalmente maior do que a diferença de desempenho preditivo entre eles, um ponto que costuma surpreender quem vem de uma formação puramente de ciência de dados e que é exatamente o tipo de consideração que se espera de um projeto avaliado em engenharia de software.

Como exercício adicional de engenharia, monte um serviço mínimo de inferência expondo o modelo treinado por trás de uma rota de API usando um framework leve como FastAPI, recebendo os atributos de uma transação e devolvendo a probabilidade de fraude e a decisão sob o limiar escolhido no passo 9. Empacote esse serviço com Docker, fixando o ambiente de execução, o que torna a entrega do projeto reproduzível em qualquer máquina sem depender do ambiente local de desenvolvimento.

## Passo 12. Monitoramento e deriva de conceito

Discuta, ainda que sem implementar um sistema de monitoramento completo, como você detectaria deriva de conceito em produção, retomando o ponto levantado por Dal Pozzolo et al. de 2014 na revisão de literatura. Uma abordagem comum é acompanhar a distribuição das variáveis de entrada ao longo do tempo e comparar contra a distribuição observada no conjunto de treino, sinalizando quando essa distribuição se afasta o suficiente para justificar um retreinamento. Ferramentas de código aberto como o Evidently existem especificamente para automatizar esse tipo de comparação e podem ser mencionadas no relatório como uma direção de trabalho futuro, mesmo que a implementação completa fique fora do escopo do prazo da disciplina.

## Passo 13. Testes automatizados do pipeline

Escreva testes automatizados para as funções determinísticas do pipeline, especialmente a etapa de pré processamento e a etapa de cálculo do custo esperado do passo 9. Adicione também uma camada de validação de schema sobre os dados de entrada, verificando por exemplo que o valor da transação nunca é negativo e que todas as colunas esperadas estão presentes antes de qualquer transação ser processada pelo pipeline, usando uma biblioteca de validação de dados como pandera. Essa validação de schema é uma prática de engenharia de dados que se torna especialmente relevante no contexto de um serviço de inferência em produção, onde uma entrada malformada não deveria derrubar o serviço nem produzir uma previsão silenciosamente errada.

## Passo 14. Interpretabilidade

Mesmo com as variáveis V1 até V28 sem significado direto, calcule valores de SHAP sobre o modelo de gradient boosting para identificar quais componentes mais contribuem para a classificação de uma transação como fraudulenta, e observe se o valor da transação e o tempo decorrido, as duas variáveis interpretáveis do dataset, aparecem entre as mais relevantes. Selecione algumas transações classificadas incorretamente, tanto falsos positivos quanto falsos negativos, e examine seus valores de SHAP individualmente, procurando um padrão qualitativo que ajude a explicar por que o modelo errou naqueles casos específicos.

## Passo 15. Limitações e considerações éticas

Declare explicitamente que o dataset cobre apenas dois dias de um único mercado europeu em 2013, o que limita fortemente a validade externa de qualquer conclusão sobre padrões atuais de fraude, já que tanto o comportamento de consumo quanto as táticas de fraude mudam substancialmente ao longo de mais de uma década. Declare também que a anonimização por componentes principais, embora resolva a questão de privacidade dos dados, impede qualquer interpretação de negócio direta sobre o que exatamente o modelo aprendeu, além do que é possível inferir indiretamente pelos valores de SHAP do passo 14.

Discuta o impacto real de falsos positivos sobre clientes legítimos, que passam por bloqueio ou fricção indevida em uma transação válida, um custo que recai de forma desigual sobre quem depende mais do uso do cartão no dia a dia, e argumente por que a escolha do limiar de decisão do passo 9 não é uma decisão puramente técnica, mas uma decisão de produto com consequências distributivas reais que merece envolvimento de áreas de negócio e de atendimento ao cliente, não apenas da equipe de modelagem.

## Passo 16. Redação do relatório final

Siga a estrutura de introdução, revisão de literatura, dados e método, resultados, discussão e conclusão. Na seção de resultados, organize a comparação entre modelos e estratégias de balanceamento em uma tabela única, com todas as métricas do passo 9 lado a lado, em vez de espalhar os números em prosa. Na seção de método, documente a estrutura de engenharia do projeto, incluindo a decisão de separar código de treino e código de inferência do passo 4 e o desenho do serviço de API do passo 11, com o mesmo nível de detalhe reservado normalmente à descrição do modelo.

## Stack tecnológica recomendada

Python continua sendo a escolha natural de linguagem, pela maturidade do ecossistema de bibliotecas de aprendizado de máquina tabular e pela facilidade de expor um modelo treinado como um serviço web leve. Use uma versão recente e estável do Python 3, isolada em um ambiente virtual, com Poetry como gerenciador de dependências se o curso valorizar rigor de empacotamento, ou um simples requirements.txt com versões fixadas caso o prazo não justifique essa camada adicional.

Para manipulação de dados, pandas e numpy. Para os modelos clássicos e as métricas de avaliação, scikit-learn, que já inclui a implementação de regressão logística, Isolation Forest e as curvas de precisão e recall necessárias no passo 9. Para o balanceamento de classes, a biblioteca imbalanced-learn, que implementa SMOTE e diversas variações dele de forma compatível com a interface do scikit-learn. Para o modelo de gradient boosting, XGBoost ou LightGBM, sendo o segundo geralmente mais rápido em datasets tabulares de porte médio como este, o que importa diretamente para a análise de latência do passo 11.

Para rastreamento de experimentos, MLflow. Para busca de hiperparâmetros, Optuna. Para interpretabilidade, a biblioteca shap, compatível diretamente com modelos baseados em árvore como XGBoost e LightGBM através de um estimador otimizado para esse tipo de modelo, o que torna o cálculo dos valores de SHAP consideravelmente mais rápido do que a versão genérica da biblioteca.

Para o serviço de inferência do passo 11, FastAPI, um framework web leve e nativamente assíncrono, adequado a serviços de baixa latência, junto de uvicorn como servidor de aplicação. Para empacotamento do serviço, Docker. Para validação de schema dos dados de entrada, pandera. Para testes automatizados, pytest. Para visualização, matplotlib e seaborn cobrem as necessidades deste projeto sem exigir ferramentas adicionais.

## Lista de verificação final

Confirme os seguintes pontos antes de entregar.

* A pergunta de pesquisa reconhece explicitamente o compromisso entre recall, precisão e latência, não apenas um objetivo único de classificação.
* A revisão de literatura cita Dal Pozzolo et al. 2014 e 2015, além de Chawla et al. 2002 sobre SMOTE e Liu et al. 2008 sobre Isolation Forest.
* Pelo menos três estratégias de balanceamento de classe são comparadas entre si, com as premissas de cada uma declaradas explicitamente.
* Existe uma divisão temporal dos dados além da divisão aleatória padrão, com a diferença de desempenho entre as duas discutida no relatório.
* A avaliação usa área sob a curva de precisão e recall como métrica principal, além de uma métrica de custo esperado por transação com premissas declaradas.
* O tempo de inferência dos modelos é medido e comparado, não apenas o desempenho preditivo.
* Existem testes automatizados para as partes determinísticas do pipeline e validação de schema sobre os dados de entrada.
* A seção de limitações reconhece a validade externa restrita do dataset e a opacidade das variáveis anonimizadas.
* A escolha do limiar de decisão é discutida como uma decisão de produto com consequências distributivas, não apenas como um parâmetro técnico.
* O relatório documenta as decisões de engenharia do projeto, incluindo a separação entre treino e inferência, com o mesmo nível de detalhe que as decisões de modelagem.

## Referências citadas neste guia

CHAWLA, Nitesh V.; BOWYER, Kevin W.; HALL, Lawrence O.; KEGELMEYER, W. Philip. SMOTE: synthetic minority over-sampling technique. Journal of Artificial Intelligence Research, v. 16, 2002.

DAL POZZOLO, Andrea; CAELEN, Olivier; LE BORGNE, Yann-Aël; WATERSCHOOT, Serge; BONTEMPI, Gianluca. Learned lessons in credit card fraud detection from a practitioner perspective. Expert Systems with Applications, v. 41, n. 10, 2014.

DAL POZZOLO, Andrea; CAELEN, Olivier; JOHNSON, Reid A.; BONTEMPI, Gianluca. Calibrating probability with undersampling for unbalanced classification. In: IEEE Symposium on Computational Intelligence and Data Mining, 2015.

LIU, Fei Tony; TING, Kai Ming; ZHOU, Zhi-Hua. Isolation forest. In: IEEE International Conference on Data Mining, 2008.
