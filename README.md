# aed_projeto_4

## Projeto do Módulo de NLP

<p align='justify'>No projeto do módulo de NLP vamos tentar resolver um problema de classificação de textos. Você escolherá um de três datasets cujo objetivo é classificar em um conjunto de classes. Você deverá utilizar três metodologias ensinadas em nossas aulas para poder classificar esses textos. A seguir, mais detalhes do projeto para que você possa resolver.</p>

<p align='justify'> Os produtos finais deste projeto será um Notebook Python. As seções seguintes detalham como você deve proceder para gerar o código e a última seção deve especificar a estrutura que seu notebook deve seguir.</p>

## Dados

<p align='justify'>Você poderá escolher três tarefas para resolver no projeto. A seguir existe a breve descrição de cada tarefa e um link para onde você poderá baixar os dados.</p>

- <p align='justify'> O <b>corpus UTL</b> é um corpus com críticas de filmes e apps coletadas automaticamente de sites. As classes são: positiva ou negativa. Assim o usuário pode ter gostado ou não gostado do produto. Referência: https://github.com/RogerFig/UTLCorpus.</p>

- <p align='justify'>O <b>corpus UOL AES-PT</b> é um corpus de redações no estilo do ENEM. Cada redação possui um tópico e um conjunto de redações relacionadas. Nesse corpus, existem vários tópicos e suas respectivas redações. O objetivo é predizer a nota final de cada redação de acordo com o grupo de notas 0, 200, 400, 600, 800 e 1000. Para mais informações e download dos dados, acesse o link: https://github.com/evelinamorim/aes-pt.</p>

- <p align='justify'>O <b>corpus TweetSentBr</b> é um corpus em português de tweets. Cada tweet está rotulado com uma das classes: positivo, negativo e neutro. Para mais informações e download do corpus, acesse o link https://bitbucket.org/HBrum/tweetsentbr/src/master/.</p>

# Representação

<p align='justify'>Vimos durante a nossa aula diversas forma de representar um documento de texto. Você vai usar cada uma dessas representações e compará-las. A seguir temos a listagem das representações que devem ser usadas para representar seu texto.</p>

- <p align='justify'><b>Representação TF-IDF:</b> Você pode usar tanto o gensim quanto o scikit para montar esta representação, mas lembre-se que é importante fazer o pré-processamento dos textos.</p>

- <p align='justify'><b>Representação com o word2vec:</b> O modelo poderá ser o apresentado na aula 03 ou algum outro modelo pré-treinado como os existentes no repositório http://nilc.icmc.usp.br/nilc/index.php/repositorio-de-word-embeddings-do-nilc . Neste caso, cada documento deverá ser representado pelo vetor que resultar da média dos vetores de todas as palavras que o compõem. Em outras palavras, se D é composto pelas palavras w1, w2, …, wn, e seus vetores embeddings são v1, v2, …, vn, então a representação do documento de D será v = (v1 + v2 + … + vn) / n.</p>

- <p align='justify'><b>Extração de features do texto:</b> Você deve pensar em ao menos 10 features para extrair do documento e que o possam representar. Aqui vão algumas sugestões: número de palavras, número de verbos, número de conjunções, número de palavras negativas, número de palavras fora do vocabulário, quantidades de entidades do tipo PESSOA, quantidade de entidades do tipo LOCAL, etc.</p>

<p align='justify'>Lembrando que você deve dividir seu conjunto em treino e teste. No TF-IDF, você só pode aplicar o método fit no conjunto de treino. Uma sugestão é dividir 80% do conjunto de dados para treino e 20% para teste. Essa divisão é aleatória, mas você pode usar o método train_test_split para essa divisão. O exemplo a seguir mostra como usar esse método:</p>

            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
            ...    X, y, test_size=0.20, random_state=42)

### Visualização dos dados

<p align='justify'>Também vimos que embora o nosso texto apresente dimensionalidade maior que 2D, é possível visualizar em apenas duas dimensões usando técnicas de redução de dimensionalidade. Vimos duas técnicas de redução de dimensionalidade, o PCA e o t-SNE. Assim, pede-se que você utilize as duas técnicas para gerar uma visualização dos seus dados e considere as classes para colorir as instâncias.</p>

<p align='justify'>Sugere-se utilizar a biblioteca yellowbrick para gerar as visualizações, devido sua simplicidade. Mas caso tenha interesse em gerar visualizações mais interativas e mais bonitas, você pode utilizar a biblioteca seaborn. Para uma galeria dos gráficos que o seaborn é capaz de fazer, acesse o link https://seaborn.pydata.org/examples/index.html . Apenas acrescentando em seu código import seaborn as sns; sns.set() , também é possível deixar o gráfico com cores mais bonitas. Todas essas bibliotecas precisam do matplotlib, que já está importado no exemplo da aula.</p>

<p align='justify'>Aqui você deve fazer a visualização apenas do seu conjunto de treino.</p>

### Classificadores

<p align='justify'>Escolha dois classificadores que você possua mais familiaridade no scikit-learn para poder classificar os seus dados. Você deve executar cada um dos classificadores nas três representações escolhidas. Você pode usar o k-nn como um dos métodos. Outros métodos estão disponíveis no scikit, como por exemplo o SVM e o RandomForest.</p>

### Métricas de avaliação

<p align='justify'>Para os corpus TweetSentBR e UTL, pede-se que se use a matriz de confusão, a precisão, o recall e o f-1 para reportar a acurácia dos seus classificadores. No caso do corpus UOL AES-PT pede-se que se use o erro médio apenas.</p>


### Estrutura do Notebook

<p align='justify'>O seu notebook deve ser dividido por seções que possuam  uma célula do tipo Markdown. Nesta célula deve ter o título da seção antecedida por um marcador do tipo #. O título de cada seção deverá ser como a lista abaixo. Além do título, é possível que a seção demande a descrição de resultados ou outro tipo de texto. Nestes casos, coloque o texto junto à célula do título. Se houver código solicitado para a seção, então as células restantes devem ser de código solicitado.  
O relatório deve ser organizado nas seguintes seções:</p>

- <p align='justify'><b>Tarefa e Dados:</b> Descreva a tarefa escolhida e os dados. Escreva código que leia os dados e calcule e imprima quantas instâncias os dados têm. Também, seu código deve calcular a média de tokens por instância, isto é, quantos tokens, na média cada documento do seu conjunto de dados possui. Imprima esse único número.

- <p align='justify'><b>Visualização dos dados:</b> Coloque nesta seção os gráficos do PCA e do t-SNE, para cada representação. Responda também às seguintes perguntas: a) Existe algum padrão com relação às classes? b) Caso exista algum padrão, você pode concluir alguma coisa? c) Caso não exista, você consegue dizer se isso tem a ver com alguma representação ou classe?</p>

- <p align='justify'><b>Classificadores:</b> Descreva sucintamente os dois classificadores escolhidos. Você usou algum parâmetro que não seja padrão? Se sim, mencione nesta seção.</p>

- <p align='justify'><b>Resultados:</b> Escreva código que execute a validação cruzada em 5-folds para os dois classificadores escolhidos. Também responda às seguintes perguntas: Os embeddings realmente mostraram um resultado melhor que o TF-IDF? Se não, qual foi a representação que teve o melhor desempenho? A diferença foi muito grande?</p>

- <p align='justify'><b>Conclusão:</b> Por fim fale aqui o que você conclui das visualizações e dos resultados. Tente explicar em detalhes por que um resultado, na sua opinião, foi melhor do que outro. Esta explicação pode incluir hipóteses para resultados melhores ou resultados piores. Também pode falar das dificuldades enfrentadas durante o trabalho e como conseguiu contorná-las.</p>
