import random
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from dados.dados import Dados
from visualizacao.visualizacao import imprimir_contagem_amostras

def treino_regular(dados: Dados, train_size: int, print_table: bool = False):
    """
    Retorna a saída do método regular de divisão de amostras do scipy em treino e teste.
    Args:
        dados: Objeto Dados contendo as amostras
        train_size: Número exato de amostras para o conjunto de treino
        print_table: Se True, imprime a contagem de amostras por classe em cada conjunto
    """
    data_dict = dados.dicionario_dados
    amostras = []
    nomes = []

    for key, value in data_dict.items():
        codigo = key
        for _, row in value.iterrows():
            amostras.append(row.values)
            nomes.append(codigo)
    
    X_train, X_test, y_train, y_test = train_test_split(amostras, nomes, train_size=train_size,\
                                                         stratify=nomes)

    if print_table:
        imprimir_contagem_amostras(y_train, y_test)
    
    return X_train, X_test, y_train, y_test

def treino_media(dados: Dados, train_size: int, print_table: bool = False):
    """
    Retorna um conjunto de treino com a acomulação das amostras de treino para cada classe em um
     único sinal, enquanto garante igual distribuição.
    Retorna o restante como conjunto de testes
    """
    data_dict = dados.dicionario_dados
    amostras_treino = []
    nomes_treino = []

    amostras_teste = []
    nomes_teste = []

    # Descobre a quantidade ideal de amostras para treino de cada classe
    train_size_for_class = round(train_size / dados.num_classes)

    # Gera um número aleatório para garantir que a divisão seja aleatória mas constante 
    # dentro do loop
    random_int = random.randint(0, 100)

    for key, value in data_dict.items():
        codigo = key
        treino_temp, teste_temp = train_test_split(value, train_size=train_size_for_class,\
                                                    random_state=random_int)
        
        # Cria uma única amostra para cada classe com a média das amostras de treino
        amostras_treino.append(treino_temp.mean().values)
        nomes_treino.append(codigo)

        amostras_teste.extend(teste_temp.values)
        [nomes_teste.append(codigo) for _ in range(teste_temp.shape[0])]
    
    # Embaralha as amostras para garantir que as classes não estejam agrupadas
    amostras_treino, nomes_treino = shuffle(amostras_treino, nomes_treino)
    amostras_teste, nomes_teste = shuffle(amostras_teste, nomes_teste)

    if print_table:
        imprimir_contagem_amostras(nomes_treino, nomes_teste)

    return amostras_treino, amostras_teste, nomes_treino, nomes_teste