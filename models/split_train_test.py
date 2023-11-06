import pandas as pd
import random
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

def treino_regular(data_dict: dict, test_size: float=0.994):
    """
    Returns the output of the scipy regular method of splitting the samples into train and test.
    """
    # Data_dict é um dicionário que armazena um dataframe por classe, contendo 
    # todas as amostras da classe
    amostras = []
    nomes = []

    for key, value in data_dict.items():
        for _, row in value.iterrows():
            amostras.append(row.values)
            nomes.append(key)
    
    # O algoritmo train_test_split randomiza e distribui a amostra entre dois conjuntos
    # o de teste e o de treino.
    return train_test_split(amostras, nomes, test_size=test_size, stratify=nomes)

def treino_regular_explicit(data_dict: dict, train_size: float=30):
    """
    Returns the output of the scipy regular method of splitting the samples into train and test.
    Keeps train with test_size samples
    """
    # Data_dict é um dicionário que armazena um dataframe por classe, contendo 
    # todas as amostras da classe
    amostras = []
    nomes = []
    for key, value in data_dict.items():
        for _, row in value.iterrows():
            amostras.append(row.values)
            nomes.append(key)
    
    # O algoritmo train_test_split randomiza e distribui a amostra entre dois conjuntos
    # o de teste e o de treino.

    return train_test_split(amostras, nomes, train_size=train_size, stratify=nomes)

def treino_media(data_dict: dict, test_size: float=0.994):
    """
        Retorna um conjunto de treino que contém apenas um exemplo por classe e esse é uma média dos primeiros 'corte' elementos de cada classe.
        Retorna o restante como conjunto de testes
    """
    
    train_amostras = []
    train_nomes = []

    test_amostras = []
    test_nomes = []

    for key, value in data_dict.items():
        # Dividindo o dataframe em treino e teste
        train_temp, test_temp = train_test_split(value, test_size=test_size)
        
        # Adicionando os dataframes de treino como sendo uma média de todos os valores escolhidos
        train_amostras.append(train_temp.mean().values) # São time series então podem ser guardados em uma lista
        train_nomes.append(key)

        # Adicionando os dataframes de teste
        test_amostras.extend(test_temp.values)
        [test_nomes.append(key) for _ in range(test_temp.shape[0])]
    
    # Embaralhando os dados de treino
    train_amostras, train_nomes = shuffle(train_amostras, train_nomes)
    
    # Embaralhando os dados de teste
    test_amostras, test_nomes = shuffle(test_amostras, test_nomes)

    return train_amostras, test_amostras, train_nomes, test_nomes

def treino_media_explicit(data_dict: dict, train_size: int=30):
    """
        Retorna um conjunto de treino que contém apenas um exemplo por classe e esse é uma média dos primeiros 'corte' elementos de cada classe.
        Retorna o restante como conjunto de testes
    """
    
    train_amostras = []
    train_nomes = []

    test_amostras = []
    test_nomes = []

    # Conta a quantidade de amostras
    n_amostras = sum(df.shape[0] for df in data_dict.values())
    n_classes = len(data_dict.keys())

    train_size_for_class = round(train_size/n_classes)

    random_int = random.randint(0, 100)

    for key, value in data_dict.items():
        # train_prop = train_size_for_class / value.shape[0]
        # Dividindo o dataframe em treino e teste
        train_temp, test_temp = train_test_split(value, train_size=train_size_for_class, random_state=random_int)
        
        # Adicionando os dataframes de treino como sendo uma média de todos os valores escolhidos
        train_amostras.append(train_temp.mean().values) # São time series então podem ser guardados em uma lista
        train_nomes.append(key)

        # Adicionando os dataframes de teste
        test_amostras.extend(test_temp.values)
        [test_nomes.append(key) for _ in range(test_temp.shape[0])]
    
    # Embaralhando os dados de treino
    train_amostras, train_nomes = shuffle(train_amostras, train_nomes)
    
    # Embaralhando os dados de teste
    test_amostras, test_nomes = shuffle(test_amostras, test_nomes)

    return train_amostras, test_amostras, train_nomes, test_nomes