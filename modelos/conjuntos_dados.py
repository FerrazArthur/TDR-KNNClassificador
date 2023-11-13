import pandas as pd
import random
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from modelos.dados import Dados

def treino_regular(dados:Dados, train_size:int):
    """
    Retorna a saída do método regular de divisão de amostras do scipy em treino e teste.
    """
    # Data_dict é um dicionário que armazena um dataframe por classe, contendo 
    # todas as amostras da classe
    data_dict = dados.dicionario_dados
    amostras = []
    nomes = []

    for key, value in data_dict.items():
        codigo = key
        for _, row in value.iterrows():
            amostras.append(row.values)
            nomes.append(codigo)
    
    # O algoritmo train_test_split randomiza e distribui a amostra entre dois conjuntos
    # o de teste e o de treino.
    return train_test_split(amostras, nomes, train_size=train_size, stratify=nomes)

def treino_media(dados:Dados, train_size:int):
    """
        Retorna um conjunto de treino que contém apenas um exemplo por classe e esse é uma média dos primeiros 'corte' elementos de cada classe.
        Retorna o restante como conjunto de testes
    """
    data_dict = dados.dicionario_dados
    amostras_treino = []
    nomes_treino = []

    amostras_teste = []
    nomes_teste = []

    train_size_for_class = round(train_size/dados.num_classes)

    random_int = random.randint(0, 100)

    for key, value in data_dict.items():
        codigo = key
        # train_prop = train_size_for_class / value.shape[0]
        # Dividindo o dataframe em treino e teste
        treino_temp, teste_temp = train_test_split(value, train_size=train_size_for_class, random_state=random_int)
        
        # Adicionando os dataframes de treino como sendo uma média de todos os valores escolhidos
        amostras_treino.append(treino_temp.mean().values) # São séries temporais então podem ser guardadas em uma lista
        nomes_treino.append(codigo)

        # Adicionando os dataframes de teste
        amostras_teste.extend(teste_temp.values)
        [nomes_teste.append(codigo) for _ in range(teste_temp.shape[0])]
    
    # Embaralhando os dados de treino
    amostras_treino, nomes_treino = shuffle(amostras_treino, nomes_treino)
    
    # Embaralhando os dados de teste
    amostras_teste, nomes_teste = shuffle(amostras_teste, nomes_teste)

    return amostras_treino, amostras_teste, nomes_treino, nomes_teste