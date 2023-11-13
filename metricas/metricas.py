from scipy.spatial.distance import minkowski

from itertools import combinations

import pandas as pd
from typing import Tuple, Dict

from utils.config import csv_order

import matplotlib.pyplot as plt
import numpy as np

def obter_matriz_correlacao_media_entre_dataframe(dataframes):
    """
    Calcula e retorna a correlação de Pearson média entre cada dataframe e armazena em um dicionário.

    Args:
        dataframes (Dict[pd.DataFrame]): Dicionário com os dataframes.
    """
    # Cria um DataFrame vazio com os nomes dos DataFrames como índice e colunas
    matriz_correlacao = pd.DataFrame(index=csv_order, columns=csv_order)

    # Percorre o dicionário e calcula a correlação entre os DataFrames
    for nome1, df1 in dataframes.items():
        for nome2, df2 in dataframes.items():
            correlacao = df1.corrwith(df2, axis=1, method='pearson')
            matriz_correlacao.loc[nome1, nome2] = correlacao.mean()
    print(matriz_correlacao)

def obter_distancia_media_minkowski_entre_dataframe(dataframes:pd.DataFrame, p:int=2)-> Dict[str, str]:
    """
    Calcula e retorna a distância euclidiana média entre cada dataframe e armazena em um dicionário.
    Args:
        dataframes (Dict[pd.DataFrame]): Dicionário com os dataframes.
        p (int, opcional): Ordem da distância de Minkowski. Padrão é 2.
    
    Returns:
        Dict[str, str]: Dicionário com a distância média entre cada dataframe.
    """
    # Cria um DataFrame vazio com os nomes dos DataFrames como índice e colunas
    matriz_distancia = pd.DataFrame(index=csv_order, columns=csv_order)

    # Percorre o dicionário e calcula a distância de Minkowski entre os DataFrames
    combinacoes_classes = combinations(csv_order, 2)
    for nome1, nome2 in combinacoes_classes:
        df1 = dataframes[nome1]
        df2 = dataframes[nome2]
        # Cria uma lista com todas as combinações de linhas entre os dois DataFrames
        combinacoes_linhas = list(combinations(df1.index, r=2))
        distancia_raw = []
        for comb in combinacoes_linhas:
            # Calcula a distância entre as duas linhas
            distancia_raw.append(minkowski(df1.loc[comb[0]], df2.loc[comb[1]], p))
        
        # Calcula a média das distâncias
        matriz_distancia.loc[nome1, nome2] = np.mean(distancia_raw)
        matriz_distancia.loc[nome2, nome1] = np.mean(distancia_raw)
    # Para representar a distância entre um DataFrame e ele mesmo, calcula a distância entre a média de suas linhas
    for nome in csv_order:
        matriz_distancia.loc[nome, nome] = minkowski(dataframes[nome].mean(), dataframes[nome].mean(), p)
    print(matriz_distancia)

def obter_distancia_media_minkowski_entre_media_dataframe(dataframes:pd.DataFrame, p:int=2)-> Dict[str, str]:
    """
    Calcula e retorna a distância euclidiana média entre as médias de cada dataframe e armazena em um dicionário.
    Args:
        dataframes (Dict[pd.DataFrame]): Dicionário com os dataframes.
        p (int, opcional): Ordem da distância de Minkowski. Padrão é 2.
    
    Returns:
        Dict[str, str]: Dicionário com a distância média entre cada dataframe.
    """
    # Cria um DataFrame vazio com os nomes dos DataFrames como índice e colunas
    matriz_distancia = pd.DataFrame(index=csv_order, columns=csv_order)

    # Percorre o dicionário e calcula a distância de Minkowski entre os DataFrames
    combinacoes_classes = combinations(csv_order, 2)
    for nome1, nome2 in combinacoes_classes:
        df1 = dataframes[nome1]
        df2 = dataframes[nome2]

        matriz_distancia.loc[nome1, nome2] = minkowski(df1.mean(), df2.mean(), p)
        matriz_distancia.loc[nome2, nome1] = matriz_distancia.loc[nome1, nome2]
    # Para representar a distância entre um DataFrame e ele mesmo, calcula a distância entre a média de suas linhas
    for nome in csv_order:
        matriz_distancia.loc[nome, nome] = minkowski(dataframes[nome].mean(), dataframes[nome].mean(), p)
    print(matriz_distancia)

def obter_distancia_media_no_dataframe(df: pd.DataFrame, p: int = 2)-> float:
    """
    Calcula a média da distância de Minkowski entre as linhas de um dataframe.

    Args:
        df (pd.DataFrame): DataFrame com os dados.
        p (int, opcional): Ordem da distância de Minkowski. Padrão é 2.
    
    Returns:
        float: Distância média.
    """
    distancia_media = 0
    count = 0
    for i in range(len(df)):
        for j in range(i+1, len(df)):
            distancia_media += minkowski(df.iloc[i], df.iloc[j], p)
            count+=1
    return distancia_media / (len(df)*(len(df)-1)/2)
