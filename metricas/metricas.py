from scipy.spatial.distance import minkowski

from itertools import combinations

import pandas as pd
from typing import Tuple, Dict

import matplotlib.pyplot as plt
import numpy as np

def obter_linha_maior_distancia_minkowski_entre_dataframes(dados:Dict[str, pd.DataFrame], p:int=2)-> Tuple[str, int, float]:
    """
    Calcula e retorna a linha com maior variação entre todos os dataframes.

    Args:
        dados (pd.DataFrame): Dados, Dicionário com os dataframes.
        p (int, opcional): Ordem da distância de Minkowski. Padrão é 2.
    
    Returns:
        Tuple[str, int, float]: Nome da classe, a linha com maior variação e o valor da variação.
    """
    # Maior variação encontrada até o momento
    maior_variacao = 0
    # Nome da classe com maior variação
    nome_classe = ''
    # Linha com maior variação
    linha_maior_variacao = ''
    # Percorre o dicionário e calcula a variação entre os DataFrames
    for nome in dados.keys():
        df = dados[nome]
        df_mean = df.mean()
        for i in range(len(df)):
            variacao = minkowski(df.iloc[i], df_mean, p)
            if variacao > maior_variacao:
                maior_variacao = variacao
                nome_classe = nome
                linha_maior_variacao = i
    return nome_classe, linha_maior_variacao, maior_variacao

def obter_distancia_media_minkowski_entre_dataframe(dados:Dict[str, pd.DataFrame], p:int=2)-> Dict[str, str]:
    """
    Calcula e retorna a distância euclidiana média entre cada dataframe e armazena em um dicionário.
    Utiliza a média do primeiro para calcular a distancia à todas as linhas do segundo e faz uma média dos resultados.
    Args:
        dados (pd.DataFrame): Dicionário com os dataframes.
        p (int, opcional): Ordem da distância de Minkowski. Padrão é 2.
    
    Returns:
        Dict[str, str]: Dicionário com a distância média entre cada dataframe.
    """
    # Cria um DataFrame vazio com os nomes dos DataFrames como índice e colunas
    matriz_distancia = pd.DataFrame(index=dados.keys(), columns=dados.keys())

    # Percorre o dicionário e calcula a distância de Minkowski entre os DataFrames
    combinacoes_classes = combinations(dados.keys(), 2)
    for nome1, nome2 in combinacoes_classes:
        df1 = dados[nome1]
        df2 = dados[nome2]
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
    for nome in dados.keys():
        matriz_distancia.loc[nome, nome] = minkowski(dados[nome].mean(), dados[nome].mean(), p)
    print(matriz_distancia)

def obter_distancia_media_minkowski_entre_media_dataframe(dados:Dict[str, pd.DataFrame], p:int=2)-> Dict[str, str]:
    """
    Calcula e retorna a distância euclidiana média entre as médias de cada dataframe e armazena em um dicionário.
    Utiliza a média do primeiro para calcular a distancia à media do segundo. (mais rápido)
    Args:
        dados (pd.DataFrame): Dicionário com os dataframes.
        p (int, opcional): Ordem da distância de Minkowski. Padrão é 2.
    
    Returns:
        Dict[str, str]: Dicionário com a distância média entre cada dataframe.
    """
    # Cria um DataFrame vazio com os nomes dos DataFrames como índice e colunas
    matriz_distancia = pd.DataFrame(index=dados.keys(), columns=dados.keys())

    # Percorre o dicionário e calcula a distância de Minkowski entre os DataFrames
    combinacoes_classes = combinations(dados.keys(), 2)
    for nome1, nome2 in combinacoes_classes:
        df1 = dados[nome1]
        df2 = dados[nome2]

        matriz_distancia.loc[nome1, nome2] = minkowski(df1.mean(), df2.mean(), p)
        matriz_distancia.loc[nome2, nome1] = matriz_distancia.loc[nome1, nome2]
    # Para representar a distância entre um DataFrame e ele mesmo, calcula a distância entre a média de suas linhas
    for nome in dados.keys():
        matriz_distancia.loc[nome, nome] = minkowski(dados[nome].mean(), dados[nome].mean(), p)
    print(matriz_distancia.to_string())

def obter_distancia_media_no_dataframe(df: pd.DataFrame, p: int = 2)-> float:
    """
    Calcula a média da distância de Minkowski entre as linhas de um dataframe.
    Calcula a distância entre todas as linhas e faz a média dos resultados. (demorado)

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

def obter_escore_padrao(df:pd.DataFrame, ddof:int=1, p:int=2)-> pd.Series:
    """
    Calcula o escore padrão de cada linha de um dataframe.

    Args:
        df (pd.DataFrame): DataFrame com os dados.
        ddof (int, opcional): Graus de liberdade. Padrão é 1.
        p (int, opcional): Ordem da distância de Minkowski. Padrão é 2.
    
    Returns:
        pd.Series: Vetor com os escores padrão.
    """
    distancias = obter_vetor_distancias_a_media_dataframe(df.as_numpy(), p)
    media = distancias.mean()
    desvio_padrao = distancias.std(ddof=ddof)
    return (distancias - media) / desvio_padrao

def obter_vetor_distancias_a_media_dataframe(df: np.array, p: int = 2)-> pd.Series:
    """
    Calcula a distância de Minkowski entre as linhas de um dataframe e a média do dataframe.

    Args:
        df (np.array(np.array(float))): representação em numpy de um DataFrame pandas com os dados.
        p (int, opcional): Ordem da distância de Minkowski. Padrão é 2.
    
    Returns:
        pd.Series: Vetor com as distâncias
    """
    sinal_medio = np.mean(df, axis=0)
    vetor_distancias = np.zeros(np.shape(df)[0])

    for i, linha in df.iterrows():
        vetor_distancias[i] = minkowski(linha, sinal_medio, p)
    return pd.Series(vetor_distancias)