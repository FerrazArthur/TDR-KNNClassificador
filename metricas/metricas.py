from scipy.spatial.distance import minkowski

from modelos.dados import Dados

from itertools import combinations

import pandas as pd
from typing import Tuple, Dict

import matplotlib.pyplot as plt
import numpy as np

# def obter_matriz_correlacao_media_entre_dataframe(dados:Dados, p:int=2)-> Dict[str, str]:
#     """
#     Calcula e retorna a correlação de Pearson média entre cada dataframe e armazena em um dicionário.

#     Args:
#         dados (Dados): Dados, Dicionário com os dataframes.
#         p (int, opcional): Ordem da distância de Minkowski. Padrão é 2.
#     """
#     # Cria um DataFrame vazio com os nomes dos DataFrames como índice e colunas
#     matriz_correlacao = pd.DataFrame(index=dados.classes_lista, columns=dados.classes_lista)

#     # Percorre o dicionário e calcula a correlação entre os DataFrames
#     for nome1 in dados.classes_lista:
#         for nome2 in dados.classes_lista[2:]:
#             df1 = dados.dicionario_dados[nome1]
#             df2 = dados.dicionario_dados[nome2]
#             #correlacao = df1.corrwith(df2, axis=1, method='pearson')
#             correlacao_out = []
#             for i in range(len(df1)):
#                 correlacao_in = []
#                 for j in range(len(df2)):
#                     correlacao_in.append(np.corrcoef(df1.iloc[i], df2.iloc[j])[0, 1])
#                 correlacao_out.append(sum(correlacao_in)/len(correlacao_in))
#             print (correlacao_out)
#             exit()
#             matriz_correlacao.loc[nome1, nome2] = sum(correlacao_out)/len(correlacao_out)
#     print(matriz_correlacao)

def obter_linha_maior_distancia_minkowski_entre_dataframes(dados:Dados, p:int=2)-> Tuple[str, int, float]:
    """
    Calcula e retorna a linha com maior variação entre todos os dataframes.

    Args:
        dados (Dados): Dados, Dicionário com os dataframes.
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
    for nome in dados.classes_lista:
        df = dados.dicionario_dados[nome]
        df_mean = df.mean()
        for i in range(len(df)):
            variacao = minkowski(df.iloc[i], df_mean, p)
            if variacao > maior_variacao:
                maior_variacao = variacao
                nome_classe = nome
                linha_maior_variacao = i
    return nome_classe, linha_maior_variacao, maior_variacao

def obter_distancia_media_minkowski_entre_dataframe(dados:Dados, p:int=2)-> Dict[str, str]:
    """
    Calcula e retorna a distância euclidiana média entre cada dataframe e armazena em um dicionário.
    Args:
        dados (Dados): Dados, Dicionário com os dataframes.
        p (int, opcional): Ordem da distância de Minkowski. Padrão é 2.
    
    Returns:
        Dict[str, str]: Dicionário com a distância média entre cada dataframe.
    """
    # Cria um DataFrame vazio com os nomes dos DataFrames como índice e colunas
    matriz_distancia = pd.DataFrame(index=dados.classes_lista, columns=dados.classes_lista)

    # Percorre o dicionário e calcula a distância de Minkowski entre os DataFrames
    combinacoes_classes = combinations(dados.classes_lista, 2)
    for nome1, nome2 in combinacoes_classes:
        df1 = dados.dicionario_dados[nome1]
        df2 = dados.dicionario_dados[nome2]
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
    for nome in dados.classes_lista:
        matriz_distancia.loc[nome, nome] = minkowski(dados.dicionario_dados[nome].mean(), dados.dicionario_dados[nome].mean(), p)
    print(matriz_distancia)

def obter_distancia_media_minkowski_entre_media_dataframe(dados:Dados, p:int=2)-> Dict[str, str]:
    """
    Calcula e retorna a distância euclidiana média entre as médias de cada dataframe e armazena em um dicionário.
    Args:
        dados (Dados): Dados, Dicionário com os dataframes.
        p (int, opcional): Ordem da distância de Minkowski. Padrão é 2.
    
    Returns:
        Dict[str, str]: Dicionário com a distância média entre cada dataframe.
    """
    # Cria um DataFrame vazio com os nomes dos DataFrames como índice e colunas
    matriz_distancia = pd.DataFrame(index=dados.classes_lista, columns=dados.classes_lista)

    # Percorre o dicionário e calcula a distância de Minkowski entre os DataFrames
    combinacoes_classes = combinations(dados.classes_lista, 2)
    for nome1, nome2 in combinacoes_classes:
        df1 = dados.dicionario_dados[nome1]
        df2 = dados.dicionario_dados[nome2]

        matriz_distancia.loc[nome1, nome2] = minkowski(df1.mean(), df2.mean(), p)
        matriz_distancia.loc[nome2, nome1] = matriz_distancia.loc[nome1, nome2]
    # Para representar a distância entre um DataFrame e ele mesmo, calcula a distância entre a média de suas linhas
    for nome in dados.classes_lista:
        matriz_distancia.loc[nome, nome] = minkowski(dados.dicionario_dados[nome].mean(), dados.dicionario_dados[nome].mean(), p)
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
