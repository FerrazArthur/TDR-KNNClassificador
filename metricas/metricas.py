from scipy.spatial.distance import minkowski

from itertools import combinations

import pandas as pd
from typing import Tuple, Dict

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

def obter_distancia_minkowski_entre_classes(dados:Dict[str, pd.DataFrame], p:int=2)-> Dict[str, str]:
    """
    Calcula e retorna a distância euclidiana entre as médias de cada dataframe e armazena em um dicionário.
    Utiliza a média do primeiro conjunto de amostras para calcular a distancia à media do segundo.
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

        matriz_distancia.loc[nome1, nome2] = minkowski(df1.mean().to_numpy(), df2.mean().to_numpy(), p)
        matriz_distancia.loc[nome2, nome1] = matriz_distancia.loc[nome1, nome2]
    # Para representar a distância entre um DataFrame e ele mesmo, calcula a distância entre a média de suas linhas
    for nome in dados.keys():
        matriz_distancia.loc[nome, nome] = 0.0
    return matriz_distancia

def obter_distancia_minkowski_min_mean_max_em_classes(dados:Dict[str, pd.DataFrame], p:int=2)-> pd.DataFrame:
    """
    Calcula e retorna a distância euclidiana minima, media e máxima entre cada amostra de uma classe e sua média.
    Também adiciona o coeficiênte de variação das distâncias.
    Args:
        dados (pd.DataFrame): Dicionário com os dataframes.
        p (int, opcional): Ordem da distância de Minkowski. Padrão é 2.
    """
    # Cria um DataFrame vazio com os nomes dos DataFrames como índice e as colunas
    matriz_distancia = pd.DataFrame(index=dados.keys(), columns=['Máxima', 'Media' , 'Mínima', 'CV'], dtype=float)

    # Percorre o dicionário e calcula a distância de Minkowski entre os DataFrames
    for nome, df in dados.items():
        distancias = obter_vetor_distancias_a_media_dataframe(df, p)
        maximo = max(distancias)
        mean = distancias.mean()
        minimo = min(distancias)
        desvio_padrao = distancias.std()
        cv = desvio_padrao / mean

        matriz_distancia.loc[nome, 'Máxima'] = maximo
        matriz_distancia.loc[nome, 'Media'] = mean
        matriz_distancia.loc[nome, 'Mínima'] = minimo
        matriz_distancia.loc[nome, 'CV'] = cv

    return matriz_distancia

# Não usar. Não faz sentido calcular correlação entre médias de classes porque da valores muito pertos de 1, mesmo depois de filtrar com
# Autocorrelação
# def obter_correlecao_entre_classes(dados:Dict[str, pd.DataFrame])-> Dict[str, str]:
#     """
#     Calcula e retorna a correlação entre as médias de cada dataframe e armazena em um dicionário.
#     Utiliza a média do primeiro conjunto de amostras para calcular a correlação com a media do segundo.
#     Args:
#         dados (pd.DataFrame): Dicionário com os dataframes.
#     """
#     # Cria um DataFrame vazio com os nomes dos DataFrames como índice e colunas
#     matriz_correlacao = pd.DataFrame(index=dados.keys(), columns=dados.keys())

#     # Percorre o dicionário e calcula a correlação entre os DataFrames
#     combinacoes_classes = combinations(dados.keys(), 2)
#     for nome1, nome2 in combinacoes_classes:
#         df1 = dados[nome1]
#         df2 = dados[nome2]
#         autocorr_df1 = pd.Series(np.correlate(df1.mean().to_numpy(), df1.mean().to_numpy(), mode='full'), index=None)
#         autocorr_df2 = pd.Series(np.correlate(df2.mean().to_numpy(), df2.mean().to_numpy(), mode='full'), index=None)
#         matriz_correlacao.loc[nome1, nome2] = autocorr_df1.corr(autocorr_df2, method='pearson')
#         matriz_correlacao.loc[nome2, nome1] = matriz_correlacao.loc[nome1, nome2]
#     # Para representar a correlação entre um DataFrame e ele mesmo, calcula a correlação entre a média de suas linhas
#     for nome in dados.keys():
#         matriz_correlacao.loc[nome, nome] = 1.0
#     return matriz_correlacao

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
    distancias = obter_vetor_distancias_a_media_dataframe(df, p)
    media = distancias.mean()
    desvio_padrao = distancias.std(ddof=ddof)
    return (distancias - media) / desvio_padrao

def obter_vetor_distancias_a_media_dataframe(df: pd.DataFrame, p: int = 2)-> pd.Series:
    """
    Calcula a distância de Minkowski entre as linhas de um dataframe e a média do dataframe.

    Args:
        df pd.Dataframe: um DataFrame pandas com os dados.
        p (int, opcional): Ordem da distância de Minkowski. Padrão é 2.
    
    Returns:
        pd.Series: Vetor com as distâncias
    """
    sinal_medio = np.mean(df, axis=0)
    vetor_distancias = np.zeros(np.shape(df)[0])

    for i, linha in df.iterrows():
        vetor_distancias[i] = minkowski(linha, sinal_medio, p)
    return pd.Series(vetor_distancias)