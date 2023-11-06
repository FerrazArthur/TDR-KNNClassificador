from scipy.spatial.distance import minkowski

from itertools import combinations

import pandas as pd
from typing import Tuple, Dict

from utils.config import csv_order

import matplotlib.pyplot as plt
import numpy as np

def get_mean_pearson_correlation_matrix_between_dataframe(dataframes):
    """
    Calculates and return the max pearson correlation between each dataframe and store in a dictionary.

    Args:
        dataframes (Dict[pd.DataFrame]): Dictionary with the dataframes.
    """
    # Crie um DataFrame vazio com os nomes de DataFrames como índice e colunas
    correlation_df = pd.DataFrame(index=csv_order, columns=csv_order)

    # Percorra o dicionário e calcule a correlação entre os DataFrames
    for name1, df1 in dataframes.items():
        for name2, df2 in dataframes.items():
            correlation = df1.corrwith(df2, axis=1, method='pearson')
            correlation_df.loc[name1, name2] = correlation.mean()
    print(correlation_df)
    # # Exiba o DataFrame de correlações
    # plt.imshow(correlation_df.values, cmap='viridis', interpolation='nearest')

    # # Adicione rótulos de coluna e linha (opcional)
    # plt.xticks(range(len(correlation_df.columns)), correlation_df.columns)
    # plt.yticks(range(len(correlation_df.columns)), correlation_df.columns)

    # # Adicione uma barra de cores (opcional)
    # plt.colorbar()

    # # Mostre o gráfico
    # plt.show()

def get_mean_minkowski_distance_between_dataframe(dataframes:pd.DataFrame, p:int=2)-> Dict[str, str]:
    """
    Calculates and return the max euclidean distance between each dataframe and store in a dictionary.
    Args:
        dataframes (Dict[pd.DataFrame]): Dictionary with the dataframes.
        p (int, optional): Order of the minkowski distance. Defaults to 2.
    
    Returns:
        Dict[str, str]: Dictionary with the mean distance between each dataframe.
    """
    # Crie um DataFrame vazio com os nomes de DataFrames como índice e colunas
    distance_df = pd.DataFrame(index=csv_order, columns=csv_order)

    # Percorra o dicionário e calcule a correlação entre os DataFrames
    class_combinations = combinations(csv_order, 2)
    for name1, name2 in class_combinations:
        df1 = dataframes[name1]
        df2 = dataframes[name2]
        # Crie uma lista com todas as combinações de linhas entre os dois DataFrames
        row_combinations = list(combinations(df1.index, r=2))
        distance_raw = []
        for comb in row_combinations:
            # Calcule a distância entre as duas linhas
            distance_raw.append(minkowski(df1.loc[comb[0]], df2.loc[comb[1]], p))
        
        # Calcule a média das distâncias
        distance_df.loc[name1, name2] = np.mean(distance_raw)
        distance_df.loc[name2, name1] = np.mean(distance_raw)
    # Para representar a distância entre um DataFrame e ele mesmo, calcule a distância entre a média de suas linhas
    for name in csv_order:
        distance_df.loc[name, name] = minkowski(dataframes[name].mean(), dataframes[name].mean(), p)
    print(distance_df)

def get_mean_minkowski_distance_between_mean_dataframe(dataframes:pd.DataFrame, p:int=2)-> Dict[str, str]:
    """
    Calculates and return the max euclidean distance between the means of each dataframe and store in a dictionary.
    Args:
        dataframes (Dict[pd.DataFrame]): Dictionary with the dataframes.
        p (int, optional): Order of the minkowski distance. Defaults to 2.
    
    Returns:
        Dict[str, str]: Dictionary with the mean distance between each dataframe.
    """
    # Crie um DataFrame vazio com os nomes de DataFrames como índice e colunas
    distance_df = pd.DataFrame(index=csv_order, columns=csv_order)

    # Percorra o dicionário e calcule a correlação entre os DataFrames
    class_combinations = combinations(csv_order, 2)
    for name1, name2 in class_combinations:
        df1 = dataframes[name1]
        df2 = dataframes[name2]

        distance_df.loc[name1, name2] = minkowski(df1.mean(), df2.mean(), p)
        distance_df.loc[name2, name1] = distance_df.loc[name1, name2]
    # Para representar a distância entre um DataFrame e ele mesmo, calcule a distância entre a média de suas linhas
    for name in csv_order:
        distance_df.loc[name, name] = minkowski(dataframes[name].mean(), dataframes[name].mean(), p)
    print(distance_df)

def get_mean_distance_in_dataframe(df: pd.DataFrame, p: int = 2)-> float:
    """
    Calculates the mean minkowski distance between the rows of a dataframe.

    Args:
        df (pd.DataFrame): Dataframe with the data.
        p (int, optional): Order of the minkowski distance. Defaults to 2.
    
    Returns:
        float: Mean distance.
    """
    mean_distance = 0
    count = 0
    for i in range(len(df)):
        for j in range(i+1, len(df)):
            mean_distance += minkowski(df.iloc[i], df.iloc[j], p)
            count+=1
    return mean_distance / (len(df)*(len(df)-1)/2)