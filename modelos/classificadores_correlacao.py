import pandas as pd
import numpy as np
import random
from sklearn.base import BaseEstimator, ClassifierMixin
from scipy.spatial.distance import minkowski
from typing import List

class ClassificadorCorrelacao(BaseEstimator, ClassifierMixin):
    """
    Classificador que utiliza a correlação de Pearson entre a média de uma classe, obtida
    do conjunto de treino e a amostra a ser classificada.
    """
    def __init__(self):
        self.medias_por_classe = {}

    def fit(self, X_treino:List[List[float]], y_treino:List[List[str]]):
        """
        Calcula a média de cada classe do conjunto de treino e armazena em um dicionário.
        Args:
            X_treino (List[List[Float]]): Conjunto de treino.
            y_treino (List[List[str]]): Classes do conjunto de treino.
        """
        X_treino_np = np.array(X_treino)
        for classe in list(set(y_treino)):
            # Cria uma máscara booleana com True em todas as posições que contém a classe atual, 
            # False nas demais.
            mask = [x == classe for x in y_treino]
            self.medias_por_classe[classe] = pd.DataFrame(X_treino_np[mask], index=None, \
                                                           columns=None).mean(axis=0)
        return self

    def predict(self, X_teste:List[List[float]])-> List[str]:
        """
        Calcula a correlação de Pearson entre a média de cada classe e cada amostra a ser 
        classificada.
        Essa classe é atribuida a amostra.
        Retorna uma lista com a classificação das amostras.
        Args:
            X_teste (List[List[Float]]): Conjunto de teste.
        Returns:
            y_pred (List[str]): Classes preditas.
        """
        if self.medias_por_classe is None:
            raise ValueError("O modelo ainda não foi treinado. Use o método fit primeiro.")

        y_pred = []
        for amostra in X_teste:
            amostra_tratada = pd.Series(amostra, index=None)

            correlacoes = [media.corr(amostra_tratada, method="pearson") for media \
                           in self.medias_por_classe.values()]

            indices_max_correlacao = [i for i, valor in enumerate(correlacoes) if \
                                      valor == max(correlacoes)]
            # Se houver duas classes com a mesma correlação, sorteia uma delas.
            classe_sorteada = random.choice(indices_max_correlacao)
            classe_predita = list(self.medias_por_classe.keys())[classe_sorteada]

            y_pred.append(classe_predita)

        return y_pred

class ClassificadorCorrelacaoCruzada(BaseEstimator, ClassifierMixin):
    """
    Classificador que utiliza o menor erro, calculado com a distância Euclidiana, entre a 
    autocorrelação da média de uma classe, obtida do conjunto de treino, e a correlação cruzada 
    entre essa média e a autocorrelação da amostra a ser classificada.

    Variáveis:
        sinal_medio_por_classe (Dict[str, pd.Series]): Dicionário que contém o sinal médio de
          cada classe do conjunto de treino.
        auto_correlacao_por_classe (Dict[str, np.ndarray]): Dicionário que contém a 
        autocorrelação de cada sinal médio de cada classe do conjunto de treino.
    """
    def __init__(self):
        self.sinal_medio_por_classe = {}
        self.auto_correlacao_por_classe = {}

    def fit(self, X_treino:List[List[float]], y_treino:List[str]):
        """
        Calcula o sinal médio de cada classe e a autocorrelação de cada sinal médio.
        Args:
            X_treino (List[List[Float]]): Conjunto de treino.
            y_treino (List[str]): Classes do conjunto de treino.
        """
        X_treino_np = np.array(X_treino)
        for classe in list(set(y_treino)):
            mask = [(valor == classe) for valor in y_treino]
            self.sinal_medio_por_classe[classe] = pd.DataFrame(X_treino_np[mask],\
                     index=None, columns=None).mean(axis=0)

        self.auto_correlacao_por_classe = {classe: np.correlate(sinal_medio.to_numpy(),\
             sinal_medio.to_numpy(), mode='full') \
            for classe, sinal_medio in self.sinal_medio_por_classe.items()}
        
        return self

    def predict(self, X_teste:List[List[float]])-> List[str]:
        """
        Calcula a correlação cruzada entre a média de cada classe e a amostra a ser classificada.
        Em seguida, calcula o erro entre a autocorrelação da média de cada classe e a correlação
        cruzada entre a média e a amostra. 
        A classe com menor erro é atribuida a amostra.
        """
        if self.sinal_medio_por_classe is None:
            raise ValueError("O modelo ainda não foi treinado. Use o método fit primeiro.")

        y_pred = []
        for amostra in X_teste:
            amostra_np = np.array(amostra)
            correlacoes_cruzadas = {classe: np.correlate(media.to_numpy(), amostra_np, mode='full')\
                                     for classe, media in self.sinal_medio_por_classe.items()}
            correlacoes_erro = {classe: minkowski(self.auto_correlacao_por_classe[classe], \
                            correlacao, p=2) for classe, correlacao in correlacoes_cruzadas.items()}
            indices_min_erro = [classe for classe, valor in correlacoes_erro.items()\
                                 if valor == min(list(correlacoes_erro.values()))]
            classe_sorteada = random.choice(indices_min_erro)
            y_pred.append(classe_sorteada)

        return y_pred