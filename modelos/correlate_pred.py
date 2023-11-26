import pandas as pd
import random
from sklearn.base import BaseEstimator, ClassifierMixin

class ClassificadorCorrelacaoCruzada(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.medias_por_classe = None

    def fit(self, X_treino, y_treino):
        self.medias_por_classe = {classe: pd.DataFrame(X_treino[y_treino == classe], index=None, columns=None).mean(axis=1) for classe in list(set(y_treino))}
        return self

    def predict(self, X_teste):
        if self.medias_por_classe is None:
            raise ValueError("O modelo ainda não foi treinado. Use o método fit primeiro.")

        y_pred = []
        for amostra in X_teste:
            amostra_tratada = pd.Series(amostra, index=None)
            correlacoes = [media.corr(amostra_tratada, method="pearson") for media in self.medias_por_classe.values()]
            indices_max_correlacao = [i for i, valor in enumerate(correlacoes) if valor == max(correlacoes)]
            classe_sorteada = random.choice(indices_max_correlacao)
            classe_predita = list(self.medias_por_classe.keys())[classe_sorteada]
            y_pred.append(classe_predita)

        return y_pred