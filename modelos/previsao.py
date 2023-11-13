from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import pandas as pd
import numpy as np
from modelos.conjuntos_dados import treino_media, treino_regular
from modelos.dados import Dados
from visualizacao.visualizacao import imprime_matriz_confusao_e_relatorio_classificacao, imprime_testes_multiplos
from typing import List, Union
from statistics import mode

def obter_matriz_confusao(dados:Dados, treino, tamanho_treino, k:int=1, repeticoes:int=10, imprime_legenda:bool=True, imprime_acuracia:bool=True, titulo:str="", save_fig:bool=False):
    """
        Realiza o treino do classificador para uma distribuição e para um valor de k
        e imprime a matriz de confusão e o relatório de classificação.
    """
    previsoes_acumuladas = []

    X_train, X_test, y_train, y_test = treino(dados, train_size=tamanho_treino)
    knn_class = KNeighborsClassifier(n_neighbors=k, weights='distance')
    for _ in range(repeticoes):
        # Treinando o classificador
        knn_class.fit(X_train, y_train)

        # Realizando o teste
        previsoes_acumuladas.append(knn_class.predict(X_test))

    previsao_media = [mode(preds) for preds in zip(*previsoes_acumuladas)]

    mat_confusao = confusion_matrix(y_test, previsao_media, normalize='true')
    rel_classificacao = classification_report(y_test, previsao_media, output_dict=True)

    imprime_matriz_confusao_e_relatorio_classificacao(dados, mat_confusao, rel_classificacao, titulo=titulo, save_fig=save_fig)

#------------------------------------------------------------
def executar_knn_acuracia(X_train, X_test, y_train, y_test, k=1, weights='distance'):
    """
        Executa um classificador KNN básico com os dados fornecidos e retorna a acurácia do modelo.
    """
    knn_class = KNeighborsClassifier(n_neighbors=k, weights=weights)

    # Treinando o classificador
    knn_class.fit(X_train, y_train)

    # Realizando o teste
    ypred=knn_class.predict(X_test)
    return accuracy_score(y_test,ypred)

def obter_acuracia(dados:Dados, treino, tamanho_treino, k:int=1):
    """
        Obtém a acurácia do modelo.
    """
    X_train, X_test, y_train, y_test = treino(dados, train_size=tamanho_treino)
    return executar_knn_acuracia(X_train, X_test, y_train, y_test, k=k)

def obter_resultados(dados:Dados, treino, k_lista:List[int], tamanho_treino_lista:List[int], repeticoes:int=10):
    """
        Obtém os resultados de multiplos testes para as distribuições e valores de k fornecidos e os retorna em uma lista.
    """
    # Realiza o treino do classificador para cada distribuição e para cada valor de k
    resultados = []
    for k in k_lista:
        k_resultados = []
        for tamanho_treino in tamanho_treino_lista:
            # repete o teste 'repeticoes' vezes e tira a média
            k_resultados.append(pd.Series([obter_acuracia(dados, treino, tamanho_treino, k) for _ in range(repeticoes)]).mean())
        resultados.append(k_resultados)
    return resultados
def obter_resultados_matriz_confusao(dados:Dados, treino, k_lista:List[int], tamanho_treino_lista:List[int], repeticoes:int=10, titulo:str="", save_fig:bool=False):
    """
        Obtém os resultados de multiplos testes para as distribuições e valores de k fornecidos e os retorna em uma lista.
    """
    # Realiza o treino do classificador para cada distribuição e para cada valor de k
    for k in k_lista:
        for tamanho_treino in tamanho_treino_lista:
            obter_matriz_confusao(dados, treino, tamanho_treino, k, repeticoes, titulo="treino="+titulo+"_k="+str(k)+"_tamanhotreino="+str(tamanho_treino), save_fig=save_fig)

def executar_multiplos_testes(dados:Dados, k_lista:List[int]=[1, 3],\
 tamanho_treino_lista:List[Union[int, float]]=[0.300], treinos_lista:List[callable]=[treino_regular, treino_media],\
 imprime_legenda:bool=True, imprime_acuracia:bool=True):
    """
        Executa testes múltiplos com diferentes valores de k e diferentes distribuições de dados de treino e teste.
    """
    # Realiza o treino do classificador para cada distribuição e para cada valor de k

    # Garante que tenhamos valores absolutos para o tamanho do treino, não em porcentagem
    tamanho_treino_lista_int = []
    for tamanho_treino in tamanho_treino_lista:
        if isinstance(tamanho_treino, float) and tamanho_treino < 1.0:
            tamanho_treino_lista_int.append(round(tamanho_treino*dados.num_amostras))
        else:
            tamanho_treino_lista_int.append(round(tamanho_treino))

    for treino in treinos_lista:
        if (imprime_legenda == True):
            print(f"Treino: {treino.__name__}")
        resultados = obter_resultados(dados, treino, k_lista, tamanho_treino_lista_int)

        imprime_testes_multiplos(resultados, tamanho_treino_lista_int, k_lista)

def executar_multiplos_testes_matriz_confusao(dados:Dados, k_lista:List[int]=[1, 3],\
     tamanho_treino_lista:List[Union[int, float]]=[0.300], repeticoes:int=10, \
    treinos_lista:List[callable]=[treino_regular, treino_media], imprime_legenda:bool=True, save_fig:bool=False):
    """
        Executa testes múltiplos com diferentes valores de k e diferentes distribuições de dados de treino e teste.
    """
    # Realiza o treino do classificador para cada distribuição e para cada valor de k

    # Garante que tenhamos valores absolutos para o tamanho do treino, não em porcentagem
    tamanho_treino_lista_int = []
    for tamanho_treino in tamanho_treino_lista:
        if isinstance(tamanho_treino, float) and tamanho_treino < 1.0:
            tamanho_treino_lista_int.append(round(tamanho_treino*dados.num_amostras))
        else:
            tamanho_treino_lista_int.append(round(tamanho_treino))

    for treino in treinos_lista:
        if (imprime_legenda == True):
            print(f"Treino: {treino.__name__}")
        obter_resultados_matriz_confusao(dados, treino, k_lista, tamanho_treino_lista_int, repeticoes=repeticoes, titulo=treino.__name__, save_fig=save_fig)

#------------------------------------------------------------