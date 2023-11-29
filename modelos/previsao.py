from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
from modelos.correlate_pred import ClassificadorCorrelacaoCruzada, ClassificadorCorrelacao
from modelos.conjuntos_dados import treino_media, treino_regular
from modelos.dados import Dados
from visualizacao.visualizacao import imprime_matriz_confusao_e_relatorio_classificacao
from typing import List, Union
from statistics import mode

def obter_matriz_confusao_KNN(dados:Dados, divisao_treino, tamanho_treino, k:int=1, repeticoes:int=10,\
                               titulo:str="", save_fig:bool=False):
    """
        Realiza o treino do classificador para uma distribuição e para um valor de k
        e imprime a matriz de confusão e o relatório de classificação.
        Args:
            dados (Dados): Objeto Dados com os dados a serem utilizados.
            divisao_treino (callable): Função de divisão de treino a ser utilizada.
            tamanho_treino (int): Tamanho do conjunto de treino.
            k (int, opcional): Número de vizinhos a serem considerados. Padrão é 1.
            repeticoes (int, opcional): Número de repetições de cada teste. Padrão é 10.
            titulo (str, opcional): Título do gráfico. Padrão é "".
            save_fig (bool, opcional): Se True, salva a figura gerada. Padrão é False.
    """
    previsoes_acumuladas = []

    X_train, X_test, y_train, y_test = divisao_treino(dados, train_size=tamanho_treino)
    knn_class = KNeighborsClassifier(n_neighbors=k, weights='distance')

    for _ in range(repeticoes):
        # Treinando o classificador
        knn_class.fit(X_train, y_train)

        # Realizando o teste
        previsoes_acumuladas.append(knn_class.predict(X_test))

    previsao_media = [mode(preds) for preds in zip(*previsoes_acumuladas)]

    mat_confusao = confusion_matrix(y_test, previsao_media, normalize='true')
    rel_classificacao = classification_report(y_test, previsao_media, output_dict=True)

    imprime_matriz_confusao_e_relatorio_classificacao(dados, mat_confusao, rel_classificacao,\
                                                       titulo=titulo, save_fig=save_fig)

def obter_resultados_matriz_confusao_KNN(dados:Dados, divisao_treino, k_lista:List[int],\
            tamanho_treino_lista:List[int], repeticoes:int=10, titulo:str="", save_fig:bool=False):
    """
        Obtém os resultados de multiplos testes para as distribuições e valores de k fornecidos e os 
        retorna em uma lista.
        Args:
            dados (Dados): Objeto Dados com os dados a serem utilizados.
            divisao_treino (callable): Função de divisão de treino a ser utilizada.
            k_lista (List[int]): Lista com os valores de k a serem testados.
            tamanho_treino_lista (List[int]): Lista com o tamanho do conjunto de treino em cada teste.
            repeticoes (int, opcional): Número de repetições de cada teste. Padrão é 10.
            titulo (str, opcional): Título do gráfico. Padrão é "".
            save_fig (bool, opcional): Se True, salva a figura gerada. Padrão é False.
    """
    # Realiza o treino do classificador para cada distribuição e para cada valor de k
    for k in k_lista:
        for tamanho_treino in tamanho_treino_lista:
            obter_matriz_confusao_KNN(dados, divisao_treino, tamanho_treino, k, repeticoes,\
                                   titulo=titulo+"/k_"+str(k)+"/amostra_treino_"+str(tamanho_treino),\
                                      save_fig=save_fig)

def executar_multiplas_previsoes_KNN_matriz_confusao(dados:Dados, k_lista:List[int]=[1, 3],\
     tamanho_treino_lista:List[Union[int, float]]=[0.300], repeticoes:int=10, \
    treinos_lista:List[callable]=[treino_regular, treino_media], imprime_legenda:bool=True,\
          save_fig:bool=False,
    fig_folder:str="resultados"):
    """
        Executa testes múltiplos com diferentes valores de k e diferentes distribuições de 
        dados de treino e teste.
        Args:
            dados (Dados): Objeto Dados com os dados a serem utilizados.
            k_lista (List[int], opcional): Lista com os valores de k a serem testados. Padrão é [1, 3].
            tamanho_treino_lista (List[Union[int, float]], opcional): Lista com o tamanho do 
            conjunto de treino em cada teste. Pode ser uma lista de inteiros ou de floats entre 0 e 1.
            repeticoes (int, opcional): Número de repetições de cada teste. Padrão é 10.
            treinos_lista (List[callable], opcional): Lista com as funções de treino a serem utilizadas.
            Padrão é [treino_regular, treino_media].
            imprime_legenda (bool, opcional): Se True, imprime o nome do classificador e o tamanho
            do conjunto de treino. Padrão é True.
            save_fig (bool, opcional): Se True, salva a figura gerada. Padrão é False.
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
        obter_resultados_matriz_confusao_KNN(dados, treino, k_lista, tamanho_treino_lista_int,\
                    repeticoes=repeticoes, titulo=fig_folder+"/"+"knn/"+treino.__name__, save_fig=save_fig)

#------------------------------------------------------------

def obter_matriz_confusao_correlacao(dados:Dados, classificador:callable, tamanho_treino:int,\
        repeticoes:int=10, titulo:str="", save_fig:bool=False):
    """
        Realiza o treino do classificador para uma distribuição e 
        imprime a matriz de confusão e o relatório de classificação.
        Args:
            dados (Dados): Objeto Dados com os dados a serem utilizados.
            classificador (callable): Construtor do classificador a ser utilizado.
            tamanho_treino (int): Tamanho do conjunto de treino.
            repeticoes (int, opcional): Número de repetições de cada teste. Padrão é 10.
            titulo (str, opcional): Título do gráfico. Padrão é "".
            save_fig (bool, opcional): Se True, salva a figura gerada. Padrão é False.
    """
    previsoes_acumuladas = []

    X_train, X_test, y_train, y_test = treino_regular(dados, train_size=tamanho_treino)

    classificador_instancia = classificador()

    for _ in range(repeticoes):
        # Treinando o classificador
        classificador_instancia.fit(X_train, y_train)

        # Realizando o teste
        previsoes_acumuladas.append(classificador_instancia.predict(X_test))

    previsao_media = [mode(preds) for preds in zip(*previsoes_acumuladas)]

    mat_confusao = confusion_matrix(y_test, previsao_media, normalize='true')
    rel_classificacao = classification_report(y_test, previsao_media, output_dict=True)

    imprime_matriz_confusao_e_relatorio_classificacao(dados, mat_confusao, rel_classificacao,\
                                                       titulo=titulo, save_fig=save_fig)

def obter_resultados_matriz_confusao_correlacao(dados:Dados, classificador:callable, \
            tamanho_treino_lista:List[int], repeticoes:int=10, titulo:str="", save_fig:bool=False):
    """
        Obtém os resultados de multiplos testes para as distribuições e os 
        retorna em uma lista.
        Args:
            dados (Dados): Objeto Dados com os dados a serem utilizados.
            classificador (callable): Construtor do classificador a ser utilizado.
            tamanho_treino_lista (List[int]): Lista com o tamanho do conjunto de treino em cada teste.
            repeticoes (int, opcional): Número de repetições de cada teste. Padrão é 10.
            titulo (str, opcional): Título do gráfico. Padrão é "".
            save_fig (bool, opcional): Se True, salva a figura gerada. Padrão é False.
    """

    for tamanho_treino in tamanho_treino_lista:
        obter_matriz_confusao_correlacao(dados, classificador, tamanho_treino, repeticoes,\
                                titulo=titulo+"/amostra_treino_"+str(tamanho_treino),\
                                    save_fig=save_fig)
            
def executar_multiplas_previsoes_correlacao_matriz_confusao(dados:Dados, classificador:callable,\
            tamanho_treino_lista:List[Union[int, float]]=[0.300], repeticoes:int=10,\
                  imprime_legenda:bool=True, save_fig:bool=False,
    fig_folder:str="resultados"):
    """
        Executa testes múltiplos com diferentes distribuições de dados para treino e teste.
        Args:
            dados (Dados): Objeto Dados com os dados a serem utilizados.
            classificador (callable): Construtor do classificador a ser utilizado.
            tamanho_treino_lista (List[Union[int, float]], opcional): Lista com o tamanho do 
            conjunto de treino em cada teste. Pode ser uma lista de inteiros ou de floats entre 0 e 1.
            repeticoes (int, opcional): Número de repetições de cada teste. Padrão é 10.
            imprime_legenda (bool, opcional): Se True, imprime o nome do classificador e o tamanho
            do conjunto de treino. Padrão é True.
            save_fig (bool, opcional): Se True, salva a figura gerada. Padrão é False.
    """
    # Realiza o treino do classificador para cada distribuição e para cada valor de k

    # Garante que tenhamos valores absolutos para o tamanho do treino, não em porcentagem
    tamanho_treino_lista_int = []
    for tamanho_treino in tamanho_treino_lista:
        if isinstance(tamanho_treino, float) and tamanho_treino < 1.0:
            tamanho_treino_lista_int.append(round(tamanho_treino*dados.num_amostras))
        else:
            tamanho_treino_lista_int.append(round(tamanho_treino))

    if (imprime_legenda == True):
        print(f"Classificador: {classificador.__class__.__name__}")
    obter_resultados_matriz_confusao_correlacao(dados, classificador, tamanho_treino_lista_int,\
             repeticoes=repeticoes, titulo=fig_folder, save_fig=save_fig)
