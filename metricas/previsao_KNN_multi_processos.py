from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from metricas.dividir_amostras import treino_media, treino_regular
from dados.dados import Dados
from visualizacao.visualizacao import imprime_matriz_confusao_e_relatorio_classificacao
from typing import List, Union, Tuple, Dict
from statistics import mode
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from functools import partial

def knn_multiprocess(process_id:int, dados:Dados, divisao_treino:callable, k:int, tamanho_treino:int,\
     repeticoes_classificacao:int, titulo:str, save_fig:bool):
    X_train, X_test, y_train, y_test = divisao_treino(dados, train_size=tamanho_treino)
    if process_id == 0: # imprime apenas uma vez
        P = obter_matriz_confusao_KNN(dados, X_train, X_test, y_train, y_test, k, repeticoes_classificacao,
                titulo=str(Path(titulo) / f"k-{k}" / f"n-{tamanho_treino}"), save_fig=save_fig)
    if process_id != 0: # imprime apenas uma vez
        P = obter_matriz_confusao_KNN(dados, X_train, X_test, y_train, y_test, k, repeticoes_classificacao,
                titulo=str(Path(titulo) / f"k-{k}" / f"n-{tamanho_treino}"), save_fig=False)
    return P

def obter_matriz_confusao_KNN(dados:Dados, X_train, X_test, y_train, y_test, k:int=1, \
        repeticoes_classificacao:int=10, titulo:str="", save_fig:bool=False)-> Tuple[int, float]:
    """
        Realiza o treino do classificador para uma distribuição e para um valor de k
        e imprime a matriz de confusão e o relatório de classificação.
        Args:
            dados (Dados): Objeto Dados com os dados a serem utilizados.
            X_train (List): Lista com os dados de treino.
            X_test (List): Lista com os dados de teste.
            y_train (List): Lista com as classes do conjunto de treino.
            y_test (List): Lista com as classes do conjunto de teste.
            k (int, opcional): Número de vizinhos a serem considerados. Padrão é 1.
            repeticoes_classificacao (int, opcional): Número de repetições de cada teste. Padrão é 10.
            titulo (str, opcional): Título do gráfico. Padrão é "".
            save_fig (bool, opcional): Se True, salva a figura gerada. Padrão é False.
        Returns:
            str: acurácia obtida.
    """

    previsoes_acumuladas = []

    knn_class = KNeighborsClassifier(n_neighbors=k, weights='distance', algorithm='brute')

    for _ in range(repeticoes_classificacao):
        # Treinando o classificador
        knn_class.fit(X_train, y_train)

        # Realizando o teste
        previsoes_acumuladas.append(knn_class.predict(X_test))

    previsao_moda = [mode(preds) for preds in zip(*previsoes_acumuladas)]

    mat_confusao = confusion_matrix(y_test, previsao_moda, normalize='true')
    rel_classificacao = classification_report(y_test, previsao_moda, output_dict=True)

    # imprimir a precisão ponderada arrendodada 5 casas decimais após a vírgula
    # precisao_ponderada = f"{rel_classificacao['weighted avg']['precision']:.5g}"
    # print(f"precisao_ponderada = {precisao_ponderada}")
    
    imprime_matriz_confusao_e_relatorio_classificacao(dados, mat_confusao, rel_classificacao,\
                                                   titulo=titulo, save_fig=save_fig)
    return rel_classificacao['Acurácia']

def obter_resultados_matriz_confusao_KNN(dados:Dados, divisao_treino, k_lista:List[int],\
        tamanho_treino_lista:List[int], repeticoes_classificacao:int=10, repeticoes_divisao:int=10,\
     titulo:str="", save_fig:bool=False, max_workers:int=10)-> Dict[str, Dict[str, Dict[str, str]]]:
    """
        Obtém os resultados de multiplos testes para as distribuições e valores de k fornecidos e os 
        retorna em uma lista.
        Args:
            dados (Dados): Objeto Dados com os dados a serem utilizados.
            divisao_treino (callable): Função de divisão de treino a ser utilizada.
            k_lista (List[int]): Lista com os valores de k a serem testados.
            tamanho_treino_lista (List[int]): Lista com o tamanho do conjunto de treino em cada teste.
            repeticoes_classificacao (int, opcional): Número de repetições de cada teste. Padrão é 10.
            repeticoes_divisao (int, opcional): Número de repetições da divisão entre treino e teste 
             para obtenção da médiadas métricas. Padrão é 10.
            titulo (str, opcional): Título do gráfico. Padrão é "".
            save_fig (bool, opcional): Se True, salva a figura gerada. Padrão é False.
            max_workers (int, opcional): Número máximo de processos a serem utilizados. Padrão é 10.

        Returns:
            Dict[str, Dict[str, str]]: Dicionario com os resultados de cada teste 
            para cada combinação de k e tamanho de amostra.
    """
    # Realiza o treino do classificador para cada distribuição e para cada valor de k
    result_list = {}
    for k in k_lista:
        print(f"k = {k}")
        tam_treino = {}
        for tamanho_treino in tamanho_treino_lista:
            print(f"Amostra de treino = {tamanho_treino}")

            resultados = []
            # Usar ProcessPoolExecutor para executar a função em vários processos
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Mapear a função para cada iteração do loop
                resultados = list(executor.map(partial(knn_multiprocess, dados=dados, \
                 divisao_treino=divisao_treino, k=k, tamanho_treino=tamanho_treino, \
                 repeticoes_classificacao=repeticoes_classificacao, titulo=titulo, \
                 save_fig=save_fig), range(repeticoes_divisao)))

            # Extrair resultados individuais
            P_v = [result for result in resultados]

            # calcula as médias
            P = sum(P_v)/len(P_v)
            # Formata a string
            P = f"{P:.5g}".replace(".", ",")

            tam_treino[str(tamanho_treino)] = {"P": P}
        
        result_list[f"k_{k}"] = tam_treino
    return result_list

def executar_multiplas_previsoes_KNN_matriz_confusao(dados:Dados, k_lista:List[int]=[1, 3],\
     tamanho_treino_lista:List[Union[int, float]]=[0.300], repeticoes_classificacao:int=10, \
     repeticoes_divisao:int=10, treinos_lista:List[callable]=[treino_regular, treino_media], \
    imprime_legenda:bool=True, save_fig:bool=False,\
    fig_folder:str="resultados", max_workers:int=10)-> Dict[str, Dict[str, Dict[str, Dict[str, str]]]]:
    """
        Executa testes múltiplos com diferentes valores de k e diferentes distribuições de 
        dados de treino e teste.
        Args:
            dados (Dados): Objeto Dados com os dados a serem utilizados.
            k_lista (List[int], opcional): Lista com os valores de k a serem testados. Padrão é [1, 3].
            tamanho_treino_lista (List[Union[int, float]], opcional): Lista com o tamanho do 
            conjunto de treino em cada teste. Pode ser uma lista de inteiros ou de floats entre 0 e 1.
            repeticoes_classificacao (int, opcional): Número de repetições de cada teste. Padrão é 10.
            repeticoes_divisao (int, opcional): Número de repetições da divisão entre treino e teste 
            para obtenção da média das métricas. Padrão é 10.
            treinos_lista (List[callable], opcional): Lista com as funções de treino a serem utilizadas.
            Padrão é [treino_regular, treino_media].
            imprime_legenda (bool, opcional): Se True, imprime o nome do classificador e o tamanho
            do conjunto de treino. Padrão é True.
            save_fig (bool, opcional): Se True, salva a figura gerada. Padrão é False.
            max_workers (int, opcional): Número máximo de processos a serem utilizados. Padrão é 10.

        Returns:
            Dict[str, Dict[str, Dict[str, Dict[str, str]]]]: Dicionario com os resultados de cada teste
              para cada combinação de k e tamanho de amostra.
    """
    # Realiza o treino do classificador para cada distribuição e para cada valor de k

    # Garante que tenhamos valores absolutos para o tamanho do treino, não em porcentagem
    tamanho_treino_lista_int = []
    for tamanho_treino in tamanho_treino_lista:
        if isinstance(tamanho_treino, float) and tamanho_treino < 1.0:
            tamanho_treino_lista_int.append(round(tamanho_treino*dados.num_amostras))
        else:
            tamanho_treino_lista_int.append(round(tamanho_treino))
    treino_dict = {}
    for treino in treinos_lista:
        if (imprime_legenda == True):
            print(f"Treino: {treino.__name__}")
        
        treino_dict[treino.__name__] = obter_resultados_matriz_confusao_KNN(dados, treino, k_lista,\
         tamanho_treino_lista_int, repeticoes_classificacao=repeticoes_classificacao,\
         repeticoes_divisao=repeticoes_divisao, \
         titulo=str(Path(fig_folder) /treino.__name__), save_fig=save_fig,\
         max_workers=max_workers)
        
    return treino_dict
