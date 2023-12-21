from sklearn.metrics import classification_report, confusion_matrix
from metricas.dividir_amostras import  treino_regular
from dados.dados import Dados
from visualizacao.visualizacao import imprime_matriz_confusao_e_relatorio_classificacao
from typing import List, Union, Tuple, Dict
from statistics import mode
from pathlib import Path
import tracemalloc
import time

def obter_matriz_confusao_correlacao(dados:Dados, classificador:callable, X_train, X_test, y_train,\
         y_test, repeticoes_classificacao:int=10, titulo:str="", save_fig:bool=False)-> Tuple[int, str]:
    """
        Realiza o treino do classificador para uma distribuição e 
        imprime a matriz de confusão e o relatório de classificação.
        Args:
            dados (Dados): Objeto Dados com os dados a serem utilizados.
            X_train (List): Lista com os dados de treino.
            X_test (List): Lista com os dados de teste.
            y_train (List): Lista com as classes do conjunto de treino.
            y_test (List): Lista com as classes do conjunto de teste.
            tamanho_treino (int): Tamanho do conjunto de treino.
            repeticoes_classificacao (int, opcional): Número de repetições de cada teste. Padrão é 10.
            titulo (str, opcional): Título do gráfico. Padrão é "".
            save_fig (bool, opcional): Se True, salva a figura gerada. Padrão é False.
        
        Returns:
            Tuple[str, str, str]: Lista com o tempo de execução, pico de memória e acurácia obtida.
    """
    tracemalloc.start()
    start_mem, _ = tracemalloc.get_traced_memory()
    start_time = time.time()

    previsoes_acumuladas = []

    # X_train, X_test, y_train, y_test = treino_regular(dados, train_size=tamanho_treino)

    classificador_instancia = classificador()

    for _ in range(repeticoes_classificacao):
        # Treinando o classificador
        classificador_instancia.fit(X_train, y_train)

        # Realizando o teste
        previsoes_acumuladas.append(classificador_instancia.predict(X_test))

    previsao_moda = [mode(preds) for preds in zip(*previsoes_acumuladas)]

    mat_confusao = confusion_matrix(y_test, previsao_moda, normalize='true')
    rel_classificacao = classification_report(y_test, previsao_moda, output_dict=True)

    # imprimir a precisão ponderada arrendodada 5 casas decimais após a vírgula
    # precisao_ponderada = f"{rel_classificacao['weighted avg']['precision']:.5g}"
    # print(f"precisao_ponderada = {precisao_ponderada}")

    end_time = time.time()
    elapsed_time = end_time - start_time

    _, peak = tracemalloc.get_traced_memory()
    
    # # Extrai minutos e segundos
    # minutos, resto = divmod(elapsed_time, 60)

    # # Formata a string
    # tempo_exec = "{:0}:{:05.2f}".format(int(minutos), resto)
    # memoria_maximo = f"{(peak - start_mem) / 10**6:.2f}"

    # tempo_exec = tempo_exec.replace(".", ",")
    # memoria_maximo = memoria_maximo.replace(".", ",")

    # print(f"Tempo de execução: {tempo_exec}")
    # print(f"Pico de memória: {memoria_maximo}")

    imprime_matriz_confusao_e_relatorio_classificacao(dados, mat_confusao, rel_classificacao,\
                                                       titulo=titulo, save_fig=save_fig)
    return elapsed_time, peak - start_mem, rel_classificacao['Acurácia']

def obter_resultados_matriz_confusao_correlacao(dados:Dados, classificador:callable, \
    tamanho_treino_lista:List[int], repeticoes_classificacao:int=10, repeticoes_divisao:int=10,\
    titulo:str="", save_fig:bool=False)-> Dict[str, Dict[str, str]]:
    """
        Obtém os resultados de multiplos testes para as distribuições e os 
        retorna em uma lista.
        Args:
            dados (Dados): Objeto Dados com os dados a serem utilizados.
            classificador (callable): Construtor do classificador a ser utilizado.
            tamanho_treino_lista (List[int]): Lista com o tamanho do conjunto de treino em cada teste.
            repeticoes_classificacao (int, opcional): Número de repetições de cada teste. Padrão é 10.
            repeticoes_divisao (int, opcional): Número de repetições da divisão entre treino e teste 
             para obtenção da médiadas métricas. Padrão é 10.
            titulo (str, opcional): Título do gráfico. Padrão é "".
            save_fig (bool, opcional): Se True, salva a figura gerada. Padrão é False.
        
        Returns:
            Dict[str, Tuple[str, str, str]]: Dicionario com os resultados de cada teste para cada 
             combinação de k e tamanho de amostra.
    """
    result = {}
    for tamanho_treino in tamanho_treino_lista:
        print(f"Amostra de treino = {tamanho_treino}")

        TE_v = []
        P_v = []
        PM_v = []
        for i in range(repeticoes_divisao):
            X_train, X_test, y_train, y_test = treino_regular(dados, train_size=tamanho_treino)
            if i == 0: # imprime apenas uma vez
                TE, PM, P = obter_matriz_confusao_correlacao(dados, classificador, X_train, X_test,\
                 y_train, y_test, repeticoes_classificacao, \
                 titulo=str(Path(titulo) / f"n-{tamanho_treino}"), save_fig=save_fig)
            if i != 0: # imprime apenas uma vez
                TE, PM, P = obter_matriz_confusao_correlacao(dados, classificador, X_train, X_test,\
                 y_train, y_test, repeticoes_classificacao, \
                 titulo=str(Path(titulo) / f"n-{tamanho_treino}"), save_fig=False)
            TE_v.append(TE)
            P_v.append(P)
            PM_v.append(PM)

        # calcula as médias
        TE = sum(TE_v)/len(TE_v)
        P = sum(P_v)/len(P_v)
        PM = sum(PM_v)/len(PM_v)
        
        # Extrai minutos e segundos
        minutos, resto = divmod(TE, 60)

        # Formata a string
        TE = "{:0}:{:05.2f}".format(int(minutos), resto).replace(".", ",")
        PM = f"{PM / 10**6:.2f}".replace(".", ",")
        P = f"{P:.5g}".replace(".", ",")
        
        result[str(tamanho_treino)] = {"TE": TE, "PM": PM, "P": P}
    return result
            
def executar_multiplas_previsoes_correlacao_matriz_confusao(dados:Dados, classificador:callable,\
            tamanho_treino_lista:List[Union[int, float]]=[0.300], repeticoes_classificacao:int=10,\
            repeticoes_divisao:int=10, imprime_legenda:bool=True, save_fig:bool=False,
    fig_folder:str="resultados")-> Dict[str, Dict[str, str]]:
    """
        Executa testes múltiplos com diferentes distribuições de dados para treino e teste.
        Args:
            dados (Dados): Objeto Dados com os dados a serem utilizados.
            classificador (callable): Construtor do classificador a ser utilizado.
            tamanho_treino_lista (List[Union[int, float]], opcional): Lista com o tamanho do 
            conjunto de treino em cada teste. Pode ser uma lista de inteiros ou de floats entre 0 e 1.
            repeticoes_classificacao (int, opcional): Número de repetições de cada teste. Padrão é 10.
            repeticoes_divisao (int, opcional): Número de repetições da divisão entre treino e teste 
            para obtenção da média das métricas. Padrão é 10.
            imprime_legenda (bool, opcional): Se True, imprime o nome do classificador e o tamanho
            do conjunto de treino. Padrão é True.
            save_fig (bool, opcional): Se True, salva a figura gerada. Padrão é False.
        Returns:
            Dict[str, Tuple[str, str, str]]: Dict com os resultados de cada teste para cada combinação
              de k e tamanho de amostra.
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
        print(f"Classificador: {classificador.__name__}")
    return obter_resultados_matriz_confusao_correlacao(dados, classificador, tamanho_treino_lista_int,\
             repeticoes_classificacao=repeticoes_classificacao, repeticoes_divisao=repeticoes_divisao,\
                  titulo=fig_folder, save_fig=save_fig)
