import tkinter as tk
import matplotlib.pyplot as plt
from modelos.dados import Dados
from sklearn.metrics import ConfusionMatrixDisplay
import pandas as pd
import seaborn as sns
import numpy as np
from pathlib import Path
from metricas.metricas import obter_distancia_minkowski_entre_classes
from typing import Dict

def imprime_distancias(distancias:pd.Series):
    """
    Imprime as distâncias em um gráfico.

    Args:
        distancias (pd.Series): Lista com as distâncias.
    """
    plt.plot(distancias)
    
    plt.xlabel('Amostras')
    plt.ylabel('Distâncias à média')
    plt.title('Distância à media por amostra')
    
    plt.show()

def imprime_distribuicao_padronizada_distancias(distancias:pd.Series, ddof:int=0):
    """
    Imprime a distribuição padronizada das distâncias em um gráfico, utilizando
    o escore padrão.

    Args:
        distancias (pd.Series): Lista com as distâncias.
        ddof (int, opcional): Graus de liberdade. Padrão é 0.
    """
    # Calcula a média e o desvio padrão das distâncias
    media = distancias.sum() / (distancias.count()-ddof)
    desvio_padrao = distancias.std(ddof=ddof)

    # Calcula o escore padrão de cada distância
    z_scores = (distancias - media) / desvio_padrao

    sns.kdeplot(z_scores, fill=True, edgecolor='black', common_norm=True, common_grid=True)

    plt.xlabel('Escore Padrão')
    plt.ylabel('Densidade')
    plt.title('Distribuição de Densidade dos Escores Padrão')

    plt.show()

def imprime_distribuicao_distancias(distancias:pd.DataFrame, save_fig=False, caminho:str=""):
    """
    Imprime a matriz com a média de todas as distâncias de cada amostra da classe à media(sinal medio)
      da classe.
    Args:
        distancias (pd.DataFrame): Matriz com as distâncias.
        caminho (str): Caminho para salvar a figura.
        save_fig (bool, opcional): Se True, salva a figura. Padrão é False.
    """
    # Tamanho da imagem em polegadas
    largura_polegadas = 160 / 50.8
    altura_polegadas = 247 / 50.8
    fig, ax = plt.subplots(figsize=(largura_polegadas, altura_polegadas))
    sns.heatmap(distancias, annot=True, fmt=".3g", cmap="Blues", cbar=False, ax=ax, annot_kws={"size": 7})
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=6)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=6)
    fig.tight_layout()
    if save_fig == True:
        caminho = Path(caminho)
        caminho.mkdir(parents=True, exist_ok=True)
        file_name = str("_".join(caminho.parts[:])) + "_distribuicao_distancias.pdf"
        plt.savefig(caminho / file_name, format="pdf", dpi=300)
        plt.close()
    else:
        plt.show()

def imprime_matriz_distancias_classes(matriz_distancias:Dict[str, str], save_fig=False, caminho:str="",\
                                       legendas:Dict[str, int]=None):
    """
    Imprime a matriz de distâncias entre as classes em um gráfico.

    Args:
        matriz_distancias (Dict[str, str]): Matriz de distâncias entre as classes.
        caminho (str): Caminho para salvar a figura.
        save_fig (bool, opcional): Se True, salva a figura. Padrão é False.
        legendas (Dict[str, int], opcional): Dicionário com as legendas das classes. Padrão é None.
    """
    # Tamanho da imagem em polegadas
    largura_polegadas = 200 / 50.8
    fig, ax = plt.subplots(figsize=(largura_polegadas, largura_polegadas))
    sns.heatmap(pd.DataFrame(matriz_distancias).astype(float), annot=True, fmt=".2g", cmap="Blues", \
                cbar=False, ax=ax, annot_kws={"size": 4})
    
    ax.set_xticks(np.arange(0.5, len(legendas.keys()), 1))
    ax.set_yticks(np.arange(0.5, len(legendas.keys()), 1))
    ax.set_yticklabels(legendas.keys(), rotation=0, fontsize=4)
    if legendas is not None:
        ax.set_xticklabels([legendas[label.get_text()] for label in ax.get_yticklabels()], rotation=0,\
                            fontsize=4)
    else:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=-90, fontsize=4)
    fig.tight_layout()

    if save_fig == True:
        caminho = Path(caminho)
        caminho.mkdir(parents=True, exist_ok=True)
        file_name = str("_".join(caminho.parts[:])) + "_matriz_distancias_classes.pdf"
        plt.savefig(caminho / file_name, format="pdf", dpi=300)
        plt.close()
    else:
        plt.show()

def imprime_matriz_confusao_e_relatorio_classificacao(dados:Dados, matriz_confusao, \
                                    relatorio_classificacao, titulo="", save_fig=False):
    """
    Imprime a matriz de confusão e o relatório de classificação em um gráfico único.
    Args:
        dados (Dados): Dados do conjunto de dados.
        matriz_confusao (np.ndarray): Matriz de confusão.
        relatorio_classificacao (Dict): Relatório de classificação.
        titulo (str, opcional): Título do gráfico. Padrão é "".
        save_fig (bool, opcional): Se True, salva a figura. Padrão é False.
    """
    # Tamanho da imagem em polegadas
    largura_polegadas = 220 / 50.8
    fig, ax = plt.subplots(figsize=(largura_polegadas, largura_polegadas))
    sns.heatmap(pd.DataFrame(matriz_confusao, columns=dados.classes_lista, index=dados.classes_lista).astype(float), annot=True, fmt=".3g", cmap="Blues", \
                cbar=False, ax=ax, annot_kws={"size": 3.5})
    ax.set_xticks(np.arange(0.5, dados.num_classes, 1))
    ax.set_yticks(np.arange(0.5, dados.num_classes, 1))
    ax.set_yticklabels(dados.classes_lista, rotation=0, fontsize=3)
    ax.set_xticklabels([dados.legenda[label.get_text()] for label in ax.get_yticklabels()], rotation=0,\
                        fontsize=3)
    fig.tight_layout()

    for text in ax.texts:
        current_text = text.get_text()
        if len(current_text) > 3:
            formatted_text = round(float(current_text), 3)  # Arredonda
            text.set_text(formatted_text)

    caminho = Path(titulo)
    
    if save_fig == True:
        caminho.mkdir(parents=True, exist_ok=True)
        file_name = str("_".join(caminho.parts[1:])) + "_matriz_confusao.pdf"
        plt.savefig(caminho / file_name, format="pdf", dpi=300)
        plt.close()
    else:
        plt.show()

    # Remove a linha 'weighted avg' do relatório de classificação
    relatorio_classificacao.pop('accuracy')
    # Remove a coluna f1-score
    for chave, _ in relatorio_classificacao.items():
        try:
            relatorio_classificacao[chave].pop('f1-score')
        except:
            pass

    # Tamanho da imagem em polegadas
    largura_polegadas = 160 / 50.8
    altura_polegadas = 247 / 50.8

    fig, ax = plt.subplots(figsize=(largura_polegadas, altura_polegadas))
    sns.heatmap(pd.DataFrame(relatorio_classificacao).iloc[:, :].T, annot=True, fmt=".5g", cmap="Blues",\
                 cbar=False, ax=ax, annot_kws={"size": 7})
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=6)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=6)
    fig.tight_layout()

    if save_fig == True:
        file_name = str("_".join(caminho.parts[1:])) + "_relatorio_classificacao.pdf"
        plt.savefig(caminho / file_name, format="pdf", dpi=300)
        plt.close()
    else:
        plt.show()

def mostra_matriz(matriz, label_x=[], label_y=[]):
    """
    Imprime a matriz em uma janela do tkinter.
    """
    janela = tk.Tk()
    frame = tk.Frame(janela)
    frame.pack()
    # Adicionar rótulos dos eixos x
    for i, rotulo in enumerate(label_x):
        label = tk.Label(frame, text=str(rotulo))
        label.grid(row=0, column=i+1, padx=5, pady=5)

    # Adicionar rótulos dos eixos y
    for i, rotulo in enumerate(label_y):
        label = tk.Label(frame, text=str(rotulo))
        label.grid(row=i+1, column=0, padx=5, pady=5)

    # Adicionar cada número da matriz em uma célula
    for i in range(len(matriz)):
        for j in range(len(matriz[i])):
            label = tk.Label(frame, text=str(round(matriz[i][j], 3)))
            label.grid(row=i+1, column=j+1, padx=5, pady=5)
    janela.mainloop()

def imprime_testes_multiplos(matriz, tamanho_treino_lista, k_lista):
    """
    Essa função cria nomes para rotular as linhas e colunas da matriz que será exibida.
    """
    label_y = []
    label_x = []

    # Cria rótulos do eixo y
    for rotulo in k_lista:
        label_y.append("k = "+str(rotulo))
    # Cria rótulos do eixo x
    for value in tamanho_treino_lista:
        # Calcula o número de amostras de cada distribuição se for porcentagem
        label_x.append(str(value)+" amostras")
    # Passa a matriz, os rótulos dos eixos x e y para a função que desenha a matriz numa janela
    mostra_matriz(matriz, label_x, label_y)

def imprimir_contagem_amostras(y_train, y_test):
    """
    Imprime a contagem de amostras em cada classe no conjunto de treino e teste.
    Args:
        y_train (List): Lista com as classes do conjunto de treino.
        y_test (List): Lista com as classes do conjunto de teste.
    Returns:
        None
    """
    train_counts = pd.Series(y_train).value_counts().to_string()
    test_counts = pd.Series(y_test).value_counts().to_string()
    print("Amostras no Conjunto de Treino:\n", train_counts)
    print("Amostras no Conjunto de Teste:\n", test_counts)
