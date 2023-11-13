import tkinter as tk
import matplotlib.pyplot as plt
from modelos.dados import Dados
from sklearn.metrics import ConfusionMatrixDisplay
import pandas as pd
import seaborn as sns

def imprime_matriz_confusao_e_relatorio_classificacao(dados:Dados, matriz_confusao, relatorio_classificacao, acuracia=None, titulo="", save_fig=False):
    """
    Imprime a matriz de confusão e o relatório de classificação em um 
    gráfico único.
    """
    if acuracia is not None:
        print("Acurácia:", acuracia)

    cmd = ConfusionMatrixDisplay(confusion_matrix=matriz_confusao, display_labels=dados.classes_lista)
    plt.figure(figsize=(10, 10), dpi=300)
    fig, ax = plt.subplots(figsize=(16,16))
    cmd.plot(values_format=".2g", ax=ax, xticks_rotation=45)

    if save_fig == True:
        plt.savefig("figs/matrizconfusao_"+titulo+".png")
        plt.close()
    else:
        plt.show()
    
    sns.heatmap(pd.DataFrame(relatorio_classificacao).iloc[:-1, :].T, annot=True)
    if save_fig == True:
        plt.savefig("figs/relatorioclassificacao_"+titulo+".png")
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