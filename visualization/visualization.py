from math import floor
import tkinter as tk
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score
import pandas as pd
import seaborn as sns

def imprime_conf_mat_e_class_rep(conf_mat, classification_rep, legenda=[], acuracia=None):
    """
    Imprime a matriz de confusão e o relatório de classificação em um 
    plot único
    """

    # if len(legenda) > 0:
    #     print("legenda: ")
    #     for i in range(len(legenda)):
    #         print(i+1, ": ",legenda[i])
    if acuracia != None:
        print("Accuracy:",acuracia)
        
    cmd = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=legenda)
    cmd.plot(values_format="d")

    plt.show()
    # fig, ax = plt.subplots(1, 1)
    # cmd = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=legenda)
    # cmd.plot(values_format="d", ax=ax[0])
    sns.heatmap(pd.DataFrame(classification_rep).iloc[:-1, :].T, annot=True)
    plt.show()


def showMat(matriz, labelx=[], labely=[]):
    """
    Imprime a matriz em uma janela do tkinter
    """
    janela = tk.Tk()
    frame = tk.Frame(janela)
    frame.pack()
    # Adicionar rótulos dos eixos x
    for i, rotulo in enumerate(labelx):
        label = tk.Label(frame, text=str(rotulo))
        label.grid(row=0, column=i+1, padx=5, pady=5)

    # Adicionar rótulos dos eixos y
    for i, rotulo in enumerate(labely):
        label = tk.Label(frame, text=str(rotulo))
        label.grid(row=i+1, column=0, padx=5, pady=5)

    # Adicionar cada número da matriz em uma célula
    for i in range(len(matriz)):
        for j in range(len(matriz[i])):
            label = tk.Label(frame, text=str(round(matriz[i][j], 3)))
            label.grid(row=i+1, column=j+1, padx=5, pady=5)
    janela.mainloop()

def imprime_testes_multiplos(matriz, distribrange, rangek):
    """
    Essa função cria nomes para rotular as linhas e colunas da matriz que será exibida.
    """
    labely = []
    labelx = []

    # Cria rótulos do eixo y
    for rotulo in rangek:
        labely.append("k = "+str(rotulo))
    #cria rótulos do eixo x
    for value in distribrange:
        #calcula o número de amostras de cada distribuição
        labelx.append(str(floor(600 * (1.0-value)))+" amostras")
    # Passa a matriz, os rótulos dos eixos x e y para a função que desenha a matriz numa janela
    showMat(matriz, labelx, labely)

def imprime_testes_multiplos_train_explicit(matriz, trainrange, rangek):
    """
    Essa função cria nomes para rotular as linhas e colunas da matriz que será exibida.
    """
    labely = []
    labelx = []

    # Cria rótulos do eixo y
    for rotulo in rangek:
        labely.append("k = "+str(rotulo))
    #cria rótulos do eixo x
    for value in trainrange:
        #calcula o número de amostras de cada distribuição
        labelx.append(str(value)+" amostras")
    # Passa a matriz, os rótulos dos eixos x e y para a função que desenha a matriz numa janela
    showMat(matriz, labelx, labely)