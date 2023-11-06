from math import floor
import tkinter as tk

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

def imprimeTestesMultiplos(matriz, distribrange, rangek):
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

def imprimeTestesMultiplos_train_explicit(matriz, trainrange, rangek):
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