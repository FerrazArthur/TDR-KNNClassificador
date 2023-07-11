from decimal import Decimal
from math import *
import numpy as np
import tkinter as tk
import matplotlib.pyplot as plt

def p_root(value, root):
    root_value = 1 / float(root)
    return round(Decimal(value) ** Decimal(root_value), 3)

def minkowski_distance(x, y, p_value):
    # pass the p_root function to calculate
    # all the value of vector parallelly
    return (p_root(sum(pow(abs(a-b), p_value) for a, b in zip(x, y)), p_value))

def calcularMedias(amostras):
    '''
    Recebe um vetor numpy onde cada elemento é uma coleção de exemplos de um mesmo sinal
    e retorna um vetor numpy com as médias desses sinais
    '''
    medias = []
    for n in range(np.size(amostras, 0)):
        #para cada classe, tire uma média e armazene no vetor medias
        medias.append(np.zeros(np.size(amostras[n], 1)))
        for i in range(np.size(amostras[n], 1)):
            for j in range(np.size(amostras[n], 0)):
                medias[-1][i] += amostras[n][j][i]
            medias[-1][i] /= float(np.size(amostras[n], 0))
    return np.array(medias)

#calcular a distância de minkowski entre os sinais e a media encontrada

def calcularDist(amostras, medias):
    '''
    Recebe um vetor numpy amostras onde cada elemento é uma coleção de exemplos de um mesmo sinal
    e um vetor numpy medias, que armazena a média para cada classe e retorna um vetor numpy
    contendo a distância entre cada sinal em amostras e a media da sua classe
    '''
    distancias = []
    for i in range(np.size(amostras, 0)):
        distancias.append([])
        for j in range(np.size(amostras[i], 0)):
            distancias[i].append(minkowski_distance(amostras[i][j], medias[i], 2))
    return np.array(distancias)

def imprimeMaiorDesvio(amostras, distancias, medias, imprime=True):
    '''
    Encontra a amostra, entre todas as classes, que mais desvia da média(tem maior distância) e 
    imprime seu sinal na tela, além de informar sua posição na amostra
    Returna o índice da classe que apresentou maior desvio, o indice da amostra nessa classe e 
    o valor do desvio
    '''
    MaiorIndex = []
    for classe in distancias:
        classe = list(classe)
        MaiorIndex.append(classe.index(max(classe)))

    maiorDesvio = 0, 0, 0#classe, amostra na classe, valor
    for i in range(np.size(amostras, 0)):
        if imprime == True:
            print('A maior distancia relativa à media na Classe', i, ' é', distancias[i][MaiorIndex[i]])
        if distancias[i][MaiorIndex[i]] > maiorDesvio[2]:
            maiorDesvio = i, MaiorIndex[i], distancias[i][MaiorIndex[i]]
    if imprime == True:
        print('A maior variação encontrada foi na amostra', maiorDesvio[1], 'da classe', maiorDesvio[0], ', cuja variância encontrada foi de', maiorDesvio[2])
        plot2Signal(amostras[maiorDesvio[0]][maiorDesvio[1]], 'mais Variado', medias[maiorDesvio[0]], 'media')
    return maiorDesvio

def imprimeDesvioPadrao(amostras, dist):
    '''
    Calcula e imprime o desvio padrao das distâncias de cada amostra à amostra media da classe, pra cada classe.
    '''
    distMedia = []
    for i in range(np.size(amostras, 0)):#para cada classe
        distMedia.append(0)
        for j in range(np.size(amostras[i], 0)):
            distMedia[-1] += dist[i][j]
        distMedia[-1] /= np.size(dist[i], 0)
    
    desvioPadrao = []
    for i in range(np.size(amostras, 0)):#para cada classe
        desvioPadrao.append(0)
        for j in range(np.size(amostras[i], 0)):
            desvioPadrao[-1] += pow(distMedia[i] - dist[i][j], 2)
        desvioPadrao[-1] /= np.size(dist[i], 0)
        desvioPadrao[-1] = sqrt(desvioPadrao[-1])
    
    for i in range(np.size(amostras, 0)):#para cada classe
        print("Distância média da classe ", i, " é igual a ", round(distMedia[i], 4)," e apresenta um desvio padrão à média de ", round(desvioPadrao[i], 4))

def imprimeDistanciaEntreClasses(amostras, medias):
    '''
    Calcula e imprime a distância euclidiana entre as médias de cada classe.
    '''
    menor = np.infty
    maior = 0
    dist = []
    for i in range(np.size(amostras, 0)):
        dist.append([])
        for j in range(np.size(amostras, 0)):
            if i == j:#evitar calcular a distância entre a mesma classe
                dist[-1].append(0)
            else:
                dist[-1].append(minkowski_distance(medias[i], medias[j], 2))
                #Mantém a menor distância entre classes calculada até então
                if dist[-1][-1] < menor:
                    menor = dist[-1][-1]
            #Mantém a maior distância calculada até então
            if dist[-1][-1] > maior:
                maior = dist[-1][-1]
    print("A menor distância entre distintas classes é ", round(menor, 4), " e a maior é ", round(maior, 4))
    labels = []
    for i in range(np.size(amostras, 0)):
        labels.append("Classe " + str(i))
    showMat(dist,labels, labels)

def mostrarVariancia(amostras):
    #amostras, arquivos = getAmostra(path)

    medias = calcularMedias(amostras)
    dist = calcularDist(amostras, medias)
    #imprimeMaiorDesvio(amostras, dist, medias)
    imprimeDesvioPadrao(amostras, dist)
    imprimeDistanciaEntreClasses(amostras, medias)

def plot2Signal(signal1, nome1, signal2, nome2):
    fig, ax = plt.subplots(2,1, sharex=True)
    ax[0].plot(range(len(signal1)), signal1)
    ax[0].set_xlabel(nome1)
    ax[0].set_ylabel("amplitude")
    ax[0].grid(True)
    ax[1].plot(range(len(signal1)), signal2)
    ax[1].set_xlabel(nome2)
    ax[1].set_ylabel("amplitude")
    ax[1].grid(True)
    fig.tight_layout()
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
