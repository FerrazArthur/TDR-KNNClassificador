from math import *
from decimal import Decimal
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import re
from minkowski import *


def p_root(value, root):
    root_value = 1 / float(root)
    return round(Decimal(value) ** Decimal(root_value), 3)

def minkowski_distance(x, y, p_value):
    # pass the p_root function to calculate
    # all the value of vector parallelly
    return (p_root(sum(pow(abs(a-b), p_value) for a, b in zip(x, y)), p_value))

def loadCSVMatrix(nomeArquivo):
    '''
    Input: nome do arquivo csv
    output: retorna um array numpy com os valores presentes no arquivo csv
    '''
    with open(nomeArquivo, "r",) as arquivo:
        dados=arquivo.read()
    linhas=dados.split('\n')
    tabela=[]
    for linha in linhas[:-1]:#ignora a ultima linha
        tabela.append(np.array(linha.split(';')))
    tabela=np.asarray(tabela, dtype=float)
    return tabela

def getAmostra(path):
    '''
    recebe uma string caminho para onde estão os arquivos csv
    e retorna um vetor numpy, onde cada elemento é um arquivo diferente
    '''
    #dados = Path(path).glob('*(.csv|.CSV)')
    dados = sorted(Path(path).glob('*'))
    amostras = []
    classes = []
    print()
    for csv in dados:
        amostras.append(loadCSVMatrix(csv))
        #armazenando os nomes sem o caminho
        classes.append(str(csv).strip(path+'/').strip('.CSV').strip('.csv'))
        #removendo a ordem (o número que acompanha o inicio dos arquivos)
        classes[-1] = re.split(r'^[0-9]+-', classes[-1])[1]

    return np.array(amostras), np.array(classes)
#def getAmostra(path):
#    '''
#    recebe uma string caminho para onde estão os arquivos csv
#    e retorna um vetor numpy, onde cada elemento é um arquivo diferente
#    '''
#    #dados = Path(path).glob('*(.csv|.CSV)')
#    dados = Path(path).glob('*')
#    amostras = []
#    classes = []
#    for csv in dados:
#        amostras.append(loadCSVMatrix(csv))
#        classes.append(str(csv).strip(path+'/').strip('.CSV').strip('.csv'))
#
#    return np.array(amostras), np.array(classes)

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
    e um vetor numpy medias, que armazena as médias para cada classe e retorna um vetor numpy
    contendo a distância entre cada sinal em amostras e a media da sua classe
    '''
    distancias = []
    for i in range(np.size(amostras, 0)):
        distancias.append([])
        for j in range(np.size(amostras[i], 0)):
            distancias[i].append(minkowski_distance(amostras[i][j], medias[i], 2))
    return np.array(distancias)

def EncontraMaioresDesvios(amostras, distancias, medias):
    '''
    Encontra a amostra, para cada classe, que mais desvia da média(tem maior distância) e a 
    Plota seu sinal na tela, além de informar sua posição na amostra
    Returna o índice da classe que apresentou maior desvio e a amostra que desvia mais
    '''
    MaiorIndex = []
    for classe in distancias:
        classe = list(classe)
        MaiorIndex.append(classe.index(max(classe)))
    maiorDesvio = 0, 0, 0#classe, amostra na classe, valor
    for i in range(np.size(amostras, 0)):
        print("Para classe ", i," : sinal mais variado x media:")
        print('Possui distancia relativa de: ', distancias[i][MaiorIndex[i]])
        if distancias[i][MaiorIndex[i]] > maiorDesvio[2]:
            maiorDesvio = i, MaiorIndex[i], distancias[i][MaiorIndex[i]]
        plot2Signal(amostras[i][MaiorIndex[i]], 'mais Variado', medias[i], 'media')
    return maiorDesvio

def MaiorDesvio(amostras, distancias, medias):
    '''
    Encontra a amostra, entre todas as classes, que mais desvia da média(tem maior distância)
    Returna o índice da classe que apresentou maior desvio, o indice da amostra nessa classe e 
    o valor do desvio
    '''
    MaiorIndex = []
    for classe in distancias:
        classe = list(classe)
        MaiorIndex.append(classe.index(max(classe)))
    maiorDesvio = 0, 0, 0#classe, amostra na classe, valor
    for i in range(np.size(amostras, 0)):
        if distancias[i][MaiorIndex[i]] > maiorDesvio[2]:
            maiorDesvio = i, MaiorIndex[i], distancias[i][MaiorIndex[i]]
    return maiorDesvio

def PlotaMaiorDesvio(amostras, distancias, medias):
    '''
    Encontra a amostra, entre todas as classes, que mais desvia da média(tem maior distância) e 
    Plota seu sinal na tela, além de informar sua posição na amostra
    Returna o índice da classe que apresentou maior desvio, o indice da amostra nessa classe e 
    o valor do desvio
    '''
    MaiorIndex = []
    for classe in distancias:
        classe = list(classe)
        MaiorIndex.append(classe.index(max(classe)))


    maiorDesvio = 0, 0, 0#classe, amostra na classe, valor
    for i in range(np.size(amostras, 0)):
        print("Classe", i,'possui distancia relativa de:', distancias[i][MaiorIndex[i]])
        if distancias[i][MaiorIndex[i]] > maiorDesvio[2]:
            maiorDesvio = i, MaiorIndex[i], distancias[i][MaiorIndex[i]]
    print('A maior variação encontrada foi na amostra', maiorDesvio[1], 'da classe', maiorDesvio[0], ', cuja variância encontrada foi de', maiorDesvio[2])
    plot2Signal(amostras[maiorDesvio[0]][maiorDesvio[1]], 'mais Variado', medias[maiorDesvio[0]], 'media')
    return maiorDesvio

def plot2Signal(signal1, nome1, signal2, nome2):
    fig, ax = plt.subplots(2,1, sharex=True)
    ax[0].stem(range(len(signal1)), signal1, linefmt='b-')
    ax[0].set_xlabel(nome1)
    ax[0].set_ylabel("amplitude")
    ax[0].grid(True)
    ax[1].stem(range(len(signal1)), signal2, linefmt='b-')
    ax[1].set_xlabel(nome2)
    ax[1].set_ylabel("amplitude")
    ax[1].grid(True)
    fig.tight_layout()
    plt.show()

def mostrarVariancia(amostras):
    #amostras, arquivos = getAmostra(path)

    medias = calcularMedias(amostras)
    dist = calcularDist(amostras, medias)
    PlotaMaiorDesvio(amostras, dist, medias)
