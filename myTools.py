from math import *
from decimal import Decimal
from pathlib import Path
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import re

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

def writeCSV(x, y, nomeArquivo):
    '''
    Input: conjunto de amostras, label das amostras, nome do arquivo csv
    output: True se ocorreu algum erro, False do contrário
    '''
    try:
        with open(nomeArquivo, "w",) as arquivo:
            #escreve cabeçalho CLASSE, VALORES
            arquivo.write("CLASSE, ")
            #escreve valor pra cada dimensão 
            for i in range(np.size(x[0], 0)):
                arquivo.write("VALOR,")
            arquivo.write('\n')
            for i in range(np.size(x, 0)):
                arquivo.write(str(y[i]))
                for j in range(np.size(x[i], 0)):
                    arquivo.write(','+str(x[i][j]))
                arquivo.write('\n')
        return False
    except Exception as e:
        print('Erro ao escrever em arquivo "'+nomeArquivo+'": '+str(e))
        return True

def getAmostra(path):
    '''
    Input: Recebe uma string caminho para onde estão os arquivos csv

    Output:amostras, um vetor de dim 3, onde dim 1 é cada arquivo csv, 
    dim 2 contém as amostras desse arquivo e dim 3 são os valores dos sinais
    e classes é um vetor com o nome de cada arquivo na ordem.
    '''
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
        print('A maior distancia relativa à media na Classe', i, ' é', distancias[i][MaiorIndex[i]])
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

def PlotaDesvioPadrao(amostras, dist, medias):
    #para calcular o desvio padrao por classe:
    #calcula média das distancias à media na classe
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
        print("Desvio padrao da classe ", i, " é igual a ", desvioPadrao[i])

def mostrarVariancia(amostras):
    #amostras, arquivos = getAmostra(path)

    medias = calcularMedias(amostras)
    dist = calcularDist(amostras, medias)
    PlotaMaiorDesvio(amostras, dist, medias)
    PlotaDesvioPadrao(amostras, dist, medias)


def treinoRegular(rawAmostras, test_size=0.97, random_state = 1):
    """
        Retorna o output método scipy regular de dividir as amostras em treino e teste.
    """
    #amostras é o vetor 2dim que concatena todas as amostras, eliminando a dimensão
    #que separa os dados por arquivo
    #nomes é o vetor que classifica cada elemento de amostras com sua respectiva classe
    #utilizando indices entre 0 e 8
    #concatenando os valores de rawAmostras para um formato 2dim
    amostras = []
    nomes = []
    for i in range(np.size(rawAmostras, 0)):
        for sinal in rawAmostras[i]:
            amostras.append(sinal)
            nomes.append(i)

    #aqui o algoritmo train_test_split randomiza e distribui a amostra entre dois conjuntos
    #o de teste e o de treino.
    return train_test_split(amostras,nomes, test_size=test_size, random_state = random_state, stratify=nomes)

def treinoMedia(rawAmostras, corte=20, embaralhar=True):
    """
        Retorna um conjunto de treino que contém apenas um exemplo por classe e esse é uma média dos primeiros 'corte' elementos de cada classe.
        Retorna o restante como conjunto de testes
    """
    #Calculando o conjunto de treino
    newRawAmostras = []
    for i in range(np.size(rawAmostras, 0)):
        #utilizando poucos elementos da classe para tirar a média
        if embaralhar == True:
            newRawAmostras.append(shuffle(rawAmostras[i])[:corte])
        else:
            newRawAmostras.append(rawAmostras[i][:corte])
    medias = calcularMedias(newRawAmostras)
    X_train = []
    y_train = []
    for i in range(np.size(medias, 0)):
        X_train.append(medias[i])
        y_train.append(i)


    #Conjunto de testes composto pelo restante
    newAmostras = []
    newNomes = []
    #para cada classe
    for i in range(np.size(rawAmostras, 0)):
        #embaralha ou não o conjunto de amostrs de teste, antes de colocar os labels
        if embaralhar == True:
            amostrasClasse = shuffle(rawAmostras[i][corte:])
        else:
            amostrasClasse = rawAmostras[i][corte:]
        #pega cada amostra fora do corte e guarda num vetor newAmostra, com sua classe salva como um label no vetor newNomes
        for classe in amostrasClasse:
            newAmostras.append(classe)
            newNomes.append(i)
    #embaralha os labels (e as amostras associadas, junto) pra que não estejam em sequência mas mantenham sua relação() tipo y_test[i] <é a classe de>-> x_test[i], para todo i válido)
    X_test, y_test = shuffle(newAmostras, newNomes)

    return X_train, X_test, y_train, y_test

