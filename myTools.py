
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import numpy as np
from matVetorial import *
from csvutils import *

def treinoRegular(rawAmostras, test_size=0.97, random_state = 0):
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

def treinoMedia(rawAmostras, corte=18, embaralhar=True, random_state = 1):
    """
        Retorna um conjunto de treino que contém apenas um exemplo por classe e esse é uma média dos primeiros 'corte' elementos de cada classe.
        Retorna o restante como conjunto de testes
    """
    if corte >= np.size(rawAmostras[0][0], 0):
        print("o corte é grande demais para os dados disponíveis.")
        return
    
    #Calculando o conjunto de treino
    newRawAmostras = []
    for i in range(np.size(rawAmostras, 0)):
        #utilizando poucos elementos da classe para tirar a média
        if embaralhar == True:
            newRawAmostras.append(shuffle(rawAmostras[i], random_state = random_state)[:])
        else:
            newRawAmostras.append(rawAmostras[i][:])

    medias = calcularMedias(newRawAmostras[:corte])
    X_train = []
    y_train = []
    for i in range(np.size(medias, 0)):
        X_train.append(medias[i])
        y_train.append(i)
    X_train, y_train = shuffle(X_train, y_train)

    #Conjunto de testes composto pelo restante
    newAmostras = []
    newNomes = []
    #para cada classe
    for i in range(np.size(newRawAmostras, 0)):
        #pega cada amostra fora do corte e guarda num vetor newAmostra, com sua classe salva como um label no vetor newNomes
        for amostra in newRawAmostras[i][corte:]:
            newAmostras.append(amostra)
            newNomes.append(i)
    #embaralha os labels (e as amostras associadas, junto) pra que não estejam em sequência mas mantenham sua relação() tipo y_test[i] <é a classe de>-> x_test[i], para todo i válido)
    X_test, y_test = shuffle(newAmostras, newNomes)

    return X_train, X_test, y_train, y_test

