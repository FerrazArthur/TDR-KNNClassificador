from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils import shuffle
from myTools import *

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
    return train_test_split(amostras,nomes, test_size=test_size, random_state = random_state)

def treinoMedia(rawAmostras, corte=25, embaralhar=True):
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
    for i in range(np.size(rawAmostras, 0)):
        for j in range(corte, np.size(rawAmostras[i], 0)):
            newAmostras.append(rawAmostras[i][j])
            newNomes.append(i)
    #x, X_test, y, y_test = train_test_split(newAmostras, newNomes, test_size=0.999, random_state = 1)
    X_test, y_test = shuffle(newAmostras, newNomes)

    return X_train, X_test, y_train, y_test 

def testarKnn(path,dividirAmostra, k=1):
    """
        Busca os dados em path, realiza a distribuição dos dados entre treino/teste com base na função dividirAmostra com os parametros args,
        cria um classificador KNN com k = k e realiza a plotagem dos resultados do teste
    """
    #rawAmostras é um vetor de dim 3, onde dim 1 é cada arquivo csv, 
    #dim 2 contém as amostras desse arquivo e dim 3 são os valores dos sinais
    
    #arquivos é um vetor com o nome de cada arquivo na ordem como serão usados
    #pelo algoritmo


    rawAmostras, arquivos = getAmostra(path)

    #mostrarVariancia(rawAmostras) #calcula e plota variância dos dados
    #sei que há um valor errado na amostra 325 da classe 5, cujo desvio é de 41. então o "removerei"
    rawAmostras[5][325] = rawAmostras[5][324]


    #-----------------------------
    #Conjunto de treino composto por apenas um exemplo para cada classe, onde esse exemplo é uma média do conjunto inteiro

    knn_class = KNeighborsClassifier(n_neighbors=1)

    X_train, X_test, y_train, y_test = dividirAmostra(rawAmostras)
    #treinando o classificador
    knn_class.fit(X_train, y_train)
    print("tamanho da amostra no treino: ", len(X_train))
    print("tamanho da amostra no teste: ", len(X_test))
    
    #realizando o teste
    ypred=knn_class.predict(X_test)
    result = confusion_matrix(y_test, ypred)
    
    #imprimindo o resultado
    print("legenda: ")
    for i in range(len(arquivos)):
        print(i, ": ",arquivos[i])
    print("Confusion Matrix:")
    print(result)
    result1 = classification_report(y_test, ypred)
    print("Classification Report:")
    print(result1)
    result2 = accuracy_score(y_test,ypred)
    print("Accuracy:",result2)

testarKnn('dados',treinoRegular)
testarKnn('dados',treinoMedia)

