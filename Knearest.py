from myTools import *

def testarKnn(path,dividirAmostra, test_size=0.994, k=1, legenda=True, amostra=True, 
              accuracy=True, plotar=True, random_state = 0):
    """
        Busca os dados em path, realiza a distribuição dos dados entre treino/teste com base na função 
            dividirAmostra com os parametros args,
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
    mostrarVariancia(rawAmostras)
    return
    #-----------------------------
    #Conjunto de treino composto por apenas um exemplo para cada classe, onde esse exemplo é uma média do 
    #   conjunto inteiro

    knn_class = KNeighborsClassifier(n_neighbors=k)

    X_train, X_test, y_train, y_test = dividirAmostra(rawAmostras, test_size=test_size,
                                                      random_state=random_state)

    #writeCSV(X_train, y_train, "xyTrain.csv")
    #treinando o classificador
    knn_class.fit(X_train, y_train)
    if amostra == True:
        print("tamanho da amostra no treino: ", len(X_train))
        print("tamanho da amostra no teste: ", len(X_test))

    #realizando o teste
    ypred=knn_class.predict(X_test)

    if plotar == True:
        result = confusion_matrix(y_test, ypred)
        
        #imprimindo o resultado

        fig, ax = plt.subplots(2, 1)

        if legenda == True:
            print("legenda: ")
            for i in range(len(arquivos)):
                print(i, ": ",arquivos[i])
        
        cmd = ConfusionMatrixDisplay(confusion_matrix=result)
        cmd.plot(values_format="d", ax=ax[0])
        result1 = classification_report(y_test, ypred, output_dict=True)
        sns.heatmap(pd.DataFrame(result1).iloc[:-1, :].T, annot=True, ax=ax[1])
        plt.show()

    result2 = accuracy_score(y_test,ypred)
    if accuracy == True:
        print("Accuracy:",result2)
    return result2

def runbasic(X_train, X_test, y_train, y_test, k=1, random_state = 0):
    knn_class = KNeighborsClassifier(n_neighbors=k)

    #treinando o classificador
    knn_class.fit(X_train, y_train)

    #realizando o teste
    ypred=knn_class.predict(X_test)
    
    return accuracy_score(y_test,ypred)

def realizaTestesMultiplos(rangek=[5, 4, 3, 2, 1], distribrange=[0.87, 0.95, 0.99, 0.995],
                            treinos=[treinoRegular, treinoMedia]):
    """
    Esse código objetiva testar o classificador KNN com diferentes valores de k e diferentes 
        distribuições de treino e exibir os resultados em uma matriz.
    """
        
    for treino in treinos:
        #realiza o treino do classificador para cada distribuição e para cada valor de k
        dados =  getdata('dados', treino, distribrange, rangek)
        #exibe os resultados em uma matriz
        imprimeTestesMultiplos(dados, distribrange, rangek)

def get_accuracy(rawAmostras, dividirAmostra, test_size=0.994, k=1, random_state = 0):
    X_train, X_test, y_train, y_test = dividirAmostra(rawAmostras, test_size=test_size,
                                                       random_state=random_state)
    return runbasic(X_train, X_test, y_train, y_test, k=k)

#para realizar o teste
def getdata(path, treino, distrib, Krange):
    rawAmostras, _ = getAmostra(path)
    return [[get_accuracy(rawAmostras, treino, distr, k=k) for distr in distrib] for k in Krange]

testarKnn('dados', treinoRegular, test_size=0.994)
realizaTestesMultiplos()
# testarMultiplos()
#knnTest()
