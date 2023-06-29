from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from myTools import *

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
    #mostrarVariancia(rawAmostras)
    #-----------------------------
    #Conjunto de treino composto por apenas um exemplo para cada classe, onde esse exemplo é uma média do conjunto inteiro

    knn_class = KNeighborsClassifier(n_neighbors=1)

    X_train, X_test, y_train, y_test = dividirAmostra(rawAmostras)
    writeCSV(X_train, y_train, "xyTrain.csv")
    #treinando o classificador
    knn_class.fit(X_train, y_train)
    print("tamanho da amostra no treino: ", len(X_train))
    print("tamanho da amostra no teste: ", len(X_test))
    
    #realizando o teste
    ypred=knn_class.predict(X_test)
    result = confusion_matrix(y_test, ypred)
    
    #imprimindo o resultado

    fig, ax = plt.subplots(2, 1)

    print("legenda: ")
    for i in range(len(arquivos)):
        print(i, ": ",arquivos[i])
    #print("Confusion Matrix:")
    #print(result)
    cmd = ConfusionMatrixDisplay(confusion_matrix=result)
    cmd.plot(values_format="d", ax=ax[0])
    result1 = classification_report(y_test, ypred, output_dict=True)
    sns.heatmap(pd.DataFrame(result1).iloc[:-1, :].T, annot=True, ax=ax[1])
    plt.show()
    #print("Classification Report:")
    
    result2 = accuracy_score(y_test,ypred)
    print("Accuracy:",result2)

#testarKnn('dados',treinoRegular)
testarKnn('dados',treinoMedia)

