from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
import seaborn as sns
from data.retrieve import explore_csv_dataframe
from models.split_train_test import treino_media, treino_regular
from visualization.visualization import imprime_conf_mat_e_class_rep, imprime_testes_multiplos, imprime_testes_multiplos_train_explicit

def get_confusion(data_dict:dict, treino, test_size:float=0.994, k:int=1, imprime_legenda:bool=True, imprime_accuracy:bool=True):
    """
        Realiza o treino do classificador para uma distribuição e para um valor de k
        e imprime a matriz de confusão e o relatório de classificação.
    """
    legenda=[]

    X_train, X_test, y_train, y_test = treino(data_dict, test_size=test_size)
    knn_class = KNeighborsClassifier(n_neighbors=k, weights='distance')
    print(f"X_train: {len(X_train)}")
    print(f"X_test: {len(X_test)}")

    # Treinando o classificador
    knn_class.fit(X_train, y_train)

    # Realizando o teste
    ypred=knn_class.predict(X_test)
    # if imprime_legenda == True:
    #     # Definindo uma legenda
    #     legenda = [key for key in data_dict.keys()]

    conf_mat = confusion_matrix(y_test, ypred, labels=legenda)
    classification_rep = classification_report(y_test, ypred, output_dict=True)
    
    acuracia = accuracy_score(y_test,ypred)
    if imprime_accuracy == True:
        imprime_conf_mat_e_class_rep(conf_mat, classification_rep, acuracia=acuracia, legenda=legenda)
    else:
        imprime_conf_mat_e_class_rep(conf_mat, classification_rep, legenda=legenda)
    return acuracia

def run_basic(X_train, X_test, y_train, y_test, k=1):
    """
        Run a basic KNN classifier with the given data.
    """
    knn_class = KNeighborsClassifier(n_neighbors=k, weights='distance')

    #treinando o classificador
    knn_class.fit(X_train, y_train)

    #realizando o teste

    ypred=knn_class.predict(X_test)
    return accuracy_score(y_test,ypred)

def run_multiple_tests(rangek=[5, 4, 3, 2, 1], distribrange=[0.87, 0.95, 0.99, 0.995],
                        treinos=[treino_regular, treino_media], path='dados_1_carga', result_method=get_confusion):
    """
        Run multiple tests with different values of k and different distributions of train and test data.
    """
    # Realiza o treino do classificador para cada distribuição e para cada valor de k
    for treino in treinos:
        dados = get_data(path, treino, distribrange, rangek, result_method=result_method)
        # Exibe os resultados em uma matriz
        imprime_testes_multiplos(dados, distribrange, rangek)

def run_multiple_tests_train_explicit(rangek=[5, 4, 3, 2, 1], trainrange=[10, 15, 30, 60],
                        treinos=[treino_regular, treino_media], path='dados_1_carga'):
    """
        Run multiple tests with different values of k and different distributions of train and test data.
    """
    # Realiza o treino do classificador para cada distribuição e para cada valor de k
    for treino in treinos:
        dados = get_data_train_explicit(path, treino, trainrange, rangek)
        # Exibe os resultados em uma matriz
        imprime_testes_multiplos_train_explicit(dados, trainrange, rangek)

def get_accuracy(data_dict:dict, treino, test_size:float=0.994, k:int=1):
    """
        Get the accuracy of the model.
    """
    X_train, X_test, y_train, y_test = treino(data_dict, test_size=test_size)
    print(f"X_train: {len(X_train)}")
    print(f"X_test: {len(X_test)}")
    return run_basic(X_train, X_test, y_train, y_test, k=k)

def get_accuracy_train_explicit(data_dict:dict, treino, train_size:int=30, k:int=1):
    """
        Get the accuracy of the model.
    """
    X_train, X_test, y_train, y_test = treino(data_dict, train_size=train_size)
    print(f"X_train: {len(X_train)}")
    print(f"X_test: {len(X_test)}")
    return run_basic(X_train, X_test, y_train, y_test, k=k)

def get_data(path, treino, distribrange, rangek, result_method=get_accuracy):
    """
        Get the data from the given path and run the tests.
    """
    # Carregando os dados
    data_dict = explore_csv_dataframe(path)
    
    # Rodando os testes
    return [[result_method(data_dict, treino, test_size, k) for test_size in distribrange] for k in rangek]

def get_data_train_explicit_old(path, treino, trainrange, rangek):
    """
        Get the data from the given path and run the tests.
    """
    # Carregando os dados
    data_dict = explore_csv_dataframe(path)
    
    # Rodando os testes
    return [[get_accuracy_train_explicit(data_dict, treino, train_size, k) for train_size in trainrange] for k in rangek]

def get_data_train_explicit(path, treino, trainrange, rangek, num_tests=10):
    """
    Get the data from the given path and run tests, returning the average of X tests with the same configuration.
    """
    results = []

    # Carregando os dados
    data_dict = explore_csv_dataframe(path)

    for k in rangek:
        k_results = []
        for train_size in trainrange:
            test_results = []
            for _ in range(num_tests):
                accuracy = get_accuracy_train_explicit(data_dict, treino, train_size, k)
                test_results.append(accuracy)
            # Calculate the average of X test results
            avg_accuracy = sum(test_results) / num_tests
            k_results.append(avg_accuracy)
        results.append(k_results)

    return results