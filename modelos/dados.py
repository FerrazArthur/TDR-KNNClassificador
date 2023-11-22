from dados.leitura import explorar_dataframe_csv
from metricas.metricas import obter_vetor_distancias_a_media_dataframe

from pathlib import Path
from scipy import stats
import numpy as np

class Dados:
    """
    Classe que representa os dados de um conjunto de dados.
    dict_dados: dicionário de dataframes com os dados.
    legenda: lista com o nome das classes.
    num_classes: número de classes.
    num_amostras: número total de amostras.
    """
    def __init__(self, caminho:Path):
        self.dicionario_dados = explorar_dataframe_csv(caminho)
        self.legenda = {classe: i for i, classe in enumerate(self.dicionario_dados.keys())}
        self.classes_lista = list(self.dicionario_dados.keys())
        self.num_classes = len(self.legenda)
        self.num_amostras = sum(df.shape[0] for df in self.dicionario_dados.values())

    def normalizar_amostras(self, imprimir:bool=False):
        """
        Normaliza o número de amostras em todas as classes, removendo o excedente das classes que têm mais amostras.

        A função percorre todas as classes no dicionário de dados e, se uma classe tiver mais amostras do que o
        número mínimo encontrado, ela remove o excedente de amostras de forma aleatória.

        Esta função é útil para garantir que todas as classes tenham o mesmo número de amostras, facilitando
        comparações e análises.

        Args:
            imprimir (bool, opcional): Se True, imprime o número de amostras de cada classe antes e depois da normalização. Padrão é False.

        Raises:
            None

        Returns:
            None
        """
        # Encontrar o número mínimo de amostras entre todas as classes
        min_amostras = min(df.shape[0] for df in self.dicionario_dados.values())

        # Iterar sobre cada classe e remover o excedente de amostras
        for classe, dataframe in self.dicionario_dados.items():
            if dataframe.shape[0] > min_amostras:
                # Remover o excedente de amostras e redefinir os índices
                self.dicionario_dados[classe] = dataframe.sample(min_amostras).reset_index(drop=True)

        # Atualizar o número total de amostras
        self.num_amostras = sum(df.shape[0] for df in self.dicionario_dados.values())
        if imprimir == True:
            print(f"Número de amostras após normalização: {self.num_amostras}")

    def remover_outliers(self, limite:int=3, ddof:int=1, imprimir:bool=False):
        """
        Remove outliers de todos os dataframes no dicionário de dados.

        A função percorre todas as classes no dicionário de dados e remove os outliers de cada classe.
        Esse processo é repetido até nao haver mais alteração no tamanho das amostras.

        Args:
            limite (int, opcional): Limite para o z-score. Padrão é 3.
            ddof (int, opcional): Graus de liberdade para o cálculo do desvio padrão. Padrão é 1.
            imprimir (bool, opcional): Se True, imprime o tamanho do conjunto de dados após cada iteração. Padrão é False.

        Raises:
            None

        Returns:
            None
        """
        tamanho_anterior = self.num_amostras
        iteracoes = 0
        while True:
            # Iterar sobre cada classe e remover os outliers
            iteracoes += 1
            for classe in self.classes_lista:
                
                # Calcular médias e desvios padrão de cada coluna
                distancias = obter_vetor_distancias_a_media_dataframe(self.dicionario_dados[classe])

                media = distancias.mean()
                desvio_padrao = distancias.std(ddof=ddof)

                # Calcular z-scores para cada linha
                z_scores = (distancias - media) / desvio_padrao

                # Verificar se alguma linha excede os limites superior ou inferior de z-score
                linhas_sem_outliers = (abs(z_scores) < limite)

                # Manter apenas as linhas sem outliers no DataFrame
                self.dicionario_dados[classe] = self.dicionario_dados[classe][linhas_sem_outliers].reset_index(drop=True)

            # Atualizar o número total de amostras
            self.num_amostras = sum(df.shape[0] for df in self.dicionario_dados.values())
            
            if self.num_amostras == tamanho_anterior:
                break
            if imprimir == True:
                print(f"Tamanho do conjunto de dados após remover os outliers {iteracoes} vez(es): {self.num_amostras}")

            tamanho_anterior = self.num_amostras