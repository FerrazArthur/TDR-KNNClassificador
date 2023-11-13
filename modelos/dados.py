from dados.leitura import explorar_dataframe_csv
from pathlib import Path

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


    def normalizar_amostras(self):
        """
        Normaliza o número de amostras em todas as classes, removendo o excedente das classes que têm mais amostras.

        A função percorre todas as classes no dicionário de dados e, se uma classe tiver mais amostras do que o
        número mínimo encontrado, ela remove o excedente de amostras de forma aleatória.

        Esta função é útil para garantir que todas as classes tenham o mesmo número de amostras, facilitando
        comparações e análises.

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
                # Remover o excedente de amostras
                self.dicionario_dados[classe] = dataframe.sample(min_amostras)

        # Atualizar o número total de amostras
        self.num_amostras = sum(df.shape[0] for df in self.dicionario_dados.values())