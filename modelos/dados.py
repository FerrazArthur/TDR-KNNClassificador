from dados.leitura import explorar_dataframe_csv
from metricas.metricas import obter_escore_padrao
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

from typing import List
from pathlib import Path
import numpy as np
import locale

class Dados:
    """
    Classe que representa os dados de um conjunto de dados.
    dict_dados: dicionário de dataframes com os dados.r
    legenda: lista com o nome das classes.
    num_classes: número de classes.
    num_amostras: número total de amostras.
    """
    def __init__(self, caminho:Path, base_codigo:int=0):
        self.dicionario_dados = explorar_dataframe_csv(caminho)
        self.legenda = {classe: i+base_codigo for i, classe in enumerate(self.dicionario_dados.keys())}
        self.classes_lista = list(self.dicionario_dados.keys())
        self.num_classes = len(self.legenda)
        self.num_amostras = sum(df.shape[0] for df in self.dicionario_dados.values())

    def normalizar_amostras(self, imprimir:bool=False):
        """
        Normaliza o número de amostras em todas as classes. Para tanto, a função percorre todas as classes no
        dicionário de dados e, se uma classe tiver mais amostras do que o
        número mínimo encontrado, ela remove o excedente de amostras.

        Esta função é útil para garantir que todas as classes tenham o mesmo número de amostras, facilitando
        comparações e análises.

        Args:
            imprimir (bool, opcional): Se True, imprime o número de amostras de cada classe antes e depois da 
            normalização. Padrão é False.

        Raises:
            None

        Returns:
            None
        """
        # Encontrar o número mínimo de amostras entre todas as classes
        max_amostras = min(df.shape[0] for df in self.dicionario_dados.values())

        # Iterar sobre cada classe e remover o excedente de amostras
        for classe, dataframe in self.dicionario_dados.items():
            if dataframe.shape[0] > max_amostras:
                # Calcular valores absolutos dos z-scores
                abs_z_scores = np.abs(obter_escore_padrao(dataframe))

                # Identificar índices das amostras com maiores valores absolutos de z-score
                idx_to_remove = abs_z_scores.argsort()[:-max_amostras]
                # Remover o excedente de amostras e redefinir os índices
                self.dicionario_dados[classe] = dataframe.drop(idx_to_remove).reset_index(drop=True)

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
            imprimir (bool, opcional): Se True, imprime o tamanho do conjunto de dados após cada iteração. 
            Padrão é False.

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
                # Calcular z-scores para cada linha
                z_scores = obter_escore_padrao(self.dicionario_dados[classe], ddof=ddof)

                # Verificar se alguma linha excede os limites superior ou inferior de z-score
                linhas_sem_outliers = (abs(z_scores) < limite)

                # Manter apenas as linhas sem outliers no DataFrame
                self.dicionario_dados[classe] = \
                    self.dicionario_dados[classe][linhas_sem_outliers].reset_index(drop=True)

            # Atualizar o número total de amostras
            self.num_amostras = sum(df.shape[0] for df in self.dicionario_dados.values())
            
            if self.num_amostras == tamanho_anterior:
                break
            if imprimir == True:
                print(f"Tamanho do conjunto de dados após remover os outliers {iteracoes} vezes:\
                       {self.num_amostras}")

            tamanho_anterior = self.num_amostras
    
    def reduzir_dimensionalidade(self, extended_slices:int=2):
        """
        Reduz a dimensionalidade de todos os dataframes no dicionário de dados.

        A função percorre todas as amostras no dicionário de dados e a substitui por uma subamostra
        com passo igual a extended_slices.

        Args:
            extended_slices (int, opcional): Tamanho do passo. Padrão é 2.

        Raises:
            None

        Returns:
            None
        """
        for chave, amostra in self.dicionario_dados.items():
            self.dicionario_dados[chave] = amostra.iloc[:, ::extended_slices].dropna(axis=1)

    def imprime_classes(self, lista_classes:List[str]=None, save_fig=False, caminho:str="", nome_arquivo:str="classes.pdf", inicio_plot:float=0, imprimir_medias:bool=False, frequencia:float=5):
        """
        Imprime as classes do conjunto de dados.

        Args:
            lista_classes (List[str], opcional): Lista com as classes a serem impressas. Padrão é None.
            save_fig (bool, opcional): Se True, salva a figura. Padrão é False.
            caminho (str): Caminho para salvar a figura.
            nome_arquivo (str): Nome do arquivo para salvar a figura.
            inicio_plot (float): Tempo inicial para plotar a figura.
            imprimir_medias (bool): Se True, imprime apenas o sinal médio de cada classe.
            frequencia (float): Frequência de amostragem em Ghz

        Raises:
            ValueError: Se a lista de classes possuir alguma classe não conhecida ou 
            se inicio_plot for maior que o tempo total.

        Returns:
            None
        """
        if lista_classes is None:
            lista_classes = self.classes_lista
        else:
            for classe in lista_classes:
                if classe not in self.classes_lista:
                    raise ValueError(f"Classe {classe} não encontrada no conjunto de dados.")
        # Define largura em polegadas
        largura_polegadas = 455/72

        np.random.seed(self.num_classes)

        fig, ax = plt.subplots(figsize=(largura_polegadas, largura_polegadas/2))
        
        num_pontos = self.dicionario_dados[lista_classes[0]].shape[1]
        # tempo_total = 5*(10**(-7)) #s seg
        frequencia = frequencia*(10**9) #GHz
        tempo_total = num_pontos / frequencia #s seg
        # frequencia = num_pontos / float(tempo_total)

        if inicio_plot > tempo_total:
            raise ValueError(f"O tempo inicial para plotar a figura deve ser menor que o tempo total {tempo_total}.")

        tempo = np.linspace(0, tempo_total, num_pontos)

        ax.set_xlabel("Tempo (s)", fontsize=7)
        # Altera o tamanho da fonte
        ax.set_ylabel("Tensão (V)", fontsize=7)
        # Adiciona grid


        # Imprime as series temporais lado a lado, cada classe receberá uma cor
        cores = {classe: np.random.rand(1, 3) * 0.65 for classe in lista_classes}
        # Adiciona uma legenda pra cada cor com um nome de classe

        for i, classe in enumerate(lista_classes):
            if imprimir_medias == True:
                ax.plot(tempo, self.dicionario_dados[classe].mean(axis=0), color=cores[classe], linewidth=0.1, alpha=1)
            else:
                ax.plot(tempo, self.dicionario_dados[classe].T, color=cores[classe], linewidth=0.075, alpha=1)
        

        legendas = [plt.Line2D([0], [0], color=cor, label=classe, linewidth=0.002) for classe, cor in cores.items()]

        legenda = ax.legend(loc="upper right", fontsize=3, framealpha=0.5, handles=legendas)
        for linha in legenda.get_lines():
            linha.set_linewidth(1.5)

        ax.grid(True)
        # Adiciona detalhes matematicos
        ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=False, useOffset=False))
        locale.setlocale(locale.LC_ALL, 'pt_BR.UTF-8')
        ax.ticklabel_format(style="sci", axis="x", scilimits=(-9, -9), useLocale=True)

        ax.set_xlim(inicio_plot, tempo_total)
        fig.tight_layout()

        if save_fig == True:
            caminho = Path(caminho)
            caminho.mkdir(parents=True, exist_ok=True)
            file_name = str("_".join(caminho.parts[1:])) + nome_arquivo
            plt.savefig(caminho / file_name, format="pdf", dpi=300)
            plt.close()
        else:
            plt.show()
            
        locale.setlocale(locale.LC_ALL, 'C')
