from pathlib import Path
from typing import List
import pickle

def imprime_tabela_latex_precisao(dict1_caminho:str="", tamanhos_lista_1:List[int]=[2232, 3128]\
                        , dict2_caminho:str="", tamanhos_lista_2:List[int]=[3192, 4494])-> None:
    """ 
    Imprime a tabela de precisão em LaTeX.
    Assume que todos os classificadores foram usados: CS, CA, kNN com k=1, k=3 e k=5 e 
    kNN clusterizado com k=1.
    Args:
        dict1_caminho (str): Caminho para o arquivo do dicionário do conjunto de dados 1.
        dict2_caminho (str): Caminho para o arquivo do dicionário do conjunto de dados 2.
        tamanho_lista_1 (List[int]): Lista com os tamanhos de treino do conjunto de dados 1.
        tamanho_lista_2 (List[int]): Lista com os tamanhos de treino do conjunto de dados 2.
    """
    # Carregar o dicionário de um arquivo
    dict1_caminho = Path(dict1_caminho)
    dict2_caminho = Path(dict2_caminho)

    linhas_tabela = []
    with open(dict1_caminho, 'rb') as f:
        dict1 = pickle.load(f)
        for index, valor in enumerate(tamanhos_lista_1):
            linha = [f"d{index+1}"]
            linha.extend([dict1["CS"][str(valor)]["P"]])
            linha.extend([dict1["CA"][str(valor)]["P"]])
            for treino, k_dict in dict1["kNN"].items():
                if treino == "treino_regular":
                    linha.extend([k_dict["k_1"][str(valor)]["P"]])
                    linha.extend([k_dict["k_3"][str(valor)]["P"]])
                    linha.extend([k_dict["k_5"][str(valor)]["P"]])
                if treino == "treino_media":
                    linha.extend([k_dict["k_1"][str(valor)]["P"]])
            linha = [f"${palavra}$" for palavra in linha]
            linhas_tabela.append(linha)

    with open(dict2_caminho, 'rb') as f:
        dict2 = pickle.load(f)
        for index, valor in enumerate(tamanhos_lista_2):
            linha = [f"d{index+3}"]
            linha.extend([dict2["CS"][str(valor)]["P"]])
            linha.extend([dict2["CA"][str(valor)]["P"]])
            for treino, k_dict in dict2["kNN"].items():
                if treino == "treino_regular":
                    linha.extend([k_dict["k_1"][str(valor)]["P"]])
                    linha.extend([k_dict["k_3"][str(valor)]["P"]])
                    linha.extend([k_dict["k_5"][str(valor)]["P"]])
                if treino == "treino_media":
                    linha.extend([k_dict["k_1"][str(valor)]["P"]])
            linha = [f"${palavra}$" for palavra in linha]
            linhas_tabela.append(linha)
    # Transforma o dicionário em uma lista de linhas para a tabela LaTeX

    # Cria a tabela LaTeX
    tabela_latex = "\\begin{tabular}{c|c|c|c|c|c|c}\n"
    tabela_latex += "\\hline\n"
    tabela_latex += "& & & \\multicolumn{3}{c|}{\\textit{kNN}} & \\textit{kNN} Clusterizado \\\\\n"
    tabela_latex += "\\cline{4-7}\n"
    tabela_latex += "Conjunto & \\textit{CS} & \\textit{CA} & $k=1$ & $k=3$ & $k=5$ & $k=1$ \\\\\n"
    tabela_latex += "\\hline\n"

    for linha in linhas_tabela:
        tabela_latex += " & ".join(linha) + " \\\\\n"

    tabela_latex += "\\end{tabular}"

    print(tabela_latex)

def imprime_tabela_latex_mem_time_C(dict1_caminho:str="", num_classes_1 = 8,\
    tamanhos_lista_1:List[int]=[2232, 3128], dict2_caminho:str="", num_classes_2 = 21, \
    tamanhos_lista_2:List[int]=[3192, 4494])-> None:
    """ 
    Imprime uma tabela com o tempo e o pico de consumo de memória entre dois classificados baseados em
     correlação no formato LaTeX.
    Assume que todos os classificadores foram usados: CS, CA
    Args:
        dict1_caminho (str): Caminho para o arquivo do dicionário do conjunto de dados 1.
        dict2_caminho (str): Caminho para o arquivo do dicionário do conjunto de dados 2.
        num_classes_1 (int): Número de classes do conjunto de dados 1.
        num_classes_2 (int): Número de classes do conjunto de dados 2.
        tamanho_lista_1 (List[int]): Lista com os tamanhos de treino do conjunto de dados 1.
        tamanho_lista_2 (List[int]): Lista com os tamanhos de treino do conjunto de dados 2.
    """
    # Carregar o dicionário de um arquivo
    dict1_caminho = Path(dict1_caminho)
    dict2_caminho = Path(dict2_caminho)

    linhas_tabela = []
    with open(dict1_caminho, 'rb') as f:
        dict1 = pickle.load(f)
        for index, valor in enumerate(tamanhos_lista_1):
            linha = [f"d{index+1}"]
            linha.extend([str(num_classes_1)])
            linha.extend([dict1["CS"][str(valor)]["TE"]])
            linha.extend([dict1["CS"][str(valor)]["PM"]])
            linha.extend([dict1["CA"][str(valor)]["TE"]])
            linha.extend([dict1["CA"][str(valor)]["PM"]])
            linha = [f"${palavra}$" for palavra in linha]
            linhas_tabela.append(linha)

    with open(dict2_caminho, 'rb') as f:
        dict2 = pickle.load(f)
        for index, valor in enumerate(tamanhos_lista_2):
            linha = [f"d{index+3}"]
            linha.extend([str(num_classes_2)])
            linha.extend([dict2["CS"][str(valor)]["TE"]])
            linha.extend([dict2["CS"][str(valor)]["PM"]])
            linha.extend([dict2["CA"][str(valor)]["TE"]])
            linha.extend([dict2["CA"][str(valor)]["PM"]])
            linha = [f"${palavra}$" for palavra in linha]
            linhas_tabela.append(linha)
    # Transforma o dicionário em uma lista de linhas para a tabela LaTeX

    # Cria a tabela LaTeX
    tabela_latex = "\\begin{tabular}{c|c|c|c|c|c}\n"
    tabela_latex += "\\hline\n"
    tabela_latex += "& &\\multicolumn{2}{c|}{\\textit{CS}} & \\multicolumn{2}{c}{\\textit{CA}} \\\\\n"
    tabela_latex += "\\cline{3-6}\n"
    tabela_latex += "Conjunto & Classes & TE & PM & TE & PM \\\\\n"
    tabela_latex += "\\hline\n"

    for linha in linhas_tabela:
        tabela_latex += " & ".join(linha) + " \\\\\n"

    tabela_latex += "\\end{tabular}"

    print(tabela_latex)

def imprime_tabela_latex_mem_time_kNN(dict1_caminho:str="", num_classes_1 = 8, tamanhos_lista_1:List[int]=[2232, 3128]\
                        , dict2_caminho:str="", num_classes_2 = 21, tamanhos_lista_2:List[int]=[3192, 4494]):
    """ 
    Imprime uma tabela com o tempo e o pico de consumo de memória entre dois classificados baseados em
     kNN no formato LaTeX.
    Assume que todos os classificadores foram usados: kNN  com k=1, 3 e 5 e kNN com k = 1 e 
    com clusterização da amostra
    Args:
        dict1_caminho (str): Caminho para o arquivo do dicionário do conjunto de dados 1.
        dict2_caminho (str): Caminho para o arquivo do dicionário do conjunto de dados 2.
        num_classes_1 (int): Número de classes do conjunto de dados 1.
        num_classes_2 (int): Número de classes do conjunto de dados 2.
        tamanho_lista_1 (List[int]): Lista com os tamanhos de treino do conjunto de dados 1.
        tamanho_lista_2 (List[int]): Lista com os tamanhos de treino do conjunto de dados 2.
    """
    # Carregar o dicionário de um arquivo
    dict1_caminho = Path(dict1_caminho)
    dict2_caminho = Path(dict2_caminho)

    linhas_tabela = []
    with open(dict1_caminho, 'rb') as f:
        dict1 = pickle.load(f)
        for index, valor in enumerate(tamanhos_lista_1):
            linha = [f"d{index+1}"]
            linha.extend([str(num_classes_1)])
            for treino, k_dict in dict1["kNN"].items():
                if treino == "treino_regular":
                    linha.extend([k_dict["k_1"][str(valor)]["TE"]])
                    linha.extend([k_dict["k_1"][str(valor)]["PM"]])
                    linha.extend([k_dict["k_3"][str(valor)]["TE"]])
                    linha.extend([k_dict["k_3"][str(valor)]["PM"]])
                    linha.extend([k_dict["k_5"][str(valor)]["TE"]])
                    linha.extend([k_dict["k_5"][str(valor)]["PM"]])
                if treino == "treino_media":
                    linha.extend([k_dict["k_1"][str(valor)]["TE"]])
                    linha.extend([k_dict["k_1"][str(valor)]["PM"]])
            
            linha = [f"${palavra}$" for palavra in linha]
            linhas_tabela.append(linha)

    with open(dict2_caminho, 'rb') as f:
        dict2 = pickle.load(f)
        for index, valor in enumerate(tamanhos_lista_2):
            linha = [f"d{index+3}"]
            linha.extend([str(num_classes_2)])
            for treino, k_dict in dict2["kNN"].items():
                if treino == "treino_regular":
                    linha.extend([k_dict["k_1"][str(valor)]["TE"]])
                    linha.extend([k_dict["k_1"][str(valor)]["PM"]])
                    linha.extend([k_dict["k_3"][str(valor)]["TE"]])
                    linha.extend([k_dict["k_3"][str(valor)]["PM"]])
                    linha.extend([k_dict["k_5"][str(valor)]["TE"]])
                    linha.extend([k_dict["k_5"][str(valor)]["PM"]])
                if treino == "treino_media":
                    linha.extend([k_dict["k_1"][str(valor)]["TE"]])
                    linha.extend([k_dict["k_1"][str(valor)]["PM"]])
            
            linha = [f"${palavra}$" for palavra in linha]
            linhas_tabela.append(linha)
    # Transforma o dicionário em uma lista de linhas para a tabela LaTeX

    # Cria a tabela LaTeX
    tabela_latex = "\\begin{tabular}{c|c|c|c|c|c|c|c|c|c}\n"
    tabela_latex += "\\hline\n"
    tabela_latex += "& &\\multicolumn{6}{c|}{\\textit{kNN}} & \\multicolumn{2}{c}{\\textit{kNN} Clusterizado} \\\\\n"
    tabela_latex += "\\cline{3-10}\n"
    tabela_latex += "& &\\multicolumn{2}{c|}{$k=1$} & \\multicolumn{2}{c|}{$k=3$} & \\multicolumn{2}{c|}{$k=5$} & \\multicolumn{2}{c}{$k=1$} \\\\\n"
    tabela_latex += "\\cline{3-10}\n"
    tabela_latex += "Conjunto & Classes & TE & PM & TE & PM & TE & PM & TE & PM \\\\\n"
    tabela_latex += "\\hline\n"

    for linha in linhas_tabela:
        tabela_latex += " & ".join(linha) + " \\\\\n"

    tabela_latex += "\\end{tabular}"

    print(tabela_latex)