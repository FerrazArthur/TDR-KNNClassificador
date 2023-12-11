from pathlib import Path
from typing import List
import pickle

def imprime_tabela_latex_precisao(dict1_caminho:str="", tamanhos_lista_1:List[int]=[32, 64, 120, 240, 480]\
                        , dict2_caminho:str="", tamanhos_lista_2:List[int]=[630, 1050, 1260, 1470, 1932]):
    """ 
    Imprime a tabela de precisão em LaTeX.
    Assume que todos os classificadores foram usados: CS, CA, kNN com k=1, k=3 e k=5.
    Args:
        dict1_caminho (str): Caminho para o arquivo do dicionário do conjunto de dados 1.
        dict2_caminho (str): Caminho para o arquivo do dicionário do conjunto de dados 2.
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
                linha.extend([k_dict["k_1"][str(valor)]["P"]])
                linha.extend([k_dict["k_3"][str(valor)]["P"]])
                linha.extend([k_dict["k_5"][str(valor)]["P"]])
            linha = [f"${palavra}$" for palavra in linha]
            linhas_tabela.append(linha)

    with open(dict2_caminho, 'rb') as f:
        dict2 = pickle.load(f)

        for index, valor in enumerate(tamanhos_lista_2):
            linha = [f"d{index+6}"]
            linha.extend([dict2["CS"][str(valor)]["P"]])
            linha.extend([dict2["CA"][str(valor)]["P"]])
            for treino, k_dict in dict2["kNN"].items():
                linha.extend([k_dict["k_1"][str(valor)]["P"]])
                linha.extend([k_dict["k_3"][str(valor)]["P"]])
                linha.extend([k_dict["k_5"][str(valor)]["P"]])
            linha = [f"${palavra}$" for palavra in linha]
            linhas_tabela.append(linha)
    # print(dict2)
    # Transforma o dicionário em uma lista de linhas para a tabela LaTeX

    # Cria a tabela LaTeX
    tabela_latex = "\\begin{tabular}{c|c|c|c|c|c|c|c|c}\n"
    tabela_latex += "\\hline\n"
    tabela_latex += "& & & \\multicolumn{3}{c|}{\\textit{kNN}} & \\multicolumn{3}{c}{\\textit{kNN} Clusterizado}\\\\\n"
    tabela_latex += "\\cline{4-9}\n"
    tabela_latex += "Conjunto & \\textit{CS} & \\textit{CA} & $k=1$ & $k=3$ & $k=5$ & $k=1$ & $k=3$ & $k=5$\\\\\n"
    tabela_latex += "\\hline\n"

    for linha in linhas_tabela:
        tabela_latex += " & ".join(linha) + " \\\\\n"

    tabela_latex += "\\end{tabular}"

    print(tabela_latex)