import numpy as np

def loadCSVMatrix(nomeArquivo):
    '''
    Input: nome do arquivo csv
    output: retorna um array numpy com os valores presentes no arquivo csv
    '''
    with open(nomeArquivo, "r",) as arquivo:
        dados=arquivo.read()
    linhas=dados.split('\n')
    tabela=[]
    for linha in linhas[:-1]:#ignora a ultima linha
        tabela.append(np.array(linha.split(';')))
    tabela=np.asarray(tabela, dtype=float)
    return tabela
