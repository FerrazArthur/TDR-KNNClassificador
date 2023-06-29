from pathlib import Path
import numpy as np
import re

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

def writeCSV(x, y, nomeArquivo):
    '''
    Input: conjunto de amostras, label das amostras, nome do arquivo csv
    output: True se ocorreu algum erro, False do contrário
    '''
    try:
        with open(nomeArquivo, "w",) as arquivo:
            #escreve cabeçalho CLASSE, VALORES
            arquivo.write("CLASSE, ")
            #escreve valor pra cada dimensão 
            for i in range(np.size(x[0], 0)):
                arquivo.write("VALOR,")
            arquivo.write('\n')
            for i in range(np.size(x, 0)):
                arquivo.write(str(y[i]))
                for j in range(np.size(x[i], 0)):
                    arquivo.write(','+str(x[i][j]))
                arquivo.write('\n')
        return False
    except Exception as e:
        print('Erro ao escrever em arquivo "'+nomeArquivo+'": '+str(e))
        return True

def getAmostra(path):
    '''
    Input: Recebe uma string caminho para onde estão os arquivos csv

    Output:amostras, um vetor de dim 3, onde dim 1 é cada arquivo csv, 
    dim 2 contém as amostras desse arquivo e dim 3 são os valores dos sinais
    e classes é um vetor com o nome de cada arquivo na ordem.
    '''
    dados = sorted(Path(path).glob('*'))
    amostras = []
    classes = []
    print()
    for csv in dados:
        amostras.append(loadCSVMatrix(csv))
        #armazenando os nomes sem o caminho
        classes.append(str(csv).strip(path+'/').strip('.CSV').strip('.csv'))
        #removendo a ordem (o número que acompanha o inicio dos arquivos)
        classes[-1] = re.split(r'^[0-9]+-', classes[-1])[1]

    return np.array(amostras), np.array(classes)