from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
import re

def carregar_csv(nome_arquivo: Path) -> pd.DataFrame:
    """
    Carrega um arquivo csv em um dataframe pandas.
    
    Args:
        nome_arquivo (Path): Caminho para o arquivo csv.
    
    Returns:
        pd.DataFrame: DataFrame com o arquivo csv.
    """
    return pd.read_csv(nome_arquivo, sep=';')

def explorar_dataframe_csv(diretorio: Path) -> Dict[str, pd.DataFrame]:
    """
    Explora um diretório e retorna um dicionário de dataframes com todos os arquivos csv nele.
    
    Args:
        diretorio (Path): Caminho para o diretório contendo os arquivos csv.
    
    Raises:
        TypeError: Se o caminho não for um Path ou str.
        NotADirectoryError: Se o caminho não for um diretório.
        ValueError: Se o diretório estiver vazio.
    
    Returns:
        Dict[str, pd.DataFrame]: Dicionário de dataframes com os arquivos csv.
    """
    # Garante que o caminho seja um objeto Path
    diretorio = Path(diretorio)
    if not isinstance(diretorio, str) and not isinstance(diretorio, Path):
        raise TypeError('O caminho deve ser um objeto Path ou str.')
    if not diretorio.is_dir():
        raise NotADirectoryError('O caminho deve ser um diretório.')
    
    csv_dict = {}
    
    for csv in sorted(diretorio.iterdir(), key=lambda x: int(x.stem.split('-')[0])):
        if csv.suffix.lower() == '.csv':
            # Formato do nome do arquivo é <número>-<nome>.csv
            csv_dict[csv.stem.split('-')[1]] = carregar_csv(csv)
    
    if not csv_dict:
        raise ValueError('O diretório está vazio.')

    return csv_dict

# ----------------------------------·OLD FUNCTIONS·----------------------------------
def obter_amostra(caminho):
    '''
    Entrada: Recebe uma string caminho para onde estão os arquivos csv

    Saída: amostras, um vetor de dim 3, onde dim 1 é cada arquivo csv, 
    dim 2 contém as amostras desse arquivo e dim 3 são os valores dos sinais
    e classes é um vetor com o nome de cada arquivo na ordem.
    '''
    dados = sorted(Path(caminho).glob('*'))
    amostras = []
    classes = []
    for csv in dados:
        amostras.append(carregar_matriz_csv(csv))
        # armazenando os nomes sem o caminho
        classes.append(str(csv).strip(caminho+'/').strip('.CSV').strip('.csv'))
        # removendo a ordem (o número que acompanha o início dos arquivos)
        classes[-1] = re.split(r'^[0-9]+-', classes[-1])[1]

    return np.array(amostras), np.array(classes)

def carregar_matriz_csv(nome_arquivo):
    '''
    Entrada: nome do arquivo csv
    Saída: retorna um array numpy com os valores presentes no arquivo csv
    '''
    with open(nome_arquivo, "r",) as arquivo:
        dados = arquivo.read()
    linhas = dados.split('\n')
    tabela = []
    for linha in linhas[:-1]:  # ignora a última linha
        tabela.append(np.array(linha.split(';')))
    tabela = np.asarray(tabela, dtype=float)
    return tabela

def escrever_csv(x, y, nome_arquivo):
    '''
    Entrada: conjunto de amostras, label das amostras, nome do arquivo csv
    Saída: True se ocorreu algum erro, False do contrário
    '''
    try:
        with open(nome_arquivo, "w",) as arquivo:
            # escreve cabeçalho CLASSE, VALORES
            arquivo.write("CLASSE, ")
            # escreve valor para cada dimensão 
            for i in range(np.size(x[0], 0)):
                arquivo.write("VALOR,")
            arquivo.write('\n')
            for i in range(np.size(x, 0)):
                arquivo.write(str(y[i]))
                for j in range(np.size(x[i], 0)):
                    arquivo.write(',' + str(x[i][j]))
                arquivo.write('\n')
        return False
    except Exception as e:
        print('Erro ao escrever em arquivo "' + nome_arquivo + '": ' + str(e))
        return True