from pathlib import Path
from typing import Dict

import pandas as pd

def carregar_csv(nome_arquivo: Path) -> pd.DataFrame:
    """
    Carrega um arquivo csv em um dataframe pandas.
    
    Args:
        nome_arquivo (Path): Caminho para o arquivo csv.
    
    Returns:
        pd.DataFrame: DataFrame com o arquivo csv.
    """
    return pd.read_csv(nome_arquivo, sep=';', header=None, index_col=None)

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

    # Filter the list of files
    csv_files = [file for file in diretorio.iterdir() if file.suffix.lower() == '.csv']

    for csv in sorted(csv_files, key=lambda x: int(x.stem.split('-')[0]) if x.stem[0].isdigit() else x.stem):
        if csv.suffix.lower() == '.csv':
            # Formato do nome do arquivo é <número>-<nome>.csv
            csv_dict[csv.stem.split('-')[1]] = carregar_csv(csv)
    
    if not csv_dict:
        raise ValueError('O diretório está vazio.')

    return csv_dict