import pandas as pd
from pathlib import Path
from typing import Dict

def salvar_dataframes_csv(dataframes: Dict[str, pd.DataFrame], diretorio_destino: Path):
    """
    Salva os dataframes em arquivos CSV separados.

    Args:
        dataframes (Dict[str, pd.DataFrame]): Dicionário de dataframes a serem salvos.
        diretorio_destino (Path): Diretório de destino dos arquivos CSV.

    Raises:
        ValueError: Se o diretório de destino não existir.

    Returns:
        None
    """
    # Verifica se o diretório de destino existe
    diretorio_destino = Path(diretorio_destino)
    if not diretorio_destino.exists():
        raise ValueError('O diretório de destino não existe.')

    # Salva cada dataframe em um arquivo CSV separado
    for indice, (key, dataframe) in enumerate(dataframes.items()):
        nome_arquivo = diretorio_destino / f"{indice:02d}-{key}.csv"
        dataframe.to_csv(nome_arquivo, sep=';', index=False)
