from modelos.previsao import executar_multiplas_previsoes_KNN_matriz_confusao, executar_multiplas_previsoes_correlacao_matriz_confusao
from metricas.metricas import obter_distancia_minkowski_entre_classes, obter_linha_maior_distancia_minkowski_entre_dataframes, obter_vetor_distancias_a_media_dataframe, obter_distancia_minkowski_min_mean_max_em_classes
from visualizacao.visualizacao import imprime_distribuicao_padronizada_distancias, imprime_distancias, imprime_matriz_distancias_classes, imprime_distribuicao_distancias
from visualizacao.tabelas import imprime_tabela_latex_precisao
from modelos.dados import Dados
from pathlib import Path

conjunto1_nome = Path('1_carga_raw')
conjunto2_nome = Path('2_cargas_raw')

conjunto_dados = Dados(conjunto1_nome)

# imprime matriz de distâncias
# imprime_matriz_distancias_classes(obter_distancia_minkowski_entre_classes(conjunto_dados.dicionario_dados), save_fig=True, caminho=conjunto1_nome)
# imprime distribuição de distâncias
# imprime_distribuicao_distancias(obter_distancia_minkowski_min_mean_max_em_classes(conjunto_dados.dicionario_dados), save_fig=True, caminho=conjunto1_nome)

conjunto_dados.imprime_classes(save_fig=True, caminho=conjunto1_nome, nome_arquivo="EXP_1_RESULT.pdf")

conjunto_dados.imprime_classes(save_fig=True, caminho=conjunto1_nome, nome_arquivo="EXP_1_CUT.pdf", inicio_plot=90)
conjunto_dados.imprime_classes(save_fig=True, lista_classes=["Circuito Aberto", "Curto Circuito"], caminho=conjunto1_nome, nome_arquivo="EXP1_abertoFechado.pdf")

conjunto_dados = Dados(conjunto2_nome)

# imprime matriz de distâncias
# imprime_matriz_distancias_classes(obter_distancia_minkowski_entre_classes(conjunto_dados.dicionario_dados), save_fig=True, caminho=conjunto2_nome)
# imprime distribuição de distâncias
# imprime_distribuicao_distancias(obter_distancia_minkowski_min_mean_max_em_classes(conjunto_dados.dicionario_dados), save_fig=True, caminho=conjunto2_nome)


# conjunto_dados.imprime_classes(save_fig=True, caminho=conjunto2_nome, nome_arquivo="EXP_2_RESULT.pdf", frequencia=2)

# conjunto_dados.imprime_classes(save_fig=True, lista_classes=["CA CA", "CA CC", "CC CA"], caminho=conjunto2_nome, nome_arquivo="EXP_2_2_2.pdf", frequencia=2)