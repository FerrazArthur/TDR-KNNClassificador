from modelos.previsao import executar_multiplas_previsoes_KNN_matriz_confusao, executar_multiplas_previsoes_correlacao_matriz_confusao
from metricas.metricas import obter_distancia_minkowski_entre_classes, obter_linha_maior_distancia_minkowski_entre_dataframes, obter_vetor_distancias_a_media_dataframe, obter_distancia_minkowski_min_mean_max_em_classes
from visualizacao.visualizacao import imprime_distribuicao_padronizada_distancias, imprime_distancias, imprime_matriz_distancias_classes, imprime_distribuicao_distancias
from visualizacao.tabelas import imprime_tabela_latex_precisao, imprime_tabela_latex_mem_time_C, imprime_tabela_latex_mem_time_kNN
from modelos.conjuntos_dados import treino_regular, treino_media
from modelos.correlate_pred import ClassificadorCorrelacaoCruzada, ClassificadorCorrelacao
from modelos.dados import Dados
from dados.escrita import salvar_dataframes_csv
from pathlib import Path
import pickle

dict1="dict_1_2.pkl"
dict2="dict_2_2.pkl"
dict1_500="dict_1_2_500.pkl"
dict2_500="dict_2_2_500.pkl"
# imprime_tabela_latex_precisao(dict1_caminho=dict1, dict2_caminho=dict2, tamanhos_lista_1=[2232, 3128], tamanhos_lista_2=[3192, 4494])
# imprime_tabela_latex_precisao(dict1_caminho=dict1_500, dict2_caminho=dict2_500, tamanhos_lista_1=[2232, 3128], tamanhos_lista_2=[3192, 4494])
# imprime_tabela_latex_mem_time_C(dict1_caminho=dict1, dict2_caminho=dict2)
# imprime_tabela_latex_mem_time_C(dict1_caminho=dict1_500, dict2_caminho=dict2_500)
imprime_tabela_latex_mem_time_kNN(dict1_caminho=dict1, dict2_caminho=dict2)
imprime_tabela_latex_mem_time_kNN(dict1_caminho=dict1_500, dict2_caminho=dict2_500)