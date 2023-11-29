from modelos.previsao import executar_multiplas_previsoes_KNN_matriz_confusao, executar_multiplas_previsoes_correlacao_matriz_confusao
from metricas.metricas import obter_distancia_media_minkowski_entre_dataframe, obter_distancia_media_minkowski_entre_media_dataframe, obter_linha_maior_distancia_minkowski_entre_dataframes, obter_vetor_distancias_a_media_dataframe
from visualizacao.visualizacao import imprime_distribuicao_padronizada_distancias, imprime_distancias
from modelos.conjuntos_dados import treino_regular, treino_media
from modelos.correlate_pred import ClassificadorCorrelacaoCruzada, ClassificadorCorrelacao
from modelos.dados import Dados
from dados.escrita import salvar_dataframes_csv

# conjunto_dados = Dados('2_cargas_3_dp')
# pior_classe, _, _ = obter_linha_maior_distancia_minkowski_entre_dataframes(conjunto_dados.dicionario_dados, p=2)
# imprime_distribuicao_padronizada_distancias(obter_vetor_distancias_a_media_dataframe(conjunto_dados.dicionario_dados[pior_classe], p=2))
# imprime_distancias(obter_vetor_distancias_a_media_dataframe(conjunto_dados.dicionario_dados[pior_classe], p=2))
# print(conjunto_dados.dicionario_dados[pior_classe].shape)

# executar_multiplas_previsoes_KNN_matriz_confusao(conjunto_dados, [1, 3, 5], [0.05, 0.1, 0.3, 1932], treinos_lista=[treino_regular, treino_media], repeticoes=10, safe_fig=True)
# executar_multiplas_previsoes_KNN_matriz_confusao(conjunto_dados, [1, 3, 5], [483, 966, 1932], treinos_lista=[treino_media, treino_regular], repeticoes=10, save_fig=True)
# executar_multiplas_previsoes_KNN_matriz_confusao(conjunto_dados, [1, 3, 5], [32, 64, 120, 240, 480], treinos_lista=[treino_regular, treino_media], repeticoes=30, save_fig=True, fig_folder='resultados')
# executar_multiplas_previsoes_KNN_matriz_confusao(conjunto_dados, [1, 3, 5], [630, 1050, 1260, 1470, 1932], treinos_lista=[treino_regular, treino_media], repeticoes=30, save_fig=True, fig_folder='resultados')
#executar_multiplas_previsoes_correlacao_matriz_confusao(conjunto_dados, classificador=ClassificadorCorrelacao, tamanho_treino_lista=[32, 480], repeticoes=1, save_fig=True, fig_folder='2_resultados_correlacao')
# executar_multiplas_previsoes_correlacao_matriz_confusao(conjunto_dados, classificador=ClassificadorCorrelacaoCruzada, tamanho_treino_lista=[32, 480], repeticoes=10, save_fig=True, fig_folder='resultados_correlacao_cruzada')
conjunto_dados = Dados('1_carga_3_dp')
# executar_multiplas_previsoes_correlacao_matriz_confusao(conjunto_dados, classificador=ClassificadorCorrelacao, tamanho_treino_lista=[32, 64, 120, 240, 480], repeticoes=30, save_fig=True, fig_folder='1resultados_correlacao')
# executar_multiplas_previsoes_correlacao_matriz_confusao(conjunto_dados, classificador=ClassificadorCorrelacaoCruzada, tamanho_treino_lista=[32, 64, 120, 240, 480], repeticoes=30, save_fig=True, fig_folder='1resultados_correlacao_cruzada')
executar_multiplas_previsoes_KNN_matriz_confusao(conjunto_dados, [1, 3, 5], [630, 1050, 1260, 1470, 1932], treinos_lista=[treino_regular, treino_media], repeticoes=30, save_fig=True, fig_folder='1resultados')
conjunto_dados = Dados('2_cargas_3_dp')
# executar_multiplas_previsoes_correlacao_matriz_confusao(conjunto_dados, classificador=ClassificadorCorrelacao, tamanho_treino_lista=[630, 1050, 1260, 1470, 1932], repeticoes=30, save_fig=True, fig_folder='2resultados_correlacao')
# executar_multiplas_previsoes_correlacao_matriz_confusao(conjunto_dados, classificador=ClassificadorCorrelacaoCruzada, tamanho_treino_lista=[630, 1050, 1260, 1470, 1932], repeticoes=30, save_fig=True, fig_folder='2resultados_correlacao_cruzada')
executar_multiplas_previsoes_KNN_matriz_confusao(conjunto_dados, [1, 3, 5], [630, 1050, 1260, 1470, 1932], treinos_lista=[treino_regular, treino_media], repeticoes=30, save_fig=True, fig_folder='2resultados')