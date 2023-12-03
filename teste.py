from modelos.previsao import executar_multiplas_previsoes_KNN_matriz_confusao, executar_multiplas_previsoes_correlacao_matriz_confusao
from metricas.metricas import obter_distancia_minkowski_entre_classes, obter_linha_maior_distancia_minkowski_entre_dataframes, obter_vetor_distancias_a_media_dataframe, obter_distancia_minkowski_min_mean_max_em_classes
from visualizacao.visualizacao import imprime_distribuicao_padronizada_distancias, imprime_distancias, imprime_matriz_distancias_classes, imprime_distribuicao_distancias
from modelos.conjuntos_dados import treino_regular, treino_media
from modelos.correlate_pred import ClassificadorCorrelacaoCruzada, ClassificadorCorrelacao
from modelos.dados import Dados
from dados.escrita import salvar_dataframes_csv
from pathlib import Path

conjunto1_nome = Path('1_carga_3_dp')
conjunto2_nome = Path('2_cargas_3_dp')
corre_cruzada_pasta = "correlacao_cruzada"
corre_simples_pasta = "correlacao_simples"
knn_pasta= "knn"

# conjunto_dados = Dados('2_cargas_3_dp')
# pior_classe, _, _ = obter_linha_maior_distancia_minkowski_entre_dataframes(conjunto_dados.dicionario_dados, p=2)
# imprime_distribuicao_padronizada_distancias(obter_vetor_distancias_a_media_dataframe(conjunto_dados.dicionario_dados[pior_classe], p=2))
# imprime_distancias(obter_vetor_distancias_a_media_dataframe(conjunto_dados.dicionario_dados[pior_classe], p=2))
# print(conjunto_dados.dicionario_dados[pior_classe].shape)


# conjunto_dados = Dados(conjunto1_nome)
# executar_multiplas_previsoes_correlacao_matriz_confusao(conjunto_dados, classificador=ClassificadorCorrelacao, tamanho_treino_lista=[32], repeticoes=1, save_fig=True, fig_folder=str(conjunto1_nome / corre_simples_pasta))
# executar_multiplas_previsoes_correlacao_matriz_confusao(conjunto_dados, classificador=ClassificadorCorrelacaoCruzada, tamanho_treino_lista=[32], repeticoes=1, save_fig=True, fig_folder=str(conjunto1_nome / corre_cruzada_pasta))
# executar_multiplas_previsoes_KNN_matriz_confusao(conjunto_dados, [1], [630], treinos_lista=[treino_media], repeticoes=1, save_fig=True, fig_folder=str(conjunto1_nome / knn_pasta))

# conjunto_dados = Dados(conjunto2_nome)
# executar_multiplas_previsoes_correlacao_matriz_confusao(conjunto_dados, classificador=ClassificadorCorrelacao, tamanho_treino_lista=[630], repeticoes=1, save_fig=True, fig_folder=str(conjunto2_nome / corre_simples_pasta))
# executar_multiplas_previsoes_correlacao_matriz_confusao(conjunto_dados, classificador=ClassificadorCorrelacaoCruzada, tamanho_treino_lista=[630], repeticoes=1, save_fig=True, fig_folder=str(conjunto2_nome / corre_cruzada_pasta))
# executar_multiplas_previsoes_KNN_matriz_confusao(conjunto_dados, [1], [630], treinos_lista=[treino_media], repeticoes=1, save_fig=True, fig_folder=str(conjunto2_nome / knn_pasta))

conjunto_dados = Dados(conjunto1_nome)
imprime_matriz_distancias_classes(obter_distancia_minkowski_entre_classes(conjunto_dados.dicionario_dados, p=2), save_fig=True, caminho=conjunto1_nome)
# imprime_distribuicao_distancias(obter_distancia_minkowski_min_mean_max_em_classes(conjunto_dados.dicionario_dados, p=2), save_fig=True, caminho=conjunto1_nome)
# executar_multiplas_previsoes_correlacao_matriz_confusao(conjunto_dados, classificador=ClassificadorCorrelacao, tamanho_treino_lista=[32, 64, 120, 240, 480], repeticoes=30, save_fig=True, fig_folder=str(conjunto1_nome / corre_simples_pasta))
# executar_multiplas_previsoes_correlacao_matriz_confusao(conjunto_dados, classificador=ClassificadorCorrelacaoCruzada, tamanho_treino_lista=[32, 64, 120, 240, 480], repeticoes=30, save_fig=True, fig_folder=str(conjunto1_nome / corre_cruzada_pasta))
# executar_multiplas_previsoes_KNN_matriz_confusao(conjunto_dados, [1, 3, 5], [32, 64, 120, 240, 480], treinos_lista=[treino_regular, treino_media], repeticoes=30, save_fig=True, fig_folder=str(conjunto1_nome / knn_pasta))

conjunto_dados = Dados(conjunto2_nome)
imprime_matriz_distancias_classes(obter_distancia_minkowski_entre_classes(conjunto_dados.dicionario_dados, p=2), save_fig=True, caminho=conjunto2_nome)
# imprime_distribuicao_distancias(obter_distancia_minkowski_min_mean_max_em_classes(conjunto_dados.dicionario_dados, p=2), save_fig=True, caminho=conjunto2_nome)
# executar_multiplas_previsoes_correlacao_matriz_confusao(conjunto_dados, classificador=ClassificadorCorrelacao, tamanho_treino_lista=[630, 1050, 1260, 1470, 1932], repeticoes=30, save_fig=True, fig_folder=str(conjunto2_nome / corre_simples_pasta))
# executar_multiplas_previsoes_correlacao_matriz_confusao(conjunto_dados, classificador=ClassificadorCorrelacaoCruzada, tamanho_treino_lista=[630, 1050, 1260, 1470, 1932], repeticoes=30, save_fig=True, fig_folder=str(conjunto2_nome / corre_cruzada_pasta))
# executar_multiplas_previsoes_KNN_matriz_confusao(conjunto_dados, [1, 3, 5], [630, 1050, 1260, 1470, 1932], treinos_lista=[treino_regular, treino_media], repeticoes=30, save_fig=True, fig_folder=str(conjunto2_nome / knn_pasta))