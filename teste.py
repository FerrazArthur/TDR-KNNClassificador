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
# conjunto_dados.imprime_classes(caminho=conjunto1_nome, save_fig=True, nome_arquivo="EXP_1_RESULT.pdf")
# conjunto_dados.imprime_classes(caminho=conjunto1_nome, save_fig=True, nome_arquivo="EXP_1_MEANS.pdf", imprimir_medias=True)

# imprime_matriz_distancias_classes(obter_distancia_minkowski_entre_classes(conjunto_dados.dicionario_dados, p=2), save_fig=True, caminho=conjunto1_nome, legendas=conjunto_dados.legenda)
# imprime_distribuicao_distancias(obter_distancia_minkowski_min_mean_max_em_classes(conjunto_dados.dicionario_dados, p=2), save_fig=True, caminho=conjunto1_nome)

# executar_multiplas_previsoes_correlacao_matriz_confusao(conjunto_dados, classificador=ClassificadorCorrelacao, tamanho_treino_lista=[32, 64, 120, 240, 480], repeticoes=30, save_fig=True, fig_folder=str(conjunto1_nome / corre_simples_pasta))
# executar_multiplas_previsoes_correlacao_matriz_confusao(conjunto_dados, classificador=ClassificadorCorrelacaoCruzada, tamanho_treino_lista=[32, 64, 120, 240, 480], repeticoes=30, save_fig=True, fig_folder=str(conjunto1_nome / corre_cruzada_pasta))
# executar_multiplas_previsoes_KNN_matriz_confusao(conjunto_dados, [1, 3, 5], [32, 64, 120, 240, 480], treinos_lista=[treino_regular, treino_media], repeticoes=30, save_fig=True, fig_folder=str(conjunto1_nome / knn_pasta))
# executar_multiplas_previsoes_KNN_matriz_confusao(conjunto_dados, [1], [40], treinos_lista=[ treino_media], repeticoes=1, save_fig=True, fig_folder=str(conjunto1_nome / knn_pasta))

# conjunto_dados = Dados("2_cargas_raw", base_codigo=8)
# for nome, df in conjunto_dados.dicionario_dados.items():
#     print(f"{conjunto_dados.legenda[nome]} & {nome} & & & {df.shape[0]}\\\\")

conjunto_dados = Dados(conjunto2_nome, base_codigo=8)
# conjunto_dados.imprime_classes(caminho=conjunto2_nome, save_fig=True, nome_arquivo="EXP_2_RESULT.pdf")
# conjunto_dados.imprime_classes(caminho=conjunto2_nome, save_fig=True, nome_arquivo="EXP_2_MEANS.pdf", imprimir_medias=True)

# Cargas_1_raw = Dados("1_carga_raw")
# Cargas_1_raw.imprime_classes(caminho="1_carga_raw", save_fig=True, nome_arquivo="EXP_1_RESULT.pdf")
# Cargas_1_raw.imprime_classes(caminho="1_carga_raw", lista_classes=["Circuito aberto", "Curto Circuito"], save_fig=True, nome_arquivo="EXP1_abertoFechado.pdf")
# Cargas_1_raw.imprime_classes(caminho="1_carga_raw", save_fig=True, nome_arquivo="EXP_1_CUT.pdf", inicio_plot=2.35*(10**-7))

# Cargas_2_raw = Dados("2_cargas_raw", base_codigo=8)
# Cargas_2_raw.imprime_classes(caminho="2_cargas_raw", save_fig=True, nome_arquivo="EXP_2_RESULT.pdf")
# Cargas_2_raw.imprime_classes(caminho="2_cargas_raw", lista_classes=["CA CA", "CA CC"], save_fig=True, nome_arquivo="EXP_2_1_2.pdf")

# Cargas_2_raw.imprime_classes(caminho="2_cargas_raw", lista_classes=["CA CA", "CC CA"], save_fig=True, nome_arquivo="EXP_2_2_1.pdf")
# Cargas_2_raw.imprime_classes(caminho="2_cargas_raw", lista_classes=["E1 CA", "E1 CC"], save_fig=True, nome_arquivo="EXP_2_2_1(2).pdf")

# imprime_matriz_distancias_classes(obter_distancia_minkowski_entre_classes(conjunto_dados.dicionario_dados, p=2), save_fig=True, caminho=conjunto2_nome, legendas=conjunto_dados.legenda)
# imprime_distribuicao_distancias(obter_distancia_minkowski_min_mean_max_em_classes(conjunto_dados.dicionario_dados, p=2), save_fig=True, caminho=conjunto2_nome)

# executar_multiplas_previsoes_correlacao_matriz_confusao(conjunto_dados, classificador=ClassificadorCorrelacao, tamanho_treino_lista=[630, 1050, 1260, 1470, 1932], repeticoes=30, save_fig=True, fig_folder=str(conjunto2_nome / corre_simples_pasta))
# executar_multiplas_previsoes_correlacao_matriz_confusao(conjunto_dados, classificador=ClassificadorCorrelacaoCruzada, tamanho_treino_lista=[630, 1050, 1260, 1470, 1932], repeticoes=30, save_fig=True, fig_folder=str(conjunto2_nome / corre_cruzada_pasta))
# executar_multiplas_previsoes_KNN_matriz_confusao(conjunto_dados, [1, 3, 5], [630, 1050, 1260, 1470, 1932], treinos_lista=[treino_regular, treino_media], repeticoes=30, save_fig=True, fig_folder=str(conjunto2_nome / knn_pasta))
# executar_multiplas_previsoes_KNN_matriz_confusao(conjunto_dados, [1], [651], treinos_lista=[treino_media], repeticoes=1, save_fig=True, fig_folder=str(conjunto2_nome / knn_pasta))
