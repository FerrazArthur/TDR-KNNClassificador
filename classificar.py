from modelos.previsao import executar_multiplas_previsoes_KNN_matriz_confusao, executar_multiplas_previsoes_correlacao_matriz_confusao
from metricas.metricas import obter_distancia_minkowski_entre_classes, obter_linha_maior_distancia_minkowski_entre_dataframes, obter_vetor_distancias_a_media_dataframe, obter_distancia_minkowski_min_mean_max_em_classes
from visualizacao.visualizacao import imprime_distribuicao_padronizada_distancias, imprime_distancias, imprime_matriz_distancias_classes, imprime_distribuicao_distancias
from visualizacao.tabelas import imprime_tabela_latex_precisao
from modelos.conjuntos_dados import treino_regular, treino_media
from modelos.correlate_pred import ClassificadorCorrelacaoCruzada, ClassificadorCorrelacao
from modelos.dados import Dados
from dados.escrita import salvar_dataframes_csv
from pathlib import Path
import pickle

conjunto1_nome = Path('1_carga_3_dp')
conjunto2_nome = Path('2_cargas_3_dp')
corre_cruzada_pasta = "correlacao_cruzada"
corre_simples_pasta = "correlacao_simples"
knn_pasta= "knn"

resultdict_1={}
conjunto_dados = Dados(conjunto1_nome)

resultdict_1["CS"] = executar_multiplas_previsoes_correlacao_matriz_confusao(conjunto_dados, classificador=ClassificadorCorrelacao, tamanho_treino_lista=[2232, 3128], repeticoes=30, save_fig=True, fig_folder=str(conjunto1_nome / corre_simples_pasta))
resultdict_1["CA"] = executar_multiplas_previsoes_correlacao_matriz_confusao(conjunto_dados, classificador=ClassificadorCorrelacaoCruzada, tamanho_treino_lista=[2232, 3128], repeticoes=30, save_fig=True, fig_folder=str(conjunto1_nome / corre_cruzada_pasta))
resultdict_1["kNN"] = executar_multiplas_previsoes_KNN_matriz_confusao(conjunto_dados, [1, 3, 5], [2232, 3128], treinos_lista=[treino_regular, treino_media], repeticoes=30, save_fig=True, fig_folder=str(conjunto1_nome / knn_pasta))
# resultdict_1["kNN"] = executar_multiplas_previsoes_KNN_matriz_confusao(conjunto_dados, [1], [2232, 3128], treinos_lista=[treino_media], repeticoes=30, save_fig=True, fig_folder=str(conjunto1_nome / knn_pasta))

print("exp1")
print(resultdict_1)
# Salvar o dicionário em um arquivo
with open('dict_1_3.pkl', 'wb') as f:
    pickle.dump(resultdict_1, f)

# resultdict_1={}

# conjunto_dados.reduzir_dimensionalidade(extended_slices=2)
# resultdict_1["CS"] = executar_multiplas_previsoes_correlacao_matriz_confusao(conjunto_dados, classificador=ClassificadorCorrelacao, tamanho_treino_lista=[2232, 3128], repeticoes=30, save_fig=True, fig_folder=str(conjunto1_nome / corre_simples_pasta))
# resultdict_1["CA"] = executar_multiplas_previsoes_correlacao_matriz_confusao(conjunto_dados, classificador=ClassificadorCorrelacaoCruzada, tamanho_treino_lista=[2232, 3128], repeticoes=30, save_fig=True, fig_folder=str(conjunto1_nome / corre_cruzada_pasta))
# resultdict_1["kNN"] = executar_multiplas_previsoes_KNN_matriz_confusao(conjunto_dados, [1, 3, 5], [2232, 3128], treinos_lista=[treino_regular, treino_media], repeticoes=30, save_fig=True, fig_folder=str(conjunto1_nome / knn_pasta))
# # resultdict_1["kNN"] = executar_multiplas_previsoes_KNN_matriz_confusao(conjunto_dados, [1], [2232, 3128], treinos_lista=[treino_media], repeticoes=30, save_fig=True, fig_folder=str(conjunto1_nome / knn_pasta))

# print("exp1_500")
# print(resultdict_1)
# # Salvar o dicionário em um arquivo
# with open('dict_1_3_500.pkl', 'wb') as f:
#     pickle.dump(resultdict_1, f)

resultdict_2={}
conjunto_dados = Dados(conjunto2_nome, base_codigo=8)

resultdict_2["CS"] = executar_multiplas_previsoes_correlacao_matriz_confusao(conjunto_dados, classificador=ClassificadorCorrelacao, tamanho_treino_lista=[3192, 4494], repeticoes=30, save_fig=True, fig_folder=str(conjunto2_nome / corre_simples_pasta))
resultdict_2["CA"] = executar_multiplas_previsoes_correlacao_matriz_confusao(conjunto_dados, classificador=ClassificadorCorrelacaoCruzada, tamanho_treino_lista=[3192, 4494], repeticoes=30, save_fig=True, fig_folder=str(conjunto2_nome / corre_cruzada_pasta))
resultdict_2["kNN"] = executar_multiplas_previsoes_KNN_matriz_confusao(conjunto_dados, [1, 3, 5], [3192, 4494], treinos_lista=[treino_regular, treino_media], repeticoes=30, save_fig=True, fig_folder=str(conjunto2_nome / knn_pasta))
# resultdict_2["kNN"] = executar_multiplas_previsoes_KNN_matriz_confusao(conjunto_dados, [1], [3192, 4494], treinos_lista=[treino_media], repeticoes=30, save_fig=True, fig_folder=str(conjunto2_nome / knn_pasta))

print("exp2")
print(resultdict_2)
# Salvar o dicionário em um arquivo
with open('dict_2_3.pkl', 'wb') as f:
    pickle.dump(resultdict_2, f)

# resultdict_2={}

# conjunto_dados.reduzir_dimensionalidade(extended_slices=2)
# resultdict_2["CS"] = executar_multiplas_previsoes_correlacao_matriz_confusao(conjunto_dados, classificador=ClassificadorCorrelacao, tamanho_treino_lista=[3192, 4494], repeticoes=30, save_fig=True, fig_folder=str(conjunto2_nome / corre_simples_pasta))
# resultdict_2["CA"] = executar_multiplas_previsoes_correlacao_matriz_confusao(conjunto_dados, classificador=ClassificadorCorrelacaoCruzada, tamanho_treino_lista=[3192, 4494], repeticoes=30, save_fig=True, fig_folder=str(conjunto2_nome / corre_cruzada_pasta))
# resultdict_2["kNN"] = executar_multiplas_previsoes_KNN_matriz_confusao(conjunto_dados, [1, 3, 5], [3192, 4494], treinos_lista=[treino_regular, treino_media], repeticoes=30, save_fig=True, fig_folder=str(conjunto2_nome / knn_pasta))
# # resultdict_2["kNN"] = executar_multiplas_previsoes_KNN_matriz_confusao(conjunto_dados, [1], [3192, 4494], treinos_lista=[treino_media], repeticoes=30, save_fig=True, fig_folder=str(conjunto2_nome / knn_pasta))

# print("exp2_500")
# print(resultdict_2)
# # Salvar o dicionário em um arquivo
# with open('dict_3_500.pkl', 'wb') as f:
#     pickle.dump(resultdict_2, f)

# dict1="dict_1_500.pkl"
# dict2="dict_2_500.pkl"
# imprime_tabela_latex_precisao(dict1_caminho=dict1, dict2_caminho=dict2)
