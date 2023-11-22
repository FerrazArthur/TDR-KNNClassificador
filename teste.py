from modelos.previsao import executar_multiplos_testes, executar_multiplos_testes_matriz_confusao
from metricas.metricas import obter_distancia_media_minkowski_entre_dataframe, obter_distancia_media_minkowski_entre_media_dataframe, obter_linha_maior_distancia_minkowski_entre_dataframes, obter_vetor_distancias_a_media_dataframe
from visualizacao.visualizacao import imprime_distribuicao_distancias
from modelos.conjuntos_dados import treino_regular, treino_media
from modelos.dados import Dados
from dados.escrita import salvar_dataframes_csv
# dict_dataframes = explore_csv_dataframe('dados')
# get_mean_minkowski_distance_between_dataframe(dict_dataframes, p=2)
# for name, df in dict_dataframes.items():
#     print(f'distância média {name}: {get_mean_distance_in_dataframe(df, p=2)}')

# run_multiple_tests([1], [0.8], [treino_media], path='dados_2_cargas')
#run_multiple_tests__train_explicit([1, 3, 5], [15, 30, 60, 120, 240, 480], [treino_regular_explicit, treino_media_explicit], path='dados_2_cargas')
conjunto_dados = Dados('dados_2_cargas')

conjunto_dados.remover_outliers(3)
conjunto_dados.normalizar_amostras()

linha_pior, _, _ = obter_linha_maior_distancia_minkowski_entre_dataframes(conjunto_dados.dicionario_dados, p=2)
imprime_distribuicao_distancias(obter_vetor_distancias_a_media_dataframe(conjunto_dados.dicionario_dados[linha_pior]), ddof=1)
# for classe in conjunto_dados.classes_lista:
#     imprime_distribuicao_distancias(obter_vetor_distancias_a_media_dataframe(conjunto_dados.dicionario_dados[classe]), ddof=1)


# conjunto_dados = Dados('dados_2_cargas')

# print("Removendo outliers utilizando 3 como limitante do z-score:\n")

# print("tamanho do conjunto de dados inicial: ", conjunto_dados.num_amostras)
# print(obter_linha_maior_distancia_minkowski_entre_dataframes(conjunto_dados, p=2))
# obter_distancia_media_minkowski_entre_media_dataframe(conjunto_dados.dicionario_dados, p=2)

# conjunto_dados.remover_outliers(3)
# print("tamanho do conjunto de dados após remover os outliers: ", conjunto_dados.num_amostras)
# print(obter_linha_maior_distancia_minkowski_entre_dataframes(conjunto_dados, p=2))

# conjunto_dados.normalizar_amostras()
# print("tamanho do conjunto de dados após realizar a normalização: ", conjunto_dados.num_amostras)
# print(obter_linha_maior_distancia_minkowski_entre_dataframes(conjunto_dados, p=2))

# obter_distancia_media_minkowski_entre_media_dataframe(conjunto_dados, p=2)

#salvar_dataframes_csv(conjunto_dados.dicionario_dados, '2_carga_5_dp')
#print(conjunto_dados.num_classes)
#obter_distancia_media_minkowski_entre_dataframe(conjunto_dados, p=2)
#obter_distancia_media_minkowski_entre_media_dataframe(conjunto_dados, p=2)
#conjunto_dados.normalizar_amostras()
#executar_multiplos_testes(conjunto_dados, [1, 3, 5], [966, 1932], [treino_regular, treino_media])
#executar_multiplos_testes(conjunto_dados, [1, 3, 5], [0.05, 0.1, 0.3, 1932], [treino_regular, treino_media])
#executar_multiplos_testes_matriz_confusao(conjunto_dados, [1, 3, 5], [0.05, 0.1, 0.3, 1932], treinos_lista=[treino_regular, treino_media], repeticoes=10, safe_fig=True)
#executar_multiplos_testes_matriz_confusao(conjunto_dados, [1, 3, 5], [483, 966, 1932], treinos_lista=[treino_media, treino_regular], repeticoes=10, save_fig=True)
#executar_multiplos_testes_matriz_confusao(conjunto_dados, [1, 3, 5], [32, 64, 120, 240, 480], treinos_lista=[treino_regular, treino_media], repeticoes=10, save_fig=True)
#executar_multiplos_testes_matriz_confusao(conjunto_dados, [1], [630, 1050, 1260, 1470, 1932], treinos_lista=[treino_regular], repeticoes=1, save_fig=True)