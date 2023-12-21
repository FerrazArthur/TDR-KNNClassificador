from visualizacao.tabelas import imprime_tabela_latex_precisao, imprime_tabela_latex_mem_time_C, imprime_tabela_latex_mem_time_kNN

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