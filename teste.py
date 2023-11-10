from metrics.metrics import get_mean_minkowski_distance_between_dataframe, get_mean_distance_in_dataframe
from data.retrieve import explore_csv_dataframe
from models.predict import run_multiple_tests, run_multiple_tests_train_explicit, get_confusion
from models.split_train_test import treino_media, treino_regular, treino_regular_explicit, treino_media_explicit

# dict_dataframes = explore_csv_dataframe('dados')
# get_mean_minkowski_distance_between_dataframe(dict_dataframes, p=2)
# for name, df in dict_dataframes.items():
#     print(f'distância média {name}: {get_mean_distance_in_dataframe(df, p=2)}')

# run_multiple_tests([1], [0.8], [treino_media], path='dados_2_cargas')
#run_multiple_tests__train_explicit([1, 3, 5], [15, 30, 60, 120, 240, 480], [treino_regular_explicit, treino_media_explicit], path='dados_2_cargas')

run_multiple_tests([1], [0.2610458046], [treino_regular], path='dados_2_cargas_3', result_method=get_confusion)