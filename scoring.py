# Подключим библиотеки
import pandas as pd
import pickle
import numpy as np
import dask.dataframe as dd
from columnselector import ColumnSelector

# Переменные путей
PATH_MODEL = 'lgbm_model.pickle'
PATH_FEATURES = 'features.csv'
PATH_DATA_TEST = 'data_test.csv'
PATH_ANSWERS = 'answers_test.csv'
PATH_DATA_WITHOUT_PRED = 'data_without_predictions.csv'

# Обработка входных данных
print('Загружаем предобученную модель...')
with open(PATH_MODEL, 'rb') as file_1:
    lgbm_model = pickle.load(file_1)

print(lgbm_model)
print('Считываем файл с признаками...')
data_features = dd.read_csv(PATH_FEATURES, sep='\t')
data_test = pd.read_csv(PATH_DATA_TEST).drop('Unnamed: 0', axis=1)

print('Отбираем признаки для присоединения датасетов...')
id_features = data_features[data_features['id'].isin(data_test['id']) == True].compute()
id_features = id_features.drop('Unnamed: 0', axis=1)

print('Сортируем данные...')
data_test = data_test.sort_values(by='buy_time')
id_features = id_features.sort_values(by='buy_time')

print('Создаем новый датасет для предсказаний...')
id_merge_test = pd.merge_asof(data_test, id_features, on='buy_time', by='id', direction='backward')
id_merge_test = id_merge_test[id_merge_test['0'].notnull()]

# Предсказания
print('Предсказываем...')
y_pred_test = lgbm_model['model'].predict_proba(id_merge_test)[:, 1]
answers = id_merge_test[data_test.columns].copy()

answers['target'] = pd.DataFrame({'target': y_pred_test})
answers.to_csv(PATH_ANSWERS, index=False)
print('Файл с предсказаниями записан /answers_test.')
print()
print(answers.head())

if answers.shape != data_test.shape:
    no_y_pred = data_test[data_test['id'].isin(answers['id']) == False]
    no_y_pred.to_csv(PATH_DATA_WITHOUT_PRED, index=False)
    print()
    print(
        'Для некоторых строк нет предсказаний, так как нет профиля абонента.\n'
        'Смотри файл /data_without_predictions.csv.')
