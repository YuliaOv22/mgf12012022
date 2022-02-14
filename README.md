# Итоговый проект курса "Видеокурс от Megafon"

Стек:

+ ML: numpy, pandas, sklearn, dask
+ TRAIN DATA: `data_train.csv`, `features.csv.zip`
+ TEST DATA: `data_test.csv`

Задача: построить алгоритм, который для каждой пары пользователь-услуга определит вероятность подключения услуги. Бинарная классификация.

Выходные признаки + target:

- id (int64)
- vas_id (float64)
- buy_time (int64)

Преобразования признаков: 
+ StandardScaler
+ OneHotEncoder

Модель: LGBMClassifier.

Для осуществления предсказаний необходимо запустить файл `scoring.py`.

Файл `columnselector.py` содержит дополнительный класс для обработки признаков.
