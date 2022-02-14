from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd


# Класс Трансформер для выделения колонок
class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)

        try:
            return X[self.columns]
        except KeyError:
            cols_error = list(set(self.columns) - set(X.columns))
            raise KeyError('DataFrame не содержит следующие колонки: %s' % cols_error)
