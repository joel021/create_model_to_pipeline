from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from sklearn.impute import SimpleImputer
import pandas as pd

class PrepareData(BaseEstimator, TransformerMixin):

    def __init__(self, drop_colums, proporcional_in_columns, merge_comlumns_to_name):
        self.drop_colums = drop_colums
        self.proporcional_in_columns = proporcional_in_columns
        self.merge_comlumns_to_name = merge_comlumns_to_name

    def fit(self, X, y=None):
        return self

    def proporcional_in_column(self, data):
        d = data.copy()
        for column in self.proporcional_in_columns:
            median = data[column].median()

            numpy_column = data[column]
            numpy_colum = np.array(numpy_column)

            for i in range(len(numpy_colum)):
                numpy_colum[i] = numpy_colum[i] / median

            d[column] = numpy_colum
        return d

    def merge_two_comlumns(self, data):

        d = pd.DataFrame(data)
        new_column = list()
        column1 = data[self.merge_comlumns_to_name[0]] #merge
        column2 = data[self.merge_comlumns_to_name[1]] #merge
        column1 = column1.median()
        column2 = column2.median()
        sum_median = column1 + column2
        column1 = column2 = None

        numpy_column1 = np.array(data[self.merge_comlumns_to_name[0]])
        numpy_column2 = np.array(data[self.merge_comlumns_to_name[1]])

        for i in range(0, len(data[self.merge_comlumns_to_name[0]])):
            v = (numpy_column1[i] + numpy_column2[i]) / sum_median
            new_column.append(v)

        d = d.drop([self.merge_comlumns_to_name[0], self.merge_comlumns_to_name[1]], axis=1)
        d[self.merge_comlumns_to_name[2]] = new_column

        return d

    def transform(self, X):
        data = self.proporcional_in_column(X)
        data = self.merge_two_comlumns(data)
        data = self.nan_values(data)
        return data.drop(labels=self.drop_colums, axis='columns')

    def nan_values(self, X):
        si = SimpleImputer(
            missing_values=np.nan,  # os valores faltantes são do tipo ``np.nan`` (padrão Pandas)
            strategy='constant',  # a estratégia escolhida é a alteração do valor faltante por uma constante
            fill_value=0,  # a constante que será usada para preenchimento dos valores faltantes é um int64=0.
            verbose=0,
            copy=True
        )

        si.fit(X=X)

        # Reconstrução de um novo DataFrame Pandas com o conjunto imputado (df_data_3)
        data = pd.DataFrame.from_records(
            data=si.transform(
                X=X
            ),  # o resultado SimpleImputer.transform(<<pandas dataframe>>) é lista de listas
            columns=X.columns  # as colunas originais devem ser conservadas nessa transformação
        )
        return data