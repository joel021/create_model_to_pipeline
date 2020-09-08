from sklearn.base import BaseEstimator, TransformerMixin
from sklearn import preprocessing
import pandas as pd
import numpy as np
import math
from sklearn.impute import SimpleImputer
from imblearn.under_sampling import RandomUnderSampler
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeRegressor

class MegeTwoColumns(BaseEstimator, TransformerMixin):

    def __init__(self ,columns ,name):
        self.columns = columns
        self.name = name

    def fit(self ,X ,y=None):
        return self

    def transform(self, data):
        d = data.copy()

        new_column = np.array(data[self.columns[0]].values +data[self.columns[1]].values)
        new_column = new_column /2.0
        d[self.name] = new_column

        return d


class RemoveNanValues(BaseEstimator, TransformerMixin):

    def __init__(self, columns=None, strategy='mean', fill_value=None):
        self.strategy = strategy
        self.fill_value = fill_value
        self.columns = columns

    def fit(self, X, y=None):
        return self

    # caso ainda existam valores nulos depois de passar pelas transformações anteriores
    def transform(self, X):
        data = X.copy()

        if self.columns:
            data = data.drop(columns=self.columns, axis=1)  # remove as colunas escolhidas
            trans_data = self.transform_data(
                X[self.columns])  # recebe o data frame transformado somente com as colunas escolhidas

            for column in self.columns:  # insere as colunas transformadas no dataset novamente
                data[column] = trans_data[column]
        else:
            data = self.transform_data(
                data)  # caso não tenha preferência de colunas, é só enviar o dataset inteiro e receber ele transformado

        return data

    def transform_data(self, X):
        si = SimpleImputer(
            missing_values=np.nan,  # os valores faltantes são do tipo ``np.nan`` (padrão Pandas)
            strategy=self.strategy,  # a estratégia escolhida é a alteração do valor faltante pela média
            fill_value=self.fill_value,
            verbose=0,
            copy=True
        )

        si.fit(X=X)
        # Reconstrução de um novo DataFrame Pandas com o conjunto imputado (df_data_3)
        data = pd.DataFrame.from_records(
            data=si.transform(X.copy()),  # o resultado SimpleImputer.transform(<<pandas dataframe>>) é lista de listas
            columns=X.columns  # as colunas originais devem ser conservadas nessa transformação
        )

        return data

class MedianRow(BaseEstimator, TransformerMixin):

    def __init__(self, columns, name='SCORE'):
        self.columns = columns
        self.name = name

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        data = X.copy()
        data = data[self.columns]
        score = list()
        for i in range(len(data[self.columns[0]])):  # de o ao tamanho -1
            row = data.iloc[i]
            score.append(row.median())

        data = X.copy()
        data[self.name] = score
        return data


class FeaturesDificuldade(BaseEstimator, TransformerMixin):

    def fit(self, X, y):
        return self

    def transform(self, X):
        data = X.copy()

        m_M_H = data.M_H.median()  # média matérias humanas
        m_H_E = data.M_E.median()  # media matérias exatas
        p_dificuldade = list()

        for i in range(len(data.SCORE)):  # score só para saber a quantidade
            row = data.iloc[i]
            # probabilidade de ser classe dificuldade pelo score (o modelo não está entendo SCORE direito)
            p1 = 1 - 0.1667 * row.SCORE
            if p1 < 0.0:
                p1 = 0.0

            # se ele for bom em exatas e humanas, a probabilidade cai
            p2 = 1.0
            r_square = math.pow(row.M_H / data.M_H.max(0), 2.0) + math.pow(row.M_E / data.M_E.max(0), 2.0)
            if r_square != 0.0:
                p2 = 1.0 / r_square  # equação do parabolóide côncavo pra cima. A probabilidade aumenta quadraticamente quando aumenta a probabilidade de ser grupo EXATAS e HUMANAS
                if p2 > 1.0:
                    p2 = 1.0

            p3 = 1.0
            # se a pessoa é ruim em uma matéria de humanas e em matéria de exatas, a probabilidade dele ser do grupo DIFICULDADE aumenta
            if row.M_H < m_M_H and row.M_E < m_H_E:
                if row.M_H != 0.0 or row.M_E != 0.0:
                    p3 = 1.0 / (math.pow(row.M_H, 2.0) + math.pow(row.M_E, 2.0))
                    if p3 > 1.0:
                        p3 = 1.0

            p_d = (p1 + p2 + p3) / 3.0
            p_dificuldade.append(p_d)

        data["P_DIFICULDADE"] = p_dificuldade
        return data


class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y):
        return self

    def transform(self, X):
        return X.drop(columns=self.columns, axis=1)


class CustomStandardScaler(BaseEstimator, TransformerMixin):

    def fit(self, X, y):
        return self

    def transform(self, X):
        standard_scaler = preprocessing.StandardScaler().fit(X)

        data = pd.DataFrame().from_records(
            data=standard_scaler.transform(X),
            columns=X.columns
        )

        return data


class CustomRemoveNanValues(BaseEstimator, TransformerMixin):
    def __init__(self, columns, categore_columns, categorical):
        self.columns = columns  # columns when has NaN values
        self.categore_columns = categore_columns
        self.categorical = categorical

    def fit(self, X, y):
        self.y = y
        return self

    def transform(self, X):

        # get the columns where has NaN values
        if self.columns is None:
            self.columns = X.isnull().sum(axis=0)  # list with null values in columns
            self.columns = self.columns.loc[self.columns != 0]  # only the rows when nan value is more than 0
            self.columns = self.columns.index

        complete_data = X.copy()
        complete_data[self.y.name] = self.y

        for column in self.columns:  # columns with nan values

            data_X = complete_data.copy()
            data_X = pd.get_dummies(data_X, columns=self.categore_columns)

            X_to_pred_ = data_X[data_X[column].isnull()]  # all rows when has nan in column 'column'
            X_to_pred_ = X_to_pred_.drop(columns=self.columns,
                                         axis=1)  # remove column that will be target: selected column

            # prepare train data: remove all rows with nan values
            X_train_ = data_X.dropna(axis='index', how='any',
                                     subset=data_X.columns)  # all rows when has not nan values in selected columns
            Y_train_ = X_train_[column]
            X_train_ = X_train_.drop(columns=self.columns, axis=1)

            # classify and predict values
            if self.categorical:
                # balance data
                X_train_, Y_train_ = RandomUnderSampler().fit_resample(X_train_, Y_train_)
                model_predict = XGBClassifier().fit(X_train_, Y_train_)
            else:
                model_predict = DecisionTreeRegressor(max_depth=6).fit(X_train_, Y_train_)

            y_pred_ = model_predict.predict(X_to_pred_)

            # insert data predicted in data
            complete_data.loc[complete_data[column].isnull(), column] = y_pred_

        complete_data = complete_data.drop(columns=self.y.name, axis=1)
        return complete_data