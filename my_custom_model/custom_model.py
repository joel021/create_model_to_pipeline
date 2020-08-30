from sklearn.base import BaseEstimator, TransformerMixin
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import pandas as pd
import numpy as np
import math
from sklearn.impute import SimpleImputer
from keras.utils import to_categorical



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

    def fit(self, X, y=None):
        return self

    # caso ainda existam valores nulos depois de passar pelas transformações anteriores
    def transform(self, X):
        si = SimpleImputer(
            missing_values=np.nan,  # os valores faltantes são do tipo ``np.nan`` (padrão Pandas)
            strategy='mean',  # a estratégia escolhida é a alteração do valor faltante pela média
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


class SelectFeatures(BaseEstimator, TransformerMixin):

    def __init__(self, features):
        self.features = features

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X[self.features]


class MedianRow(BaseEstimator, TransformerMixin):

    def __init__(self, columns, name='SCORE'):
        self.columns = columns
        self.name = name

    def fit(self, X, y):
        self.y = y
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

class DenseModel(BaseEstimator, TransformerMixin):

    def __init__(self, input_s, num_classes, batch_size, epochs, activation):
        self.epochs = epochs
        self.batch_size = batch_size
        self.input_s = input_s
        self.activation = activation
        self.num_classes = num_classes

    def fit(self, X, Y):
        self.Y = Y
        return self

    def transform(self, X):
        # separa os dados de treinamento dos de validação
        # codifica Y para categorias e codifica as categorias para vetores do tipo : [1,0,0,0,0] = classe 1, ...
        X_train, X_val_test, Y_train, Y_val_test = train_test_split(X, to_categorical(self.label_encoder(self.Y),
                                                                                      self.num_classes), test_size=0.4,
                                                                    random_state=337)  # 40% para validação e teste
        X_test, X_val, Y_test, Y_val = train_test_split(X_val_test, Y_val_test, test_size=0.5,
                                                        random_state=337)  # 50% para validação

        self.model = self.create()  # cria o modelo
        self.history = self.model.fit(X_train, Y_train,
                                      batch_size=self.batch_size,
                                      epochs=self.epochs,
                                      validation_data=(X_val, Y_val)
                                      )  # treina
        data = self.predict(X)  # tem que ser com todo X
        return data

    def create(self):
        model = Sequential()

        model.add(Dense(2 * self.input_s,
                        activation=self.activation,
                        input_shape=(self.input_s,)))

        model.add(Dense(2 * self.input_s,
                        activation=self.activation))

        model.add(Dense(self.num_classes,
                        activation='softmax'))

        model.compile(optimizer="adam",
                      loss='categorical_crossentropy',
                      metrics=['accuracy']
                      )
        return model

    def predict(self, x):
        # return self.reverse_predicitions_to_classification_score(self.model.predict(x))
        predictions = self.model.predict(x)

        muito_bom = list()
        dificuldade = list()
        exatas = list()
        humanas = list()
        excelente = list()

        for prediction in predictions:

            # rodar classes e predicitons
            for i in range(len(self.classes)):

                if 'DIFICULDADE' in self.classes[i]:
                    dificuldade.append(prediction[i])
                if 'EXCELENTE' in self.classes[i]:
                    excelente.append(prediction[i])
                if 'MUITO_BOM' in self.classes[i]:
                    muito_bom.append(prediction[i])
                if 'HUMANAS' in self.classes[i]:
                    humanas.append(prediction[i])
                if 'EXATAS' in self.classes[i]:
                    exatas.append(prediction[i])

        data = x.copy()
        data['DIFICULDADE'] = dificuldade
        data['EXCELENTE'] = excelente
        data['MUITO_BOM'] = muito_bom
        data['EXATAS'] = exatas
        data['HUMANAS'] = humanas

        return data

    def get_dense_model(self):
        return self.model

    def get_history(self):
        return self.history

    def label_encoder(self, y):
        le = preprocessing.LabelEncoder()
        le.fit(y)
        self.classes = le.classes_
        return le.transform(y)

    def get_classes(self):
        return self.classes

    # uma classificação vem na forma: [0,0,1.0,0,0]
    def reverse_predicitions_to_classification_score(self, predictions):
        score_classe = list()
        classe = list()
        for prediction in predictions:

            # prediction tem a quantidade target.
            # em particular, são 5: "DIFICULDADE","EXATAS","MUITO BOM","EXCELENTE" e "HUMANAS"
            maior = prediction[0]  # pega o primeiro número
            p = 0
            for i in range(1, len(prediction)):
                if prediction[i] > maior:  # verifica se existe algum maior que ele
                    maior = prediction[i]
                    p = i
            score_classe.append(maior)
            classe.append(self.classes[p])

        classe = self.label_encoder(classe)  # categoriza as targets
        return classe, score_classe  # retorna uma lista na forma: [classe1,classe2,...] assim: [0.]