from sklearn.base import BaseEstimator, TransformerMixin
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LeakyReLU
from sklearn import preprocessing
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# All sklearn Transforms must have the `transform` and `fit` methods
class PrepareData(BaseEstimator, TransformerMixin):

    def __init__(self, drop_colums):
        self.drop_colums = drop_colums

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Primeiro realizamos a cópia do dataframe 'X' de entrada
        data = X.copy()
        # Retornamos um novo dataframe sem as colunas indesejadas
        data = self.nota_proporcional(data)
        data = self.horas_atividades(data)
        return data.drop(labels=self.drop_colums, axis='columns')

    ### Divide a nota de cada aluno pela média nas matérias
    def nota_proporcional(self, data):
        d = data.copy()

        for coluna in ["NOTA_DE", "NOTA_MF", "NOTA_GO", "NOTA_EM"]:
            media = data[coluna].median()
            for i in range(len(d[coluna])):
                d.at[i, coluna] = data.at[i, coluna] / media

        return d

    ### calcular a quantidade total de horas de atividades e sua proporcionalidade
    def horas_atividades(self, data):
        d = data.copy()
        atividades = list()
        soma_medias = data.H_AULA_PRES.median() + data.TAREFAS_ONLINE.median()
        for i in range(0, len(data["NOME"])):
            v = (data.at[i, "H_AULA_PRES"] + data.at[i, "TAREFAS_ONLINE"]) / soma_medias
            atividades.append(v)

        d = d.drop(["H_AULA_PRES", "TAREFAS_ONLINE"], axis=1)
        d['ATIVIDADES'] = atividades

        return d

#a classe que cria o modelo tem que ser um estimator... Não sei se precisa ser TranformerMixin também, maaasss
class CreateModel(BaseEstimator, TransformerMixin):

    def __init__(self,input_s, batch_size, epochs):
        self.epochs = epochs
        self.batch_size = batch_size
        self.input_s = input_s

        pass

    def put_X_Y(self,X,Y):
        self.X = X
        self.Y = Y
        return self

    def fit(self, x_bruto, y_bruto):
        x_prepared, y_prepared = self.prepare_input_output_transformer(self.X, self.Y)
        X_train, X_val, self.X_test, Y_train,Y_val, self.Y_test = self.split_data_he_he(x_prepared, y_prepared)

        model = self.create()
        self.history = model.fit(X_train, Y_train,
                            batch_size=self.epochs,
                            epochs=self.batch_size,
                            validation_data=(X_val, Y_val)
                            )
        return self

    def get_X_Y_test(self):
        return self.X_test, self.Y_test

    def create(self):
        model = Sequential()

        model.add(Dense(18,
                        activation=LeakyReLU(alpha=0.1),
                        input_shape=(self.input_s,)))
        model.add(Dense(36,
                        activation=LeakyReLU(alpha=0.1)))

        model.add(Dense(5,
                        activation='softmax'))

        model.compile(optimizer="adam",
                      loss='categorical_crossentropy',
                      metrics=['accuracy']
                      )
        return model

    def prepare_input_output_transformer(self, x_bruto, y_bruto):

        le = preprocessing.LabelEncoder()
        le.fit(y_bruto)

        y_prepared = to_categorical(le.transform(y_bruto), 5)

        min_max_scaler = preprocessing.MinMaxScaler()
        x_prepared = min_max_scaler.fit_transform(x_bruto)

        return x_prepared, y_prepared

    def split_data_he_he(self, x_prepared, y_prepared):
        X_train, X_val_test, Y_train, y_val_test = train_test_split(x_prepared, y_prepared, test_size=0.3,
                                                                    random_state=337)
        X_val, X_test, Y_val, Y_test = train_test_split(X_val_test, y_val_test, test_size=0.5) #avali

        return X_train, X_val, X_test, Y_train,Y_val, Y_test
