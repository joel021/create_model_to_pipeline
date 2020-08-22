from sklearn.base import BaseEstimator, TransformerMixin
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LeakyReLU
from sklearn import preprocessing
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

class CreateModel(BaseEstimator, TransformerMixin):

    def __init__(self,input_s, batch_size, epochs):
        self.epochs = epochs
        self.batch_size = batch_size
        self.input_s = input_s

    def put_X_Y(self,X,Y,test_size):
        self.X=X
        self.Y=Y
        self.test_size=test_size
        return self

    def fit(self, x_bruto, y_bruto):
        x_prepared, y_prepared = self.prepare_input_output_transformer(self.X, self.Y)
        X_train, X_val, self.X_test, Y_train,Y_val, self.Y_test = self.split_data_he_he(x_prepared, y_prepared)

        self.model = self.create()
        self.history = self.model.fit(X_train, Y_train,
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
        X_train, X_val_test, Y_train, y_val_test = train_test_split(x_prepared, y_prepared, test_size=self.test_size,
                                                                    random_state=337)
        X_val, X_test, Y_val, Y_test = train_test_split(X_val_test, y_val_test, test_size=0.5) #avali

        return X_train, X_val, X_test, Y_train,Y_val, Y_test

    def predict(self,X):
        return self.model.predict(X)

    def evaluate(self,X_test,Y_test):
        return self.model.evaluate(X_test,Y_test)

    def get_dense_model(self):
        return self.model

    def get_history(self):
        return self.history