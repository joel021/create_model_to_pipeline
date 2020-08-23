from sklearn.base import BaseEstimator, TransformerMixin
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from keras.utils import to_categorical

class DenseModel(BaseEstimator, TransformerMixin):

    def __init__(self, input_s, num_classes, batch_size, epochs, activation, prepare_data):
        self.epochs = epochs
        self.batch_size = batch_size
        self.input_s = input_s
        self.activation = activation
        self.num_classes = num_classes
        self.prepare_data = prepare_data

    def fit(self, X, Y):
        X_train, X_val, Y_train, Y_val = train_test_split(X, self.transform_label(Y), test_size=0.2, random_state=337)
        self.model = self.create()
        self.history = self.model.fit(X_train, Y_train,
                                      batch_size=self.epochs,
                                      epochs=self.batch_size,
                                      validation_data=(X_val, Y_val)
                                      )
        return self

    def create(self):
        model = Sequential()

        model.add(Dense(2 * self.input_s,
                        activation=self.activation,
                        input_shape=(self.input_s,)))

        model.add(Dense(4 * self.input_s,
                        activation=self.activation))

        model.add(Dense(self.num_classes,
                        activation='softmax'))

        model.compile(optimizer="adam",
                      loss='categorical_crossentropy',
                      metrics=['accuracy']
                      )
        return model

    def predict(self, X):
        data = self.prepare_data.transform(X.copy())
        return self.reverse_predicitions_to_label(self.model.predict(data))

    def evaluate(self, X_test, Y_test):
        data = self.prepare_data.transform(X_test.copy())
        return self.model.evaluate(data, self.transform_label(Y_test))

    def get_dense_model(self):
        return self.model

    def get_history(self):
        return self.history

    def transform_label(self, y):
        le = preprocessing.LabelEncoder()
        le.fit(y)
        self.classes = le.classes_
        y_prepared = to_categorical(le.transform(y), self.num_classes)
        return y_prepared

    def get_classes(self):
        return self.classes

    def reverse_predicitions_to_label(self, predictions):
        labels = list()
        for prediction in predictions:

            # prediction tem a quantidade target.
            # em particular, são 5: "DIFICULDADE","EXATAS","MUITO BOM"
            for i in range(0, len(prediction)):
                if prediction[i] >= 0.5:
                    labels.append(self.classes[i])
        return labels
