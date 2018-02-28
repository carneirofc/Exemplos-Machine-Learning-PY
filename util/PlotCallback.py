import pandas as pd
import keras
import copy

from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers

from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


class PlotCallback(keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.fig = plt.figure()

    def on_train_begin(self, logs={}):
        self.ite = 0
        self.logs = []
        self.y_loss = []
        self.y_val_loss = []
        self.x_loss = []
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        self.ite += 1
        self.logs.append(logs)
        self.x_loss.append(self.ite)
        self.y_loss.append(logs.get('loss'))
        self.y_val_loss.append(logs.get('val_loss'))

    def on_train_end(self, logs={}):
        plt.plot(self.x_loss, self.y_loss, label="loss")
        plt.plot(self.x_loss, self.y_val_loss, label="loss")

        plt.legend()
        self.fig.show()


dataset = pd.read_csv('Churn_Modelling.csv')
info_txt = open("dados\info.txt", "w+")

x = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values.reshape(-1, 1)

geographySet = set()
geographyOriginal = copy.copy(x[:, 1])
geographyEncoder = LabelEncoder()
x[:, 1] = geographyEncoder.fit_transform(x[:, 1])
for s, v in zip(geographyOriginal, x[:, 1]):
    geographySet.add("{} {}".format(s, v))

info_txt.writelines("Dados Entrada Col. 1 Geography (OneHotEncoder)\n")
for s in geographySet:
    info_txt.writelines("{}\n".format(s))
sexoSet = set()
sexoOriginal = copy.copy(x[:, 2])
sexoEncoder = LabelEncoder()
x[:, 2] = sexoEncoder.fit_transform(x[:, 2])
for s, v in zip(sexoOriginal, x[:, 1]):
    sexoSet.add("{} {}".format(s, v))

info_txt.writelines("Dados Entrada Col. 2 Sexo (Encoder)\n")
for s in sexoSet:
    info_txt.writelines("{} \n".format(s))

info_txt.writelines("Primeiro conj. Geography\n")
oneHotEncoder = OneHotEncoder(categorical_features=[1])
x = oneHotEncoder.fit_transform(x).toarray()
info_txt.close()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

standard_scalar = StandardScaler()
x_train = standard_scalar.fit_transform(x_train)
x_test = standard_scalar.transform(x_test)

rna = Sequential()
rna.add(Dense(6, activation='tanh', input_shape=(12,)))
rna.add(Dense(6, activation='tanh', input_shape=(12,)))
rna.add(Dense(1, activation='tanh'))
# opt = optimizers.Adam
rna.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

call = PlotCallback()
rna.fit(x_train, y_train, epochs=250, callbacks=[call])

y_pred = rna.predict(x_test)
y_pred = (y_pred > 0.5)

cm = confusion_matrix(y_test, y_pred)
cm
