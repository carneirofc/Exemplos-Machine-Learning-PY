import copy
import pandas as pd

from keras.layers import Dense
from keras.models import Sequential

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from util.GraficoPlotCallback import PlotCallback
from util.print_cm import print_cm

dataset = pd.read_csv('Churn_Modelling.csv')
arquivo_aux = open("dados/info.txt", "w+")

x = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values.reshape(-1, 1)

geographySet = set()
geographyOriginal = copy.copy(x[:, 1])
geographyEncoder = LabelEncoder()
x[:, 1] = geographyEncoder.fit_transform(x[:, 1])
for s, v in zip(geographyOriginal, x[:, 1]):
    geographySet.add("{} {}".format(s, v))

arquivo_aux.writelines("Dados Entrada Col. 1 Geography (OneHotEncoder)\n")
for s in geographySet:
    arquivo_aux.writelines("{}\n".format(s))
sexoSet = set()

sexoOriginal = copy.copy(x[:, 2])
sexoEncoder = LabelEncoder()
x[:, 2] = sexoEncoder.fit_transform(x[:, 2])
for s, v in zip(sexoOriginal, x[:, 1]):
    sexoSet.add("{} {}".format(s, v))

arquivo_aux.writelines("Dados Entrada Col. 2 Sexo (Encoder)\n")
for s in sexoSet:
    arquivo_aux.writelines("{} \n".format(s))

arquivo_aux.writelines("Primeiro conj. Geography\n")
oneHotEncoder = OneHotEncoder(categorical_features=[1])
x = oneHotEncoder.fit_transform(x).toarray()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)

arquivo_aux.close()

plot_callback = PlotCallback()
rna = Sequential()
rna.add(Dense(9, activation='tanh', input_shape=(12,)))
rna.add(Dense(1, activation='linear'))

rna.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
rna.fit(x_train, y_train, epochs=200, batch_size=21)
#rna.fit(x_train, y_train, epochs=200, callbacks=[plot_callback], batch_size=21)

y_pred = rna.predict(x_test)
y_pred = (y_pred > 0.8)
cm = confusion_matrix(y_test, y_pred)
print_cm(cm, ['0', '1'])
