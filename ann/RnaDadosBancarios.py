import copy
import pandas as pd

from keras.layers import Dense
from keras.models import Sequential

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from util.GraficoPlotCallback import PlotCallback
from util.visualizacao import confusion_matrix_print
from util.visualizacao import confusion_matrix_plot

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
for s, v in zip(sexoOriginal, x[:, 2]):
    sexoSet.add("{} {}".format(s, v))

arquivo_aux.writelines("Dados Entrada Col. 2 Sexo (Encoder)\n")
for s in sexoSet:
    arquivo_aux.writelines("{} \n".format(s))

arquivo_aux.writelines("Primeiro conj. Geography\n")
oneHotEncoder = OneHotEncoder(categorical_features=[1])
x = oneHotEncoder.fit_transform(x).toarray()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)

sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)

arquivo_aux.close()

"""
    todo: Comparar com a implementação do curso e observar os motivos da baixa precisão.
    estou tendo dificuldade em conseguir valores acima de 86%, respostas verdadeiras pouco precisas ...    
"""
plot_callback = PlotCallback()
rna = Sequential()
rna.add(Dense(12, activation='tanh', input_shape=(12,)))
rna.add(Dense(12, activation='tanh', input_shape=(12,)))
# rna.add(Dense(6, activation='tanh', input_shape=(12,)))
rna.add(Dense(1, activation='linear'))

# rna.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
rna.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["accuracy"])
rna.fit(x_train, y_train, epochs=100, batch_size=25)

tolerancia = 0.25
y_pred = rna.predict(x_test)
y_pred = (y_pred > tolerancia)
cm = confusion_matrix(y_test, y_pred)
confusion_matrix_print(cm, ['Ficou', 'Saiu'])
confusion_matrix_plot(cm, ['Ficou', 'Saiu'],
                      title='Cliente com {}% de chances de sair'.format(tolerancia * 100.0))
