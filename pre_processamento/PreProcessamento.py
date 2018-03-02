## Imports Libs
import copy

import numpy as np
import pandas as pd

# Import Dataset
dataset = pd.read_csv('dados\Data.csv')

# Dividindo as variáveis de entrada/saída
# Pegando tudo nas linhas, colunas -1 (a última coluna é de saída)
x_puro = dataset.iloc[:, :-1].values
y_puro = dataset.iloc[:, 3].values

# Preparanto o Dataset

# Substituindo os dados que faltam pela média na coluna em que ele está localizado
# Biblioteca de preprocessamento
from sklearn.preprocessing import Imputer

imputer = Imputer(missing_values="NaN", strategy='mean', axis=0, copy=True)
# 1:3 o limite superior é exluido, estou pegando as colunas 1 e 2
imputer.fit(x_puro[:, 1:3])
x_puro[:, 1:3] = imputer.transform(x_puro[:, 1:3])

# Tratamento dos dados categóricos
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

set_encode = set()
x_numerico = copy.copy(x_puro)

labelEncoderX = LabelEncoder()
x_numerico[:, 0] = labelEncoderX.fit_transform(x_puro[:, 0])

arquivo_aux = open("dados\info.txt", "w+")
arquivo_aux.writelines("Dados de Entrada. Info Países. Col.0\n")

for s, n in zip(x_puro[:, 0], x_numerico[:, 0]):
    set_encode.add("{} {}".format(s, n))
for e in set_encode:
    arquivo_aux.writelines(e + "\n")
set_encode.clear()
arquivo_aux.writelines("----------------------------------------\n")
# As colunas enumeradas em ordem a,lfabética pelo LabelEncoder serão trasnformadas em colunas
# Ficando uma coluna para cada país
oneHotEncoder = OneHotEncoder(categorical_features=[0])
x_numerico = oneHotEncoder.fit_transform(x_numerico).toarray()

# São enumerados por ordem alfabética
labelEncoderY = LabelEncoder()

y_numerico = labelEncoderY.fit_transform(y_puro)
for label, val in zip(y_puro, y_numerico):
    set_encode.add("{} {}".format(label, val))

# Divisão em conjunto de treinamento e conjunto de testes
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(x_numerico, y_numerico, test_size=0.25, random_state=0)

# Normalização dos dados
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Para saídas categoricas não precisa normalizar o Y
# para aproximação sim !!!
np.savetxt("dados\dados-teste-entrada.csv", X_test, delimiter=' , ')
np.savetxt("dados\dados-treino-entrada.csv", X_train, delimiter=' , ')

np.savetxt("dados\dados-teste-saida.csv", Y_test, delimiter=' , ')
np.savetxt("dados\dados-treino-saida.csv", Y_train, delimiter=' , ')

arquivo_aux.writelines("Conversao coluna saida. Col.0\n")
for a in set_encode:
    arquivo_aux.writelines(a)
    arquivo_aux.writelines("\n")
arquivo_aux.close()

# Exportando com o pandas ele joga as coordenadas junto, adicionando uma linha e uma coluna...
# pd.DataFrame(X_train).to_csv("dados-pandas-teste.csv")
