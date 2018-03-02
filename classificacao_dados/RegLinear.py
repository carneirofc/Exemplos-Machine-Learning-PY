import copy

import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from util.visualizacao import confusion_matrix_plot


def grafico_2d(x_set=None, y_set=None, step_plot=0.01, model=None, title='Title', label_x='x',
               label_y='y'):
    if model is not None:
        x1, x2 = np.meshgrid(np.arange(start=x_set[:, 0].min() - 1, stop=x_set[:, 0].max() + 1, step=step_plot),
                             np.arange(start=x_set[:, 1].min() - 1, stop=x_set[:, 1].max() + 1, step=step_plot))
        plt.contourf(x1, x2, model.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
                     alpha=0.75, cmap=ListedColormap(('red', 'green')))
        plt.xlim(x1.min(), x1.max())
        plt.ylim(x2.min(), x2.max())
        for i, j in enumerate(np.unique(y_set)):
            plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
                        c=ListedColormap(('red', 'green'))(i), label=j)
        plt.title(title)
        plt.xlabel(label_x)
        plt.ylabel(label_y)
        plt.legend()
        plt.show()


dataset = pd.read_csv("Social_Network_Ads.csv")
x_puro = dataset.iloc[:, :-1].values
y_puro = dataset.iloc[:, -1].values

info = set()

x_numerico = copy.copy(x_puro)
y_numerico = copy.copy(y_puro)
genero_encoder = LabelEncoder()
x_numerico[:, 1] = genero_encoder.fit_transform(x_puro[:, 1])

info.add("Dados Entrada Col.1")
for string, num in zip(x_puro[:, 1], x_numerico[:, 1]):
    info.add("{} {}".format(string, num))

x_treino, x_teste, y_treino, y_teste = train_test_split(x_numerico, y_numerico, test_size=.25)

x_scaler = StandardScaler()
x_treino = x_scaler.fit_transform(x_treino)
x_teste = x_scaler.fit_transform(x_teste)

# Criando uma regressão logística

log_reg = LogisticRegression()
log_reg.fit(x_treino[:, [2, 3]], y_treino)
pred_res = log_reg.predict(x_teste[:, [2, 3]])

# Confusion matrix

cm = confusion_matrix(y_teste, pred_res)
print(cm)
confusion_matrix_plot(cm, ['Sim', 'Não'])
# Gráfico
# Visualising the Test set results
#X_set, y_set = x_teste[:, [2, 3]], y_teste
#grafico_2d(x_set=x_teste[:, [2, 3]], y_set=y_teste, step_plot=0.001, model=log_reg)
