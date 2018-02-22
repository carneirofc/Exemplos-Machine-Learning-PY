import pandas as pd
import copy

dataset = pd.read_csv("Social_Network_Ads.csv")
x_puro = dataset.iloc[:, :-1].values
y_puro = dataset.iloc[:, -1].values

from sklearn.preprocessing import LabelEncoder

info = set()

x_numerico = copy.copy(x_puro)
y_numerico = copy.copy(y_puro)
genero_encoder = LabelEncoder()
x_numerico[:, 1] = genero_encoder.fit_transform(x_puro[:, 1])

info.add("Dados Entrada Col.1")
for string, num in zip(x_puro[:, 1], x_numerico[:, 1]):
    info.add("{} {}".format(string, num))

from sklearn.model_selection import train_test_split

x_treino, x_teste, y_treino, y_teste = train_test_split(x_numerico, y_numerico, test_size=.25)

from sklearn.preprocessing import StandardScaler

x_scaler = StandardScaler()
x_treino = x_scaler.fit_transform(x_treino)
x_teste = x_scaler.fit_transform(x_teste)

# Criando uma regressão logística
from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression()
log_reg.fit(x_treino[:, [2, 3]], y_treino)
pred_res = log_reg.predict(x_teste[:, [2, 3]])

# Confusion matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_teste, pred_res)
print(cm)

# Gráfico
# Visualising the Test set results
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

X_set, y_set = x_teste[:, [2, 3]], y_teste
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
                     np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))
plt.contourf(X1, X2, log_reg.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha=0.75, cmap=ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c=ListedColormap(('red', 'green'))(i), label=j)
plt.title('Classifier (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# Visualizando dados
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, random_state=0)
markers = ('u', 'g', 'a', 's')
color_map = {0: 'red', 1: 'blue', 2: 'green', 3: 'pink'}
plt.figure()
for idx, cl in enumerate(np.unique(dataset)):
    plt.scatter(x=dataset[cl, 0], y=dataset[cl, 1], c=color_map, marker=markers, label=cl)
plt.show()
