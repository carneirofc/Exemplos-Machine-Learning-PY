# <script src="https://gist.github.com/zachguo/10296432.js"></script>
# Retirado do git ...
from sklearn.base import ClassifierMixin


def print_cm(cm, labels, hide_zeroes=False, hide_diagonal=False, hide_threshold=None):
    """pretty print for confusion matrixes"""
    columnwidth = max([len(x) for x in labels] + [5])  # 5 is value length
    empty_cell = " " * columnwidth
    # Print header
    print("    " + empty_cell, end=" ")
    for label in labels:
        print("%{0}s".format(columnwidth) % label, end=" ")
    print()
    # Print rows
    for i, label1 in enumerate(labels):
        print("    %{0}s".format(columnwidth) % label1, end=" ")
        for j in range(len(labels)):
            cell = "%{0}.1f".format(columnwidth) % cm[i, j]
            if hide_zeroes:
                cell = cell if float(cm[i, j]) != 0 else empty_cell
            if hide_diagonal:
                cell = cell if i != j else empty_cell
            if hide_threshold:
                cell = cell if cm[i, j] > hide_threshold else empty_cell
            print(cell, end=" ")
        print()


# Gráfico
# Visualising the Test set results
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np


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
