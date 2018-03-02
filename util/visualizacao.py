# <script src="https://gist.github.com/zachguo/10296432.js"></script>
# Retirado do git ...
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from typing import List


def confusion_matrix_print(cm, labels, hide_zeroes=False, hide_diagonal=False, hide_threshold=None):
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


def confusion_matrix_plot(array=None, labels=List[str], normalizar=True, title='Confusion Matrix', **kwargs):
    array_aux = []
    if not normalizar:
        array_aux = array
    else:
        for linha in array:
            total_linha = sum(linha, 0)
            linha_aux = []
            for celula in linha:
                linha_aux.append((float(celula) / float(total_linha)) * 100.0)
            array_aux.append(linha_aux)

    df_cm = pd.DataFrame(array_aux, index=[i for i in labels],
                         columns=[i for i in labels])
    plt.figure(**kwargs)
    sn.heatmap(df_cm, annot=True)
    plt.title(title)
    plt.show()
