import keras
from matplotlib import pyplot as plt


# Todo: Implementar a atualização da imagem em tempo real ....
class PlotCallback(keras.callbacks.Callback):
    def __init__(self, dado_por_batch=False, titulo_grafico='Treinamento', eixo_x_grafico='Época',
                 eixo_y_grafico='Erro', label_precisao_grafico='Precisão'):
        super().__init__()
        self.logs = []
        self.x_loss = []
        self.y_val_loss = []
        self.y_loss = []
        self.y_accuracy = []
        self.ite = 0
        self.dado_por_batch = dado_por_batch
        self.titulo_grafico = titulo_grafico
        self.eixo_x_grafico = eixo_x_grafico
        self.eixo_y_grafico = eixo_y_grafico
        self.label_precisao_grafico = label_precisao_grafico

    def on_train_begin(self, logs={}):
        self.logs = []
        self.x_loss = []
        self.y_val_loss = []
        self.y_loss = []
        self.ite = 0
        self.fig = plt.figure()

    def on_epoch_end(self, epoch, logs={}):
        if not self.dado_por_batch:
            self.ite += 1
            self.x_loss.append(self.ite)
            self.y_loss.append(logs.get('loss'))
            self.y_val_loss.append(logs.get('val_loss'))
            self.y_accuracy.append(logs.get('acc'))

    def on_batch_end(self, batch, logs=None):
        if self.dado_por_batch:
            self.ite += 1
            self.x_loss.append(self.ite)
            self.y_loss.append(logs.get('loss'))
            self.y_val_loss.append(logs.get('val_loss'))
            self.y_accuracy.append(logs.get('acc'))

    def on_train_end(self, logs={}):
        plt.subplot(2, 1, 1)
        plt.title(self.titulo_grafico)
        plt.grid()
        plt.plot(self.x_loss, self.y_loss, label="loss")
        plt.plot(self.x_loss, self.y_val_loss, label="val_loss")
        plt.legend()
        if self.dado_por_batch:
            plt.xlabel(self.eixo_x_grafico)
        else:
            plt.xlabel(self.eixo_x_grafico)
        plt.ylabel(self.eixo_y_grafico)
        plt.subplot(2, 1, 2)
        plt.plot(self.x_loss, self.y_accuracy, label=self.label_precisao_grafico)
        plt.grid()
        self.fig.show()
