import keras
from matplotlib import pyplot as plt


# Todo: Implementar a atualização da imagem em tempo real ....
class PlotCallback(keras.callbacks.Callback):
    def __init__(self, dado_por_batch=False, plot_tempo_real=False):  # , plot_por_batch=False):
        super().__init__()
        self.logs = []
        self.x_loss = []
        self.y_val_loss = []
        self.y_loss = []
        self.ite = 0
        self.dado_por_batch = dado_por_batch
        self.plot_tempo_real = plot_tempo_real

    def on_train_begin(self, logs={}):
        self.logs = []
        self.x_loss = []
        self.y_val_loss = []
        self.y_loss = []
        self.ite = 0
        self.fig = plt.figure()
        if self.plot_tempo_real:
            plt.ion()
            plt.plot(self.x_loss, self.y_loss, label="loss")
            plt.plot(self.x_loss, self.y_val_loss, label="val_loss")
            plt.legend()
            self.fig.show()
        else:
            plt.ioff()

    def on_epoch_end(self, epoch, logs={}):

        self.ite += 1
        self.logs.append(logs)
        self.x_loss.append(self.ite)
        self.y_loss.append(logs.get('loss'))
        self.y_val_loss.append(logs.get('val_loss'))

        if self.plot_tempo_real:
            plt.draw()

    def on_train_end(self, logs={}):
        if not self.plot_tempo_real:
            plt.plot(self.x_loss, self.y_loss, label="loss")
            plt.plot(self.x_loss, self.y_val_loss, label="val_loss")
            plt.legend()
            self.fig.show()
