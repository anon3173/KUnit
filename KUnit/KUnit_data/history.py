from tensorflow import keras


class LossAccuracyHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.accuracy = []
        self.mae = []
    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.accuracy.append(logs.get('accuracy'))
        self.mae.append(logs.get('mean_absolute_error'))
        return self.losses, self.accuracy,self.mae