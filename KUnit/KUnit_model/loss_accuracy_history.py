from tensorflow import keras
from design_model import MyModel

class LossAccuracyHistory(keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.model = MyModel()
        self.my_model = self.model.call()
        self.metric = self.my_model.compiled_metrics._metrics
        
    def on_train_begin(self, logs={}):
        self.losses = []
        self.met = []
       
    def on_epoch_end(self, epoch, logs={}):
        
        self.losses.append(logs.get('loss'))
        self.met.append(logs.get(self.metric[0]))
        return self.losses, self.met
    
    