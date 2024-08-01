

from tensorflow.keras.optimizers import SGD, Adam, RMSprop
import tensorflow 
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense , Activation, Dropout,Conv1D, MaxPooling1D, Flatten, InputLayer, BatchNormalization
class MyModel(tensorflow.keras.layers.Layer):
     
    def __init__(self):
        super(MyModel, self).__init__()
        
    def call(self):
        model = Sequential([
            Dense(2, input_shape=(2,), activation='sigmoid'),
            Dense(2, activation='sigmoid'),
        ])

        print(model.weights)
        opt= SGD(learning_rate=0.01)
        model.compile(optimizer=opt, loss='mean_squared_error', metrics=['accuracy'])
        return model
       
    
