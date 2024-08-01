
import keras
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
import tensorflow 
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense , Activation, Dropout,Conv1D, MaxPooling1D, Flatten, InputLayer, BatchNormalization
class MyModel(tensorflow.keras.layers.Layer):
     
    def __init__(self):
        super(MyModel, self).__init__()
        
    def call(self):
        nn_model = Sequential([Dense(4, activation='relu', input_dim=4),
                       BatchNormalization(),
                       Dropout(.3),
                       Dense(4, activation='relu'),
                       BatchNormalization(),
                       Dropout(.3),
                       Dense(1, activation='sigmoid')])

        nn_model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
        return nn_model
      
 
    