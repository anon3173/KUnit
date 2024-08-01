

from tensorflow.keras.optimizers import SGD, Adam, RMSprop
import tensorflow 
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense , Activation, Dropout,Conv1D, MaxPooling1D, Flatten, InputLayer, BatchNormalization
class MyModel(tensorflow.keras.layers.Layer):
     
    def __init__(self):
        super(MyModel, self).__init__()
        
    def call(self):
        #Design your model here and return the model
        model = Sequential()
        model.add(Dense(21, input_dim=14))
        model.add(Activation('relu'))
        model.add(Dense(2))
        model.add(Activation('softmax'))
        #compile
        model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["accuracy"])
        return model
       
    

    