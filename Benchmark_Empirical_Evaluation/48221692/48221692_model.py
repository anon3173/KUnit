

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
        neural_network = Sequential() # create model
        neural_network.add(Dense(5, input_dim=1)) # hidden layer
        neural_network.add(Activation('sigmoid'))
        neural_network.add(Dense(1)) # output layer
        neural_network.add(Activation('sigmoid'))
        neural_network.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])
        return neural_network
       
