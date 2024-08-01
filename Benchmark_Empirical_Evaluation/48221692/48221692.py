
import numpy
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Activation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import keras
import time

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)


# Import the dataset
dataset = pd.read_csv("../Linear_Data.csv", header=None)
df = dataset.values
data = df[:,0:1]
label =  df[:,1]

X_train, X_test, Y_train, Y_test = train_test_split(data, label, 
                                                    test_size=0.25)

# Now we build the model
neural_network = Sequential() # create model
neural_network.add(Dense(5, input_dim=1)) # hidden layer
neural_network.add(Activation('sigmoid'))
neural_network.add(Dense(1)) # output layer
neural_network.add(Activation('sigmoid'))
neural_network.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])
neural_network_fitted = neural_network.fit(X_train, Y_train, epochs=100, verbose=1, 
                                           batch_size=X_train.shape[0], initial_epoch=0)
