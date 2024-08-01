
# Multiclass Classification with the Iris Flowers Dataset 
import numpy as np
import pandas as pd
from keras.models import Sequential 
from keras.layers import Dense, Activation, Dropout
from keras.wrappers.scikit_learn import KerasClassifier 
from keras.utils import np_utils 
from sklearn.model_selection import cross_val_score 
from sklearn.model_selection import KFold 
from sklearn.preprocessing import LabelEncoder 
from sklearn.pipeline import Pipeline
from keras.optimizers import SGD,Adam
import time
import sys
import keras

# seed weights
np.random.seed(3)

dataframe = pd.read_csv('../dataset.csv', delimiter=',')
data = dataframe.values 
X_train = data[:,0:16].astype(float) 
Y = data[:,16]

# encode class values as integers 
encoder = LabelEncoder() 
encoder.fit(Y) 
y_train = encoder.transform(Y) 

# convert integers to dummy variables (i.e. one hot encoded) 
y_train = np_utils.to_categorical(y_train) 

print(X_train)

model = Sequential()
model.add(Dense(64, input_dim=16, kernel_initializer='uniform'))
model.add(Activation('tanh'))
model.add(Dropout(0.5))
model.add(Dense(64, kernel_initializer='uniform'))
model.add(Activation('tanh'))
model.add(Dropout(0.5))
model.add(Dense(2, kernel_initializer='uniform'))
model.add(Activation('softmax'))
# model.add(Dense(1, kernel_initializer='uniform'))
# model.add(Activation('sigmoid'))

sgd = SGD(learning_rate=0.1)

model.compile(loss='mean_squared_error', optimizer=sgd,metrics=[ 'accuracy' ])
# model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=[ 'accuracy' ])
model.fit(X_train, y_train, nb_epoch=20, batch_size=16)

