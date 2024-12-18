
from keras.layers import Dense, Activation
from keras.models import Sequential
import keras.utils
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from keras.utils.np_utils import to_categorical
import numpy as np
import pandas as pd
import keras
import time
import sys


df = pd.read_csv("../adult.data",header=None)
X = df.iloc[:,0:14]
Y = df.iloc[:,14]

encoder = LabelEncoder()
#X
for i in [1,3,5,6,7,8,9,13]:
   column = X[i]
   encoder.fit(column)
   encoded_C = encoder.transform(column)
   X[i] = keras.utils.to_categorical(encoded_C)

#Y
encoder.fit(Y)
en_Y = encoder.transform(Y)
Y = keras.utils.to_categorical(en_Y)


#model
model = Sequential()
model.add(Dense(21, input_dim=14))
model.add(Activation('relu'))
model.add(Dense(2))
model.add(Activation('softmax'))
#compile
model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["accuracy"])

#train
model.fit(X,Y, epochs=50, batch_size=100)

