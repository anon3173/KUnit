
import numpy 
import pandas 
from keras.models import Sequential 
from keras.layers import Dense, Activation
from keras.wrappers.scikit_learn import KerasClassifier 
from keras.utils import np_utils 
from sklearn.model_selection import cross_val_score 
from sklearn.model_selection import KFold 
from sklearn.preprocessing import LabelEncoder 
from sklearn.pipeline import Pipeline 
from sklearn.model_selection import train_test_split
import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
import keras
import sys
# fix random seed for reproducibility 
seed = 7 
numpy.random.seed(seed) 
# load dataset 
dataframe = pandas.read_csv("../iris.data", header=None) 
dataset = dataframe.values 
X = dataset[:,0:4].astype(float) 
Y = dataset[:,4] 

# encode class values as integers 
encoder = LabelEncoder() 
encoder.fit(Y) 
encoded_Y = encoder.transform(Y) 

# convert integers to dummy variables (i.e. one hot encoded) 
dummy_y = np_utils.to_categorical(encoded_Y) 
batch_size = 5
print(dummy_y)
X_train, X_val, y_train, y_val = train_test_split(X, dummy_y, test_size=0.2, shuffle=True)
# define baseline model 
# create model
model = Sequential()
model.add(Dense(4, input_dim=4, kernel_initializer="normal"))
model.add(Activation('relu'))
model.add(Dense(3, kernel_initializer="normal"))
model.add(Activation('sigmoid'))
# model.add(Activation('softmax'))

# Compile model
model.compile(loss= 'categorical_crossentropy' , optimizer= 'adam' , metrics=[ 'accuracy' ])


# model.fit( X, dummy_y, nb_epoch=200, batch_size=5, verbose=1,) 
model.fit( X_train, y_train, nb_epoch=200, batch_size=5, verbose=1,) 

# Assuming you have a trained model named 'model' and input data 'X_test'
predictions = model.predict(X_val)

# 'predictions' will contain the predicted outputs for the input data
print(numpy.argmax(predictions,axis=1))
print('These are true',numpy.argmax(y_val,axis=1))
# print("Classification Report:")
# print(classification_report(y_val, predictions))