import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.optimizers import SGD, Adam
import pandas as pd

n_samples = 200
n_feats = 2

cls0 = np.random.uniform(low=0.2, high=0.4, size=(n_samples,n_feats))
cls1 = np.random.uniform(low=0.5, high=0.7, size=(n_samples,n_feats))
x_train = np.concatenate((cls0, cls1))
y_train = np.concatenate((np.zeros((n_samples,)), np.ones((n_samples,))))

# shuffle data because all negatives (i.e. class "0") are first
# and then all positives (i.e. class "1")
indices = np.arange(x_train.shape[0])
np.random.shuffle(indices)
x_train = x_train[indices]
y_train = y_train[indices]

label_enc = LabelEncoder()
label_enc.fit(y_train)
y_train = to_categorical(label_enc.transform(y_train), n_feats)

model = Sequential()
model.add(Dense(units=2, activation='sigmoid', input_shape=(n_feats,)))
model.add(Dense(units=n_feats, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=500, batch_size=5, verbose=True)
