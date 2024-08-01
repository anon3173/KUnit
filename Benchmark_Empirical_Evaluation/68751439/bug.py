import keras
from sklearn.datasets import make_classification
from tensorflow.keras.utils import to_categorical

X_train, y_train = make_classification(n_samples=100, n_features= 14,n_informative = 10,n_classes=11, n_clusters_per_class=1,)
y_train = to_categorical(y_train)
# define the keras model
model1 = keras.Sequential()
model1.add(keras.layers.Dense(64, input_dim=14, activation='relu'))
model1.add(keras.layers.Dense(128, activation='relu'))
model1.add(keras.layers.Dense(64, activation='relu'))  
model1.add(keras.layers.Dense(1, activation='softmax'))

# compile the keras model
model1.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit the keras model on the dataset
performance1 = model1.fit(X_train, y_train, epochs=100, validation_split=0.2)
