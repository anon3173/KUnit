

from sklearn import preprocessing
from sklearn.datasets import load_iris
from tensorflow.keras.utils import to_categorical
from keras.utils import np_utils 
import pandas
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder, StandardScaler
import tensorflow 
import numpy as np
import random
import seaborn as sns
from sklearn.datasets import make_regression, make_classification


class DataPreprocess():
    
    def __init__(self):
        super(DataPreprocess, self).__init__()
        # self.data_path = '' # path for training data 
   
    def load(self):
        
        # dataset = pandas.read_csv(self.data_path,header=None)
        
        N = 5000
        X = np.empty((N,))
        Y = np.empty((N,))

        for i in range(N):
            X[i] = random.uniform(-10, 10)
        X = np.sort(X).reshape(-1, 1)

        for i in range(N):
            Y[i] = np.sin(X[i])
        Y = Y.reshape(-1, 1)

        X_scaler = MinMaxScaler()
        Y_scaler = MinMaxScaler()
        X = X_scaler.fit_transform(X)
        Y = Y_scaler.fit_transform(Y)
      
        
        df_data = X
        df_label = Y
        return df_data,df_label

