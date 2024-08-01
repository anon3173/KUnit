

from sklearn import preprocessing
from tensorflow.keras.utils import to_categorical
from keras.utils import np_utils 
import pandas
import sys
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder, StandardScaler
import tensorflow 
import numpy as np
import random
import seaborn as sns
from sklearn.datasets import make_regression, make_classification


class DataPreprocess():
    
    def __init__(self):
        super(DataPreprocess, self).__init__()
        self.data_path = '..fixed.csv' # path for training data 
       
       
    def load(self):
        data = pandas.read_csv(self.data_path)
        X = data.drop(['status', 'name'], axis = 1)
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)
        Y = data['status']
        df_data = X
        df_label = Y
        return df_data,df_label

    