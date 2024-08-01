

from sklearn import preprocessing
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
   
        number_of_samples = 100
        MAX_CONTEXTS = 430
        X, y = make_regression(number_of_samples, 3)
        scaler = MinMaxScaler(feature_range=(-1.9236537371711413, 1.9242677998256246))
        X = scaler.fit_transform(X, y)
        df_data = X
        df_label = y
        return df_data,df_label
