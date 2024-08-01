

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
        #self.data_path = '' # path for training data 
    
    def load(self):
        X_train, y_train = make_classification(1000, 28 * 28)
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        df_data = pandas.DataFrame(X_train)
        df_label = pandas.DataFrame(y_train)
        return df_data,df_label
