

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
from sklearn.datasets import make_regression, make_regression


class DataPreprocess():
    
    def __init__(self):
        super(DataPreprocess, self).__init__()
        # self.data_path = '' # path for training data 
       
       
    def load(self):
        
        X, y = make_regression(1000, 5, random_state=42)
        df_data = X
        df_label = y
        return df_data,df_label

    