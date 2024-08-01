

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
    
        X, y = make_classification(50000, 1000 * 1, n_classes=3, n_informative=990, random_state=42)
        
        y = to_categorical(y)
        df_data = X
        df_label = y
        return df_data,df_label


    