

from sklearn import preprocessing
from tensorflow.keras.utils import to_categorical
from keras.utils import np_utils 
import pandas
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder, StandardScaler
import tensorflow 
import numpy as np
import seaborn as sns
from sklearn.datasets import make_regression, make_classification


class DataPreprocess():
    
    def __init__(self):
        super(DataPreprocess, self).__init__()
        # self.data_path = '' # path for training data 
   
    def load(self):
        
        # dataset = pandas.read_csv(self.data_path,header=None)
       
        X, y = make_classification(n_samples=1000, n_features=5, n_classes=5, n_informative=4, n_redundant=0, random_state=42)
        sc = StandardScaler()
        X = sc.fit_transform(X)
        df_data = X
        df_label = y
        return df_data,df_label

