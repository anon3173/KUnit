

from sklearn import preprocessing
from tensorflow.keras.utils import to_categorical
from keras.utils import np_utils 
import pandas
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder, StandardScaler
import tensorflow 
import numpy as np
from sklearn.datasets import make_regression

class DataPreprocess():
    
    def __init__(self):
        super(DataPreprocess, self).__init__()
        # self.data_path = '' # path for training data 
       
       
    def load(self):
        
        # df = pandas.read_csv(self.data_path,header=None)
        X = np.arange(0.0, 10.0, 0.05)
        Y = np.empty(shape=0, dtype=float)
        # 
        # Calculate Y-Values
        for x in X:
            Y = np.append(Y, float(0.05*(15.72807*x - 7.273893*x**2 + 1.4912*x**3 - 0.1384615*x**4 + 0.00474359*x**5)))
        df_data = X
        df_label = Y
        return df_data,df_label


    