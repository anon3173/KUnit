

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
        
        seed = 7
        #datapoints
        X = np.arange(0.0, 5.0, 0.1, dtype='float32').reshape(-1,1)
        y = 5 * np.power(X,2) + np.power(np.random.randn(50).reshape(-1,1),3)
        df_data = pandas.DataFrame(X)
        df_label = pandas.DataFrame(y)
        return df_data,df_label


    