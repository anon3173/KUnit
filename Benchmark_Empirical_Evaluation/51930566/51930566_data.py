

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
        # self.data_path = '../iris.data' # path for training data 
       
       
    def load(self):
        
        # dataset = pandas.read_csv(self.data_path,header=None)
       
        iris = load_iris()
        X = iris['data']
        y = iris['target']
        names = iris['target_names']
        feature_names = iris['feature_names']

        # One hot encoding
        enc = OneHotEncoder()
        Y = enc.fit_transform(y[:, np.newaxis]).toarray()

        # Scale data to have mean 0 and variance 1 
        # which is importance for convergence of the neural network
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        df_data = X_scaled
        df_label = Y
        
        
        return df_data,df_label

