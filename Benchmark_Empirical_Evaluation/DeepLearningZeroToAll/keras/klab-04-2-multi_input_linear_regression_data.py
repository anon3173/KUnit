

from sklearn import preprocessing
from sklearn.datasets import load_iris
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
from sklearn.datasets import load_breast_cancer

class DataPreprocess():
    
    def __init__(self):
        super(DataPreprocess, self).__init__()
        # self.data_path = '/Users/ruchira/Desktop/FinalIRBupload/SO Posts/58055105/fixed.csv' # path for training data 
    
    def load(self):
        
        x_data = np.array([[73., 80., 75.],
          [93., 88., 93.],
          [89., 91., 90.],
          [96., 98., 100.],
          [73., 66., 70.]])
        y_data = np.array([[152.],
          [185.],
          [180.],
          [196.],
          [142.]])
        df_data = pandas.DataFrame(x_data)
        df_label = pandas.DataFrame(y_data)
        return df_data,df_label
