

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
        # self.data_path = '' # path for training data 
    
    def load(self):
        dataset = np.loadtxt("../pima-indians-diabetes.data", delimiter=",")
        # split into input (X) and output (Y) variables
        X = dataset[:,0:8]
        Y = dataset[:,8]
        df_data = pandas.DataFrame(X)
        df_label = pandas.DataFrame(Y)
        return df_data,df_label
