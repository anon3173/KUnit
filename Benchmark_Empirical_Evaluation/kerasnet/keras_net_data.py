

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

    def generate_dataset(self,values):
        dataset = []

        for _ in range(values):
            n1 = random.randint(0, 100)
            n2 = random.randint(0, 100)

            average = (n1 + n2) / 2

            dataset.append([n1, n2, average])

        return np.array(dataset)


    # scale
    def preprocess(self,data):
        return data/100


    # unscale
    def postprocess(self,data):
        return data * 100

    
  
    def load(self):
        training_data_length = 10000
        training_data = self.generate_dataset(training_data_length)
        split_data = np.split(training_data, [2, ], axis=1)
        training_input = self.preprocess(split_data[0])
        training_output = self.preprocess(split_data[1]).flatten()  
        df_data = training_input
        df_label = training_output
        return df_data,df_label
