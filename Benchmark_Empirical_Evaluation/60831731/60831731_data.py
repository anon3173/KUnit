

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
       
    def abc(self,x1, x2):
        b = -2 * x1 * x2
        c = x1 * x2
        sol = [b, c]
        return sol

    def Nx2(self,N, M):
        matrix = []
        n = N + 1
        m = M + 1
        for i in range(1, n):
            for j in range(1, m):
                temp = [i, j]
                matrix.append(temp)
        final_matrix = np.array(matrix)
        return final_matrix


    

    # print(output)

    


    def load(self):
        
        # dataset = pandas.read_csv(self.data_path,header=None)
        # dataset = pandas.read_csv(self.data_path,header=None)
        # dataset= dataset.values
        a = 10
        b = 10
        c = a * b
        output = self.Nx2(a, b)
        input = []
        for i in range(0, c):
            temp2 = self.abc(output[i, 0], output[i, 1])
            input.append(temp2)
        input = np.array(input)

        print(input)
        
        train_labels = output
        train_samples = input

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_train_samples = scaler.fit_transform(train_samples.reshape(-1, 1))
        df_data = scaled_train_samples.reshape(-1, 2)

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_train_labels = scaler.fit_transform(train_labels.reshape(-1, 1))
        df_label = scaled_train_labels.reshape(-1, 2)  
        return df_data,df_label


    