

from sklearn import preprocessing
from tensorflow.keras.utils import to_categorical
from keras.utils import np_utils 
import pandas
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder, StandardScaler
import tensorflow 
import numpy 

class DataPreprocess():
    
    def __init__(self):
        super(DataPreprocess, self).__init__()
        self.data_path = '../adult.data' # path for training data 
       
       
    def load(self):
        
        df = pandas.read_csv(self.data_path,header=None)
        X = df.iloc[:,0:14]
        Y = df.iloc[:,14]

        encoder = LabelEncoder()
        #X
        for i in [1,3,5,6,7,8,9,13]:
            column = X[i]
            encoder.fit(column)
            encoded_C = encoder.transform(column)
            X[i] = to_categorical(encoded_C)

        print(X.shape)
        #Y
        encoder.fit(Y)
        en_Y = encoder.transform(Y)
        Y = to_categorical(en_Y)
        df_data = pandas.DataFrame(X)
        df_label = pandas.DataFrame(Y)
        return df_data,df_label

