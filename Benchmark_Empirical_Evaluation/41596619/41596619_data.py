

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
        # self.data_path = '' # path for training data 
       
       
    def load(self):
        
        df_data = numpy.array([[0.5, 1, 1], [0.9, 1, 2], [0.8, 0, 1], [0.3, 1, 1], [0.6, 1, 2], [0.4, 0, 1], [0.9, 1, 7], [0.5, 1, 4],
                 [0.1, 0, 1], [0.6, 1, 0], [1, 0, 0]])
        df_label = numpy.array([[1], [1], [1], [2], [2], [2], [3], [3], [3], [0], [0]])


        return df_data,df_label


    