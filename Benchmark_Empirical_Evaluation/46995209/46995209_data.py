

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
        
        # data = pandas.read_csv(self.data_path) 
        numpy.random.seed(7)

        
        T = 1000
        X = numpy.array(range(T))
        Y = numpy.sin(3.5 * numpy.pi * X / T) 
        df_data = pandas.DataFrame(X)
        df_label = pandas.DataFrame(Y)
        return df_data,df_label


    