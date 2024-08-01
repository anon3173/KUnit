

from sklearn import preprocessing
from tensorflow.keras.utils import to_categorical
import pandas
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder, StandardScaler
import tensorflow 
import numpy 

class DataPreprocess():
    
    def __init__(self):
        super(DataPreprocess, self).__init__()
        # self.data_path = '' # path for training data 
       
       
    def load(self):
        
        df_data = numpy.array([[1., 1.], [0., 0.], [1., 0.], [0., 1.], [1., 1.], [0., 0.]])
        df_label = numpy.array([[0.], [0.], [1.], [1.], [0.], [0.]])

        return df_data,df_label

