

from sklearn import preprocessing
from tensorflow.keras.utils import to_categorical
import pandas
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder, StandardScaler
import tensorflow 
import numpy as np

class DataPreprocess():
    
    def __init__(self):
        super(DataPreprocess, self).__init__()
        self.data_path = '..train.csv' # path for df_dataing data 
      
    def load(self):
        
        df_data = pandas.read_csv(self.data_path) 
        df_label = df_data.Survived
       
        en = preprocessing.LabelEncoder()
        df_label = en.fit_transform(df_label)
       
        df_data.drop('Survived', axis=1, inplace=True)

        columns = df_data.columns
        

        df_data['Age'] = df_data['Age'].fillna(df_data['Age'].mean())
        df_data['Fare'] = df_data['Fare'].fillna(df_data['Fare'].mean())
        

        category_index = len(columns)
        for i in range(category_index):
            print (str(i)+" : "+columns[i])
            df_data[columns[i]] = df_data[columns[i]].fillna('missing')
            

        df_data = np.array(df_data)
        

        ### label encode the categorical variables ###
        for i in range(category_index):
            print (str(i)+" : "+str(columns[i]))
            lbl = preprocessing.LabelEncoder()
            
            df_data[:,i] = lbl.transform(df_data[:,i])
          

        ### making data as numpy float ###
        df_data = df_data.astype(np.float32)
        
        

        return df_data,df_label

