
import pandas 
import re
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from scipy import stats
import numpy as np
from imblearn.over_sampling import SMOTE,RandomOverSampler
from sklearn.utils import resample

class DataPreprocess():
    
    def __init__(self):
        super(DataPreprocess, self).__init__()
        self.data_path = '../truck_data.csv' # path for training data 

        
    
    def load(self):

        dataset = pandas.read_csv(self.data_path)

        # Add preprocessing steps here
        # Convert categorical data to numeric
        dataset['Fuel_Type'].replace(['Petrol', 'Diesel','CNG'], [0, 1, 2], inplace=True)
        dataset['Transmission'].replace(['Manual', 'Automatic'], [0, 1], inplace=True)
        dataset['Seller_Type'].replace(['Dealer', 'Individual'], [0, 1], inplace=True)

        # Separate data and labels 
        y = dataset['Selling_Price']
        X  = dataset.drop(['Selling_Price'], axis = 1)

        # Standardize the data
        sc = StandardScaler() 
        X = sc.fit_transform(X)


        # Return data as df_data and label as df_label
        df_data = X
        df_label = y
        return df_data,df_label

def main():
        dp = DataPreprocess()
        dp.load()
   
   
if __name__ == "__main__":
    main()