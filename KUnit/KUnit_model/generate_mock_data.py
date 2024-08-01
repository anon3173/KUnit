
from sklearn.datasets import make_classification, make_regression
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from interface_test_phase_2 import InterfacePhase2
from specification import Spec
from design_model import MyModel
import numpy as np
import warnings 


class GenerateMockData():
    def __init__(self):
        super(GenerateMockData, self).__init__()
        self.spec = Spec()
        self.problem_type,self.classification_type,self.number_of_classes = self.spec.specification()
        self.features = self.spec.features()

    def MockData(self):
        if self.problem_type == 1:
            n_features = self.features 
            X1, y = make_regression(
                                n_samples=n_features*10, 
                                n_features= n_features, 
                                n_informative = n_features, 
                                random_state=42
                            )
            scaler = StandardScaler()
            train_x1 = scaler.fit_transform(X1)
                
        if self.problem_type == 2:
            n_classes = self.number_of_classes
            n_features = self.features 
            if n_features < 2:
                n_clusters = 1
            else: 
                n_clusters = 2
            
            X1, y = make_classification(
                              
                                n_samples=n_classes*100, 
                                n_features= n_features, 
                                n_informative = n_features, 
                                n_classes=n_classes, 
                                n_redundant=0,
                                n_clusters_per_class=n_clusters,
                                random_state=42 )
    
            scaler = StandardScaler()
            train_x1 = scaler.fit_transform(X1)
           
        return train_x1,y,self.problem_type


   