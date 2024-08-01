
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder
import time
from data_preprocess import DataPreprocess
from specification1 import Spec

class DataPhase2(DataPreprocess):
    
    def __init__(self):
        super(DataPhase2, self).__init__()
        
        self.data_phase1 = DataPreprocess()
        self.clean_data, self.label = self.data_phase1.load()
        self.spec = Spec()
        self.problem_type, self.classification_type, self.number_of_classes, self.model_type = self.spec.specification()
        
    def labelConsistency(self):
     
            
        return self.number_of_classes, self.problem_type, self.model_type

def main():
    dp = DataPhase2()
    start = time.time()
    dp.labelConsistency()
    end = time.time()
    time_taken = str(end-start)
    with open('../time_phase_1.txt', 'a') as f:
        f.write('\n Phase 2: '+ time_taken)

if __name__ == "__main__":
    main()


   