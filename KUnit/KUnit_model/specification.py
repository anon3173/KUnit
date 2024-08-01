class Spec():
    def __init__(self):
        super(Spec, self).__init__()

    def specification(self):

        problem_type = int(input('Enter type of problem: 1. Regression 2. Classification \n'))
        classification_type = 0
        number_of_classes  = 0
        if problem_type == 2:
            classification_type = int(input('Enter type of classification 1. Binary 2. Multiclass \n' ))
            if classification_type == 1:
                number_of_classes = 2
            elif classification_type == 2:
                number_of_classes = int(input('Enter number of classes \n'))
        return problem_type, classification_type, number_of_classes  

    def features(self):
        feature_number = int(input('Enter number of features \n'))   
        return feature_number