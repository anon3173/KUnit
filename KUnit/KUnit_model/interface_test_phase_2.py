from specification import Spec
class InterfacePhase2():
    def __init__(self):
        super(InterfacePhase2, self).__init__()
        self.spec = Spec()
        self.problem_type, self.classification_type, self.number_of_classes = self.spec.specification()

    def specphase2(self):
       
        return self.problem_type,self.classification_type,self.number_of_classes 

   