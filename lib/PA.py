#!/usr/bin/python

import numpy as np
from lib.BinaryClassifier import BinaryClassifier
from lib.McBinaryClassifier import McBinaryClassifier

#standard Passive Aggressive Algorithm
class PA(BinaryClassifier):
    def __init__(self, Features, Labels,   Iteration):
        super(PA, self).__init__(Features, Labels, Iteration)

    def CalLearnRate(self, x, y):
        return (1 - y * np.dot(self.W, x)) / (np.square(np.linalg.norm(x)))
   
    def UpdateWeight (self, x, y, Pred):
        LearnRate = self.CalLearnRate(x, y)
        self.W = self.W + LearnRate * np.dot(y, x)
 
#Multi-Class Passive Aggressive Algorithm
class McPA(McBinaryClassifier):
    def __init__(self, Features, Labels,   Iteration):
        super(McPA, self).__init__(Features, Labels, Iteration)

    def CalLearnRate(self, y, Pred):
        Factor = 1 - (np.dot(self.W, np.array(self.Fxy[y])) - np.dot(self.W, np.array(self.Fxy[Pred])))
        return Factor / (np.square(np.linalg.norm(self.Fxy[y] - self.Fxy[Pred])))
   
    def UpdateWeight (self, x, y, Pred):
        LearnRate = self.CalLearnRate(y, Pred)
        self.W = np.add(self.W, LearnRate * np.subtract(self.Fxy[y], self.Fxy[Pred]))
    