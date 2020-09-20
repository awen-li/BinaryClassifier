#!/usr/bin/python

import numpy as np
from lib.BinaryClassifier import BinaryClassifier
from lib.McBinaryClassifier import McBinaryClassifier


#standard Perceptron
class Perceptron(BinaryClassifier):
    def __init__(self, Features, Labels,   Iteration):
        super(Perceptron, self).__init__(Features, Labels, Iteration)
   
    def UpdateWeight (self, x, y, Pred):
        self.W = self.W + np.dot(y, x)
 

#Multi-Class Perceptron
class McPerceptron(McBinaryClassifier):
    def __init__(self, Features, Labels,   Iteration):
        super(McPerceptron, self).__init__(Features, Labels, Iteration)
   
    def UpdateWeight (self, x, y, Pred):
        self.W = np.add(self.W, np.subtract(self.Fxy[y], self.Fxy[Pred]))
    