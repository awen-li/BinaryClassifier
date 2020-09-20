#!/usr/bin/python

import numpy as np
from lib.BinaryClassifier import BinaryClassifier
from lib.McBinaryClassifier import McBinaryClassifier


#standard Perceptron
class Perceptron(BinaryClassifier):
    def __init__(self, Features, Labels,   Iteration):
        super(Perceptron, self).__init__(Features, Labels, Iteration)
        self.Labels = self.BinaryClass (Labels)
        self.Name = "Perceptron"
   
    def UpdateWeight (self, x, y, Pred):
        self.W = self.W + np.dot(y, x)

#Average Perceptron
class AveragePerceptron(BinaryClassifier):
    def __init__(self, Features, Labels,   Iteration):
        super(AveragePerceptron, self).__init__(Features, Labels, Iteration)
        self.Labels = self.BinaryClass (Labels)
        self.Name = "Average Perceptron"

        self.SumW = self.InitWV ()
   
    def UpdateWeight (self, x, y, Pred):
        self.W = self.W + np.dot(y, x)
        self.SumW = self.SumW + self.W

    def Fit (self):
        super(AveragePerceptron, self).Fit()
        self.W = self.SumW / self.Features.shape[0]
 

#Multi-Class Perceptron
class McPerceptron(McBinaryClassifier):
    def __init__(self, Features, Labels,   Iteration):
        super(McPerceptron, self).__init__(Features, Labels, Iteration)
        self.Name = "Multi-class Perceptron"
   
    def UpdateWeight (self, x, y, Pred):
        self.W = np.add(self.W, np.subtract(self.Fxy[y], self.Fxy[Pred]))


#Multi-Class Average Perceptron
class McAveragePerceptron(McBinaryClassifier):
    def __init__(self, Features, Labels,   Iteration):
        super(McAveragePerceptron, self).__init__(Features, Labels, Iteration)
        self.Name = "Multi-class Average Perceptron"

        self.SumW = self.InitWV ()
   
    def UpdateWeight (self, x, y, Pred):
        self.W = np.add(self.W, np.subtract(self.Fxy[y], self.Fxy[Pred]))
        self.SumW = self.SumW + self.W

    def Fit (self):
        super(McAveragePerceptron, self).Fit()
        self.W = self.SumW / self.Features.shape[0]
    