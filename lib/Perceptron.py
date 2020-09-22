#!/usr/bin/python

import numpy as np
from lib.BinaryClassifier import BinaryClassifier
from lib.McBinaryClassifier import McBinaryClassifier


#standard Perceptron
class Perceptron(BinaryClassifier):
    def __init__(self, Features, Labels, TestFeatures, TestLabels, Iteration):
        super(Perceptron, self).__init__(Features, Labels, TestFeatures, TestLabels, Iteration)
        self.Name = "Perceptron"
   
    def UpdateWeight (self, x, y, Pred):
        self.W = self.W + np.dot(y, x)

#Average Perceptron
class AveragePerceptron(BinaryClassifier):
    def __init__(self, Features, Labels, TestFeatures, TestLabels, Iteration):
        super(AveragePerceptron, self).__init__(Features, Labels, TestFeatures, TestLabels, Iteration)
        self.Name = "Average Perceptron"

        self.SumW = self.InitWV ()
   
    def UpdateWeight (self, x, y, Pred):
        self.W = self.W + np.dot(y, x)
        self.SumW = self.SumW + self.W

    def Test (self, Itr):
        WBack  = self.W
        self.W = self.SumW / self.Features.shape[0]
        super(AveragePerceptron, self).Test(Itr)
        if (Itr != self.Iteration):
            self.W = WBack
 

#Multi-Class Perceptron
class McPerceptron(McBinaryClassifier):
    def __init__(self, Features, Labels, TestFeatures, TestLabels, Iteration):
        super(McPerceptron, self).__init__(Features, Labels, TestFeatures, TestLabels, Iteration)
        self.Name = "Multi-class Perceptron"
   
    def UpdateWeight (self, x, y, Pred):
        self.W = np.add(self.W, np.subtract(self.Fxy[y], self.Fxy[Pred]))


#Multi-Class Average Perceptron
class McAveragePerceptron(McBinaryClassifier):
    def __init__(self, Features, Labels, TestFeatures, TestLabels, Iteration):
        super(McAveragePerceptron, self).__init__(Features, Labels, TestFeatures, TestLabels, Iteration)
        self.Name = "Multi-class Average Perceptron"

        self.SumW = self.InitWV ()
   
    def UpdateWeight (self, x, y, Pred):
        self.W = np.add(self.W, np.subtract(self.Fxy[y], self.Fxy[Pred]))
        self.SumW = self.SumW + self.W

    def Test (self, Itr):
        WBack  = self.W
        self.W = self.SumW / self.Features.shape[0]
        super(McAveragePerceptron, self).Test(Itr)
        if (Itr != self.Iteration):
            self.W = WBack
    