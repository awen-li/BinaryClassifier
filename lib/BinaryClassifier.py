#!/usr/bin/python

import abc
import csv
import numpy as np

class BinaryClassifier(metaclass=abc.ABCMeta):
    def __init__(self, Features, Labels,   Iteration):
        self.W = None
        self.Features  = Features
        self.Labels    = Labels
        self.Iteration = Iteration

        self.Classes = set (self.Labels)

    def InitWV (self):
        self.W = np.zeros(self.Features[0].shape)
        self.W = np.squeeze(np.asarray(self.W, dtype=np.float64))
        self.W = np.ravel(self.W)
        
    def Predict (self, x):
        Prdt = np.sign(np.dot(x, self.W))
        if (Prdt == 0):
            Prdt = -1
        return Prdt
        
    @abc.abstractmethod    
    def UpdateWeight (self, x, y, Pred):
        print ("UpdateWeight...")

    def Fit (self):
        self.InitWV ()

        FeatureNum = self.Features.shape[0]
        for Itr in range (0, self.Iteration):
            for i in range(FeatureNum):
                x = np.squeeze(np.asarray(self.Features[i], dtype=np.float64))
                x = np.ravel(x)
                y = self.Labels[i]
           
                Pred = self.Predict (x)              
                if (Pred != y):
                    w = self.UpdateWeight (x, y, Pred)

                    
        



    
