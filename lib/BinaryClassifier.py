#!/usr/bin/python

import abc
import csv
import numpy as np
import matplotlib.pyplot as plot

class BinaryClassifier(metaclass=abc.ABCMeta):
    def __init__(self, Features, Labels, Iteration):
        
        self.Features  = Features
        self.Labels    = Labels
        self.Iteration = Iteration

        self.Classes = set (self.Labels)
        self.W = None
        self.Mistakes = {}
        self.Accuracy = {}

        self.Name = "BinaryClassifier"

    def InitWV (self):
        W = np.zeros(self.Features[0].shape)
        W = np.squeeze(np.asarray(W, dtype=np.float64))
        W = np.ravel(W)
        return W
        
    def Predict (self, x):
        Prdt = np.sign(np.dot(x, self.W))
        if (Prdt == 0):
            Prdt = -1
        return Prdt

    def BinaryClass(self, Labels):
        BinLabels = []
        for l in Labels:
            if l%2 == 0:
                BinLabels.append(1)
            else:
                BinLabels.append(-1)
        return BinLabels

    def PlotLearnIngCurve(self, Type):
        Itrs = list(self.Mistakes.keys ())
        Mistakes = list(self.Mistakes.values ())
        plot.plot(Itrs, Mistakes, color = 'g', marker='o', linestyle='solid')
        plot.xlabel('Number of iterations')
        plot.ylabel('Number of mistakes')
        plot.title("Learning curve of %s" %(self.Name))
        plot.savefig(self.Name + "-" + Type)
        plot.show()
        plot.close()

    def PlotAccuracyCurve(self, Type):
        Itrs = list(self.Accuracy.keys ())
        Accuracy = list(self.Accuracy.values ())
        plot.plot(Itrs, Accuracy, color = 'r', marker='o', linestyle='solid')
        plot.xlabel('Number of iterations')
        plot.ylabel('Accuracy (%)')
        plot.title("Accuracy curve of %s" %(self.Name))
        plot.savefig(self.Name + "-" + Type)
        plot.show()
        plot.close()
        
    @abc.abstractmethod    
    def UpdateWeight (self, x, y, Pred):
        print ("UpdateWeight...")

    def Fit (self):
        self.W = self.InitWV ()

        ExampleNum = self.Features.shape[0]
        for Itr in range (1, self.Iteration+1):
            Mist = 0
            for i in range(ExampleNum):
                x = np.squeeze(np.asarray(self.Features[i], dtype=np.float64))
                x = np.ravel(x)
                y = self.Labels[i]
           
                Pred = self.Predict (x)              
                if (Pred != y):
                    Mist = Mist + 1
                    w = self.UpdateWeight (x, y, Pred)
            
            self.Mistakes[Itr] = Mist
            self.Accuracy[Itr] = (1 - Mist/ExampleNum)*100
            #print ("Mistake: %d, Accuracy:%0.2f" %(self.Mistakes[Itr], self.Accuracy[Itr]))
        return

    def Test (self, Features, Labels):
        
        self.Mistakes = {}
        self.Accuracy = {}

        Labels = self.BinaryClass (Labels)
        
        ExampleNum = Features.shape[0]
        for Itr in range (1, self.Iteration+1):
            Mist = 0
            for i in range(ExampleNum):
                x = np.squeeze(np.asarray(Features[i], dtype=np.float64))
                x = np.ravel(x)
                y = Labels [i]
                
                Pred = self.Predict (x)        
                if Pred != y:
                    Mist = Mist + 1
                    
            self.Mistakes[Itr] = Mist
            self.Accuracy[Itr] = (1 - Mist/ExampleNum)*100
        return

                    
        



    
