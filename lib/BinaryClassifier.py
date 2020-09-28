#!/usr/bin/python

import abc
import numpy as np
import matplotlib.pyplot as plot

class BinaryClassifier(metaclass=abc.ABCMeta):
    def __init__(self, Features, Labels, TestFeatures, TestLabels, Iteration):
        
        self.Features     = Features
        self.Labels       = self.BinaryClass (Labels)      
        self.TestFeatures = TestFeatures
        self.TestLabels   = self.BinaryClass (TestLabels)      
        self.Iteration    = Iteration

        self.W        = None
        self.Mistakes = {}
        self.Accuracy = {}
        self.ExampleNums = {}
        
        self.TestMistakes = {}
        self.TestAccuracy = {}
        self.TestExampleNums = {}

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

    def PlotLearnIngCurve(self, Mistakes, Type):
        Itrs = list(Mistakes.keys ())
        Mistakes = list(Mistakes.values ())
        if (len(Itrs) == 0):
            return
        plot.plot(Itrs, Mistakes, color = 'g', marker='o', linestyle='solid')
        plot.xlabel('Number of iterations')
        plot.ylabel('Number of mistakes')
        plot.title("Learning curve of %s" %(self.Name))
        plot.savefig("result/" + self.Name + "-" + Type)
        #plot.show()
        plot.close()

    def PlotAccuracyCurve(self, Accuracy, Type):
        Itrs = list(Accuracy.keys ())
        Accuracy = list(Accuracy.values ())
        if (len(Itrs) == 0):
            return
        plot.plot(Itrs, Accuracy, color = 'r', marker='o', linestyle='solid')
        plot.xlabel('Number of iterations')
        plot.ylabel('Accuracy (%)')
        plot.title("Accuracy curve of %s" %(self.Name))
        plot.savefig("result/" + self.Name + "-" + Type)
        #plot.show()
        plot.close()

    def Plot (self, Type):
        self.PlotLearnIngCurve (self.Mistakes, "Train_LearningCurve-" + Type)
        self.PlotAccuracyCurve (self.Accuracy, "Train_AccuracyCurve-" + Type)
        self.PlotLearnIngCurve (self.TestMistakes, "Test_LearningCurve-" + Type)
        self.PlotAccuracyCurve (self.TestAccuracy, "Test_AccuracyCurve-" + Type)


    def PlotGeneralCurve(self, Type):
        Itrs = list(self.ExampleNums.values ())
        Accuracy = list(self.TestAccuracy.values ())
        plot.plot(Itrs, Accuracy, color = 'r', marker='o', linestyle='solid')
        plot.xlabel('Number of training examples')
        plot.ylabel('Testing Accuracy (%)')
        plot.title("Accuracy curve of %s" %(self.Name))
        plot.savefig("result/" + self.Name + "-generallearning-" + Type)
        #plot.show()
        plot.close()
        
        
    @abc.abstractmethod    
    def UpdateWeight (self, x, y, Pred):
        print ("UpdateWeight...")

    def Test (self, Itr):
        ExampleNum = self.TestFeatures.shape[0]
        Mist = 0
        for i in range(ExampleNum):
            x = np.squeeze(np.asarray(self.TestFeatures[i], dtype=np.float64))
            x = np.ravel(x)
            y = self.TestLabels [i]
                    
            Pred = self.Predict (x)        
            if Pred != y:
                Mist = Mist + 1
                        
        self.TestMistakes[Itr] = Mist
        self.TestAccuracy[Itr] = (1 - Mist/ExampleNum)*100
        return

    def Train (self, ExampleNum, Itr):
        Mist = 0
        for i in range(ExampleNum):
            x = np.squeeze(np.asarray(self.Features[i], dtype=np.float64))
            x = np.ravel(x)
            y = self.Labels[i]

            Pred = self.Predict (x)
            if (Pred != y):
                Mist = Mist + 1
                self.UpdateWeight (x, y, Pred)
            
        self.Mistakes[Itr] = Mist
        self.Accuracy[Itr] = (1 - Mist/ExampleNum)*100

    def Fit (self, IsTest=True):
        self.W = self.InitWV ()
        ExampleNum = self.Features.shape[0]
        
        for Itr in range (1, self.Iteration+1):
            print ("\r%s iteration: %d%-64s" %(self.Name, Itr, ""), end = "")
            self.Train (ExampleNum, Itr)
            if (IsTest == True):
                self.Test (Itr)
        print ("\r\n")
        return

    def GeneralLearning (self, Start, StepSize, Iteration):
        TotalExampleNum = self.Features.shape[0]

        Itr = 0
        StepNum = StepSize * (Iteration+1)
        for ExampleNum in range (Start, StepNum, StepSize):
            self.W = self.InitWV ()
            print ("\r%s ExampleNum: %d%-64s" %(self.Name, ExampleNum, ""), end = "")
            if (ExampleNum > TotalExampleNum):
                break
            
            for I in range (1, self.Iteration+1):
                self.Train (ExampleNum, Itr)
                self.Test (Itr)
            self.ExampleNums [Itr] = ExampleNum
            
            Itr = Itr + 1
        print ("\r\n")
        return
                    
        



    
