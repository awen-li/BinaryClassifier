#!/usr/bin/python

import sys, getopt
import pandas as pd
from lib.Perceptron import Perceptron, AveragePerceptron, McPerceptron, McAveragePerceptron
from lib.PA import PA, McPA

def LoadData (FileName):
    RawData = pd.read_csv("data/" + FileName, header=0)
    Data = RawData.values

    Features = Data[0::, 1::]
    Labels   = Data[::, 0]

    return Features, Labels

def Train (Clf, Tag):
    Clf.Fit ()
    Clf.Plot(Tag)

def GeneralLearning (Clf, Tag):
    Clf.GeneralLearning (100, 100, 20)
    Clf.PlotGeneralCurve(Tag)
    
def RunQuestion_5_1a(Features, Labels, TestFeatures, TestLabels):
    print ("\r\nQuestion5.1a: Compute learning curve for both Perceptron and PA")    
    Clf = Perceptron (Features, Labels, TestFeatures, TestLabels, 50)
    Train (Clf, "5_1a")
    Clf = PA (Features, Labels, TestFeatures, TestLabels, 50)
    Train (Clf, "5_1a")

def RunQuestion_5_1b(Features, Labels, TestFeatures, TestLabels):
    print ("\r\nQuestion5.1b: Compute accuracy of both Perceptron and PA")   
    Clf = Perceptron (Features, Labels, TestFeatures, TestLabels, 20)
    Train (Clf, "5_1b")   
    Clf = PA (Features, Labels, TestFeatures, TestLabels, 20)
    Train (Clf, "5_1b")

def RunQuestion_5_1c(Features, Labels, TestFeatures, TestLabels):
    print ("\r\nQuestion5.1c: Compute accuracy of Average Perceptron")
    Clf = AveragePerceptron (Features, Labels, TestFeatures, TestLabels, 20);
    Train (Clf, "5_1c")

def RunQuestion_5_1d(Features, Labels, TestFeatures, TestLabels):
    print ("\r\nQuestion5.1d: Compute the general learning curve")
    Clf = Perceptron (Features, Labels, TestFeatures, TestLabels, 20)
    GeneralLearning (Clf, "5_1d")
    Clf = PA (Features, Labels, TestFeatures, TestLabels, 20)
    GeneralLearning (Clf, "5_1d")
    Clf = AveragePerceptron (Features, Labels, TestFeatures, TestLabels, 20);
    GeneralLearning (Clf, "5_1d")

def RunQuestion_5_2a(Features, Labels, TestFeatures, TestLabels):
    print ("\r\nQuestion5.2a: Compute learning curve for both Multi-class Perceptron and PA")    
    Clf = McPerceptron (Features, Labels, TestFeatures, TestLabels, 50)
    Train (Clf, "5_2a")
    Clf = McPA (Features, Labels, TestFeatures, TestLabels, 50)
    Train (Clf, "5_2a")

def RunQuestion_5_2b(Features, Labels, TestFeatures, TestLabels):
    print ("\r\nQuestion5.2b: Compute accuracy of both Multi-class Perceptron and PA")   
    Clf = Perceptron (Features, Labels, TestFeatures, TestLabels, 20)
    Train (Clf, "5_2b")   
    Clf = PA (Features, Labels, TestFeatures, TestLabels, 20)
    Train (Clf, "5_2b")

def RunQuestion_5_2c(Features, Labels, TestFeatures, TestLabels):
    print ("\r\nQuestion5.2c: Compute accuracy of Multi-class Average Perceptron")
    Clf = McAveragePerceptron (Features, Labels, TestFeatures, TestLabels, 20);
    Train (Clf, "5_2c")

def RunQuestion_5_2d(Features, Labels, TestFeatures, TestLabels):
    print ("\r\nQuestion5.2d: Compute the general learning curve of Multi-class algorighms")
    Clf = McPerceptron (Features, Labels, TestFeatures, TestLabels, 20)
    GeneralLearning (Clf, "5_2d")
    Clf = McPA (Features, Labels, TestFeatures, TestLabels, 20)
    GeneralLearning (Clf, "5_2d")
    Clf = McAveragePerceptron (Features, Labels, TestFeatures, TestLabels, 20);
    GeneralLearning (Clf, "5_2d")

def main(argv):
    print ("Loading training data...")
    Features, Labels = LoadData ("fashion-mnist_train.csv")
    print ("Loading testing data...")
    TestFeatures, TestLabels = LoadData ("fashion-mnist_test.csv")

    RunQuestion_5_1a(Features, Labels, TestFeatures, TestLabels)
    RunQuestion_5_1b(Features, Labels, TestFeatures, TestLabels)
    RunQuestion_5_1c(Features, Labels, TestFeatures, TestLabels)
    RunQuestion_5_1d(Features, Labels, TestFeatures, TestLabels)

    RunQuestion_5_2a(Features, Labels, TestFeatures, TestLabels)
    RunQuestion_5_2b(Features, Labels, TestFeatures, TestLabels)
    RunQuestion_5_2c(Features, Labels, TestFeatures, TestLabels)
    RunQuestion_5_2d(Features, Labels, TestFeatures, TestLabels)
   

if __name__ == "__main__":
   main(sys.argv[1:])
