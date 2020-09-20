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

def RunPerceptron(Features, Labels, TestFeatures, TestLabels):
    print("RunPerceptron")
    Clf = Perceptron (Features, Labels, 50);
    Clf.Fit ()
    Clf.PlotLearnIngCurve ("Train_LearnIngCurve")
    Clf.PlotAccuracyCurve ("Train_AccuracyCurve")

    Clf.Test (TestFeatures, TestLabels)
    Clf.PlotLearnIngCurve ("Test_LearnIngCurve")
    Clf.PlotAccuracyCurve ("Test_AccuracyCurve")

def RunMcPerceptron(Features, Labels, TestFeatures, TestLabels):
    print("McPerceptron")
    Clf = McPerceptron (Features, Labels, 50);
    Clf.Fit ()
    Clf.PlotLearnIngCurve ("Train_LearnIngCurve")
    Clf.PlotAccuracyCurve ("Train_AccuracyCurve")

def RunAveragePerceptron(Features, Labels, TestFeatures, TestLabels):
    print("AveragePerceptron")
    Clf = AveragePerceptron (Features, Labels, 50);
    Clf.Fit ()
    Clf.PlotLearnIngCurve ("Train_LearnIngCurve")
    Clf.PlotAccuracyCurve ("Train_AccuracyCurve")  

def RunMcAveragePerceptron(Features, Labels, TestFeatures, TestLabels):
    print("McAveragePerceptron")
    Clf = McAveragePerceptron (Features, Labels, 50);
    Clf.Fit ()
    Clf.PlotLearnIngCurve ("Train_LearnIngCurve")
    Clf.PlotAccuracyCurve ("Train_AccuracyCurve")


def RunPA(Features, Labels, TestFeatures, TestLabels):
    print("Passive-Aggressive")
    Clf = PA (Features, Labels, 50);
    Clf.Fit ()
    Clf.PlotLearnIngCurve ("Train_LearnIngCurve")
    Clf.PlotAccuracyCurve ("Train_AccuracyCurve")

def RunMcPA(Features, Labels, TestFeatures, TestLabels):
    print("McPassive-Aggressive")
    Clf = McPA (Features, Labels, 50);
    Clf.Fit ()
    Clf.PlotLearnIngCurve ("Train_LearnIngCurve")
    Clf.PlotAccuracyCurve ("Train_AccuracyCurve")
    

def RunAll(Features, Labels, TestFeatures, TestLabels):
    RunPerceptron (Features, Labels, TestFeatures, TestLabels)
    RunMcPerceptron (Features, Labels, TestFeatures, TestLabels)
    
    RunAveragePerceptron (Features, Labels, TestFeatures, TestLabels)
    RunMcAveragePerceptron (Features, Labels, TestFeatures, TestLabels)
    
    RunPA (Features, Labels, TestFeatures, TestLabels)
    RunMcPA(Features, Labels, TestFeatures, TestLabels)


def Help ():
    print ("--------------------------------------------------");
    print ("Run.py -t perceptron ---  run perceptron algorithm");
    print ("Run.py -t PA         ---  run PA algorithm");
    print ("--------------------------------------------------");

def main(argv):
    Type = ''

    #########################################################
    # get Type
    #########################################################
    try:
        opts, args = getopt.getopt(argv,"ht:",["type="])
    except getopt.GetoptError:
        print ("Run.py -t <type>")
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            Help ()
            sys.exit()
        elif opt in ("-t", "--type"):
            Type = arg;

    Features, Labels = LoadData ("train.csv")
    TestFeatures, TestLabels = LoadData ("fashion-mnist_test.csv")
    #########################################################
    # collect and analysis
    #########################################################
    if (Type == "perceptron"):
        RunPerceptron (Features, Labels, TestFeatures, TestLabels)
        RunMcPerceptron (Features, Labels, TestFeatures, TestLabels)
            
    elif (Type == "PA"):
        pass
 
    else:
        Help ()
        RunAll (Features, Labels, TestFeatures, TestLabels)
   

if __name__ == "__main__":
   main(sys.argv[1:])
