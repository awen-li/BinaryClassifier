#!/usr/bin/python

import sys, getopt
import pandas as pd
from lib.Perceptron import Perceptron, McPerceptron
from lib.PA import PA, McPA


def LoadData (FileName):
    RawData = pd.read_csv("data/" + FileName, header=0)
    Data = RawData.values

    Features = Data[0::, 1::]
    Labels   = Data[::, 0]

    return Features, Labels

def RunPerceptron(Features, Labels):
    print("RunPerceptron")
    Clf = Perceptron (Features, Labels, 1);
    Clf.Fit ()

def RunMcPerceptron(Features, Labels):
    print("McPerceptron")
    Clf = McPerceptron (Features, Labels, 1);
    Clf.Fit ()


def RunPA(Features, Labels):
    print("Passive-Aggressive")
    Clf = PA (Features, Labels, 1);
    Clf.Fit ()

def RunMcPA(Features, Labels):
    print("Passive-Aggressive")
    Clf = McPA (Features, Labels, 1);
    Clf.Fit ()
    

def RunAll(Features, Labels):
    RunPerceptron (Features, Labels)
    RunMcPerceptron(Features, Labels)
    RunPA (Features, Labels)
    RunMcPA(Features, Labels)


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
    #########################################################
    # collect and analysis
    #########################################################
    if (Type == "perceptron"):
        RunPerceptron (Features, Labels)
        RunMcPerceptron (Features, Labels)
            
    elif (Type == "PA"):
        pass
 
    else:
        Help ()
        RunAll (Features, Labels)
   

if __name__ == "__main__":
   main(sys.argv[1:])
