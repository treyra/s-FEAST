"""
Module of common methods for visualization
"""

import os
import pickle
import matplotlib.pyplot as plt
import numpy as onp

def loadBeliefData(savedDataDirPath,experimentName,solverName,nSimulationsPerTree,nTrial=0):
    """
    Function that takes experiment ResultsList and grabs the first trial result to visualize

    Parameters
    ----------
    savedDataDirPath : str
        String of the absolute path to the saveDirectory for we are loading the experiment from.
    experimentName : str
        Name of the experiment we are loading
    solverName : str
        The solver to get the data from
    nSimulationsPerTree: int
        The  number of simulations per tree to get data from.
    nTrial : int (default=None)
        The trial to get the data from

    Returns
    -------
    beliefList : list
        List of the beliefs for each time step
    initialFailureParticles : array, shape(nMaxFailureParticles,numAct+numSen)
        List of possible failures to be considered by the solver. If not exhaustive, these are a random subset. Belief failure weights are are over these possibilities
    failureState : array, shape(numAct+numSen)
        True failure state
    """
    #First we need to collect the data
    experimentDataDirPath = os.path.join(savedDataDirPath,experimentName)
    solverDirectoryPath = os.path.join(experimentDataDirPath,solverName)
    nSimPath =  os.path.join(solverDirectoryPath,str(nSimulationsPerTree))
    #Assuming only one trial/interested only in first trial
    trialDataPath = os.path.join(nSimPath,str(nTrial))
    trialDataPath = os.path.join(trialDataPath,"trialData.dict")

    #Load
    with open(trialDataPath,'rb') as trialDataFile:
        trialDataDict = pickle.load(trialDataFile)

    #Now get out the data we want
    failureState = trialDataDict["failureStateList"][0]

    beliefList = trialDataDict["beliefList"]

    initialFailureParticles = trialDataDict["possibleFailures"]

    physicalStateList = trialDataDict["physicalStateList"]

    return beliefList,initialFailureParticles,failureState,physicalStateList


def setColorCyclerDistinct(repeatNum,ax):
    """
    Creates a color cycler with the specified number of distinct colors and assigns to the provided axis
    """
    if repeatNum <=8:
        ax.set_prop_cycle('color',plt.cm.Dark2(onp.linspace(0,1,repeatNum)))
    elif repeatNum <= 10:
        ax.set_prop_cycle('color',plt.cm.tab10(onp.linspace(0,1,repeatNum)))
    elif repeatNum <= 20:
        ax.set_prop_cycle('color',plt.cm.tab20(onp.linspace(0,1,repeatNum)))
    else:
        ax.set_prop_cycle('color',plt.cm.gist_rainbow(onp.linspace(0,1,repeatNum)))
