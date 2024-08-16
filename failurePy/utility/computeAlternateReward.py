""""
Function to compute a different reward metric given the trial data. Uses the fact the beliefs are saved to reprocess
"""

import os
#from functools import partial
import pickle
import numpy as onp

from tqdm import tqdm
#import jax
import jax.numpy as jnp

from failurePy.utility.reprocessData import confirmDataDeletion, reprocessTrialDataDicts
from failurePy.rewards.squareSumFailureBeliefReward import squareSumFailureBeliefReward


#@partial(jax.jit, static_argnames=['rewardF'])
def computeAlternateReward(trialResultsDict, rewardF):
    """
    Computes the new reward given a trialDataDict. Currently doesn't support approximate chance constraint evaluation
    """

    beliefList = trialResultsDict["beliefList"]

    altRewards = onp.zeros(len(beliefList))
    for iReward, belief in enumerate(beliefList):
        altRewards[iReward] = rewardF(belief)

    trialResultsDict["altRewards"] = altRewards

    return trialResultsDict


def makeAlternativeRewardReprocessor(rewardF):
    """
    Constructor for making reprocessing function. Doing this to fit into reprocessData modular design
    """

    def alternativeRewardReprocessorF(trialResultsDict):
        return computeAlternateReward(trialResultsDict, rewardF)

    return alternativeRewardReprocessorF


def processAlternativeRewardsAverages(saveDirectoryAbsolutePath):
    """
    Adds alternative rewards averages to existing data averages

    Parameters
    ----------
    saveDirectoryAbsolutePath : str
        String of the absolute path to the saveDirectory for the results of the experiment. Auto determined from the experiment parameters, unless overwritten.
    """

    #Get all the solvers using os walk
    solverNamesList = next(os.walk(saveDirectoryAbsolutePath))[1]
    #Get the experiments (different number of trials)
    solverDirectoryPath = os.path.join(saveDirectoryAbsolutePath, solverNamesList[0])
    #Using numpy to cast strings to ints then to jax
    nSimulationsPerTrees = jnp.array(onp.array(next(os.walk(solverDirectoryPath))[1], dtype=onp.int32))
    #And get number of simulations per point
    nSimPath = os.path.join(solverDirectoryPath, str(nSimulationsPerTrees[0]))
    trialRandomSeeds = next(os.walk(nSimPath))[1]
    nTrialsPerPoint = len(trialRandomSeeds)
    #Get number of time steps
    trialDataPath = os.path.join(nSimPath, str(trialRandomSeeds[0]), "trialData.dict")
    with open(trialDataPath, "rb") as trialDataFile:
        trialDataDict = pickle.load(trialDataFile)
    numTimeSteps = len(trialDataDict["altRewards"])

    #Now we can loop
    for solverName in solverNamesList:
        solverDirectoryPath = os.path.join(saveDirectoryAbsolutePath, solverName)

        #Loop through each number of sims per tree
        for nSimulationsPerTree in nSimulationsPerTrees:
            nSimPath = os.path.join(solverDirectoryPath, str(nSimulationsPerTree))
            #Array to average over (use onp for mutability later when plotting or otherwise)
            avgAltRewards = onp.zeros(numTimeSteps)
            varAltRewards = onp.zeros(numTimeSteps)
            #Not going to include points where the reward diverged in variance calculation
            numNonNans = onp.zeros(numTimeSteps)
            #Loop through each trial (Might be inefficient? Will check later)
            try:
                for trialRandomSeed in tqdm(trialRandomSeeds):
                    trialDataPath = os.path.join(nSimPath, str(trialRandomSeed), "trialData.dict")
                    with open(trialDataPath, "rb") as trialDataFile:
                        trialDataDict = pickle.load(trialDataFile)
                    #Check for nans
                    trialDataDict["altRewards"][onp.isnan(trialDataDict["altRewards"])] = 0
                    avgAltRewards = onp.add(avgAltRewards, trialDataDict["altRewards"])
                    #Use fact reward is only zero if we just set it to 0 because of nan, otherwise strictly positive
                    numNonNans = onp.add(numNonNans, onp.sign(trialDataDict["altRewards"]))
            except ValueError as arrayError:
                tooManyAltRewards = f"too many alt rewards found in {trialDataPath}"
                raise ValueError(tooManyAltRewards) from arrayError
            #Average for this experiment
            avgAltRewards = avgAltRewards / nTrialsPerPoint

            #Compute variance as well (need to already know the average rewards, so have to loop again)

            for trialRandomSeed in tqdm(trialRandomSeeds):
                trialDataPath = os.path.join(nSimPath, str(trialRandomSeed), "trialData.dict")
                with open(trialDataPath, "rb") as trialDataFile:
                    trialDataDict = pickle.load(trialDataFile)
                #Check for nans, in this case, we set them to the *average* (so variance isn't effected, since already taking the skipped ones into account)
                trialDataDict["altRewards"][onp.isnan(trialDataDict["altRewards"])] = avgAltRewards[onp.isnan(trialDataDict["altRewards"])]

                #Only update variance if not nan (as will be 0 if it was nan)
                varAltRewards = onp.add(varAltRewards,onp.square(jnp.subtract(trialDataDict["altRewards"], avgAltRewards)),)

            #Add to average data dict and save
            experimentAvgDataPath = os.path.join(nSimPath, "averageData.dict")
            with open(experimentAvgDataPath, "rb") as experimentAvgDataFile:
                experimentAvgDataDict = pickle.load(experimentAvgDataFile)
            experimentAvgDataDict["avgAltRewards"] = avgAltRewards
            experimentAvgDataDict["varAltRewards"] = onp.divide(varAltRewards, numNonNans - 1)  #Each time step could have different non-nans

            with open(experimentAvgDataPath, "wb") as experimentAvgDataFile:
                experimentAvgDataDict = pickle.dump(experimentAvgDataDict, experimentAvgDataFile)


def computeAlternativeRewardsThroughout(absoluteSavedDataPath, alternativeRewardReprocessorF, force=False):
    """
    Computes alternative reward based on the last belief in the specified directory, and also logs the first time the system becomes unsafe

    Parameters
    ----------
    absoluteSavedDataPath : string
        Absolute path to directory where we will recompute success based on the last belief
    """

    if force or confirmDataDeletion(absoluteSavedDataPath, "any previous alternative reward"):
        reprocessTrialDataDicts(absoluteSavedDataPath, alternativeRewardReprocessorF)

        #Add altRewards to the averages.
        processAlternativeRewardsAverages(absoluteSavedDataPath)
    else:
        print("The alternate rewards will not be updated")


def computeSquareSumFailureBeliefRewardAlternativeThroughout(absoluteSavedDataPath,force=False):
    """
    Computes alternative reward as squareSumFailureBeliefReward
    """

    computeAlternativeRewardsThroughout(absoluteSavedDataPath,alternativeRewardReprocessorF=makeAlternativeRewardReprocessor(squareSumFailureBeliefReward),force=force)


if __name__ == "__main__":
    SAVED_DATA_DIRECTORY = None #SET TO THE DIRECTORY YOU WISH TO COMPUTE THE ALTERNATIVE REWARD IN


    computeSquareSumFailureBeliefRewardAlternativeThroughout(SAVED_DATA_DIRECTORY,force=True)
