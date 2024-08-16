"""
Module for considering only a sub part of a trial, useful for reprocessing the extended trials in to shorter experiments

Will also support changing handling of fault divergence and re-running trials as needed
"""
import os
import pickle
import multiprocessing as mp

import jax.numpy as jnp

from failurePy.pipeline import multiprocessingMain, setUpMultiPRocessing
from failurePy.utility.pipelineHelperMethods import diagnoseFailure
from failurePy.utility.reprocessData import reprocessDataAverages, runNewExperimentsWithOldData
from failurePy.utility.computeAlternateReward import computeSquareSumFailureBeliefRewardAlternativeThroughout



def computeAllPartialTrials(originalExperimentAbsoluteSavedDataPath,newExperimentAbsoluteSavedDataPath, subHorizonTimeSteps, reRunFilterDivergence=True,
                            reRunDivergenceHandlingMethod="acceptDiagnosisBeforeNan"):
    """
    Looks at existing (and already safety processed) trials to consider first subHorizonTimeSteps (plus initial condition).

    Recomputes success based on the last belief in the specified directory, and also logs the first time the system becomes unsafe

    Parameters
    ----------
    absoluteSavedDataPath : string
        Absolute path to directory where we will recompute success based on the last belief
    """

    computePartialTrialF=makeComputePartialTrialF(subHorizonTimeSteps, reRunFilterDivergence)
#
    #Updates to the experiment parameters dict to save
    inputUpdateDict = {}
    inputUpdateDict["saveDirectoryPath"] = newExperimentAbsoluteSavedDataPath
    inputUpdateDict["filterDivergenceHandlingMethod"] = reRunDivergenceHandlingMethod
    inputUpdateDict["diagnosisThreshold"] = 1.9 #Overloaded so we accept if we are above 1.9-1 .9 confidence when we diverge. Adding to configs of extended exp (since it doesn't chance anything)
    inputUpdateDict["nExperimentSteps"] = subHorizonTimeSteps
#
#
    returnCode = runNewExperimentsWithOldData(originalExperimentAbsoluteSavedDataPath, computePartialTrialF,
                                              newExperimentAbsoluteSavedDataPath=newExperimentAbsoluteSavedDataPath,inputUpdateDict=inputUpdateDict)
#
    if returnCode == -1:
        #Aborted
        return

    #Recompute the averages using the saved config file!
    #Need to do this first as the alt rewards assumes this exists
    reprocessDataAverages(os.path.join(newExperimentAbsoluteSavedDataPath, "config.yaml")) #Abs path in the config files

    #Recompute last free colli

    #Compute alt rewards here! Since we already got the success (and are assuming safety was already computed on the full trials)\
    computeSquareSumFailureBeliefRewardAlternativeThroughout(newExperimentAbsoluteSavedDataPath,force=True)


def makeComputePartialTrialF(subHorizonTimeSteps, reRunFilterDivergence):
    """
    Constructor for partial trail runner
    """
    def computePartialTrialF(trialDataPath,experimentParamsDict,newExperimentAbsoluteSavedDataPath):
        return computePartialTrial(trialDataPath,experimentParamsDict,newExperimentAbsoluteSavedDataPath, subHorizonTimeSteps, reRunFilterDivergence)
    return computePartialTrialF

def computePartialTrial(trialDataPath,inputDict,newExperimentAbsoluteSavedDataPath, subHorizonTimeSteps, reRunFilterDivergence=True):
    """
    Abridges Trial to just the horizon requested. If the filter has diverged at this point, re-runs the trial using
    method to deal with the divergence

    Parameters
    ----------
    trialDataPath : string
        Path to trial data, useful because it contains the rng seed used for this trial
    inputDict
        Yaml load of experiment parameters
    newExperimentAbsoluteSavedDataPath : string
        Absolute path to directory where we should save modified (or rerun) trailData.dict files.
    subHorizonTimeSteps : int
        The number of time steps in this re-processed dict (should be saved to new location to avoid loosing data!)
    reRunFilterDivergence : Boolean (default=True)
        If true, will re-run any trials where the filter diverges before the end of the time period
    """

    #Load dictionary
    with open(trialDataPath, "rb") as trialDataFile:
        trialResultsDict = pickle.load(trialDataFile)
    #Reprocess! (And Rerun if needed)

    #Need to get relative path. Could probably be more efficient by just using string ops directly
    dictNameAndPathTuple = os.path.split(trialDataPath)
    trialNameAndPathTuple = os.path.split(dictNameAndPathTuple[0])
    subExpNameAndPathTuple = os.path.split(trialNameAndPathTuple[0])
    solverNameAndPathTuple = os.path.split(subExpNameAndPathTuple[0])
    #expNameAndPathTuple = os.path.split(solverNameAndPathTuple[0])
    #New path
    newTrialDataPath = os.path.join(newExperimentAbsoluteSavedDataPath,solverNameAndPathTuple[1],
                                    subExpNameAndPathTuple[1],trialNameAndPathTuple[1],dictNameAndPathTuple[1])



    #First check if filter diverged at end of desired horizon
    if jnp.isnan(trialResultsDict["rewards"][subHorizonTimeSteps+1]) and reRunFilterDivergence:
        #Should we dispatch this as a sub process??? No collision with other processes
        #Handle with new method, will call pipeline with modified config
        #will modify experiment parameters a little and hand to pipeline's running method
        inputDict["nSimulationsPerTree"] = [int(subExpNameAndPathTuple[1])]
        inputDict["nTrialsPerPoint"] = 1
        inputDict["multiprocessingFlag"] = False #Shouldn't matter here
        inputDict["rngKeysOffset"] = int(trialNameAndPathTuple[1])
        print(f"Re Running {inputDict['nSimulationsPerTree']} {inputDict['rngKeysOffset']}")
        #NOT using multiprocessingQueue
        process = mp.Process(target=multiprocessingMain,args=(inputDict,newExperimentAbsoluteSavedDataPath,None,False),daemon=True)
        #All seem to die, not sure why
        #process = mp.Process(target=runExperimentsAndLog,args=(experimentParamsDict,newExperimentAbsoluteSavedDataPath,True),daemon=True)
        return process #Dict will be saved when this ends, calling method handles process joining
        #runExperimentsAndLog(experimentParamsDict,newExperimentAbsoluteSavedDataPath,True)
        #return None

    #Otherwise, abridging existing data

    #Abridge, first check it isn't already shorter due to early termination
    originalHorizonTimeStepsPlusOne = len(trialResultsDict["physicalStateList"])
    if originalHorizonTimeStepsPlusOne < subHorizonTimeSteps+1:
        trialResultsDict["rewards"] = trialResultsDict["rewards"][subHorizonTimeSteps+1:]
        return trialResultsDict
    for key in trialResultsDict.keys():
        if key in ("success", "wallClockTime"):
            #Will update at the end or assume average is still good
            continue
        if key == "steps":#This is the nominal time steps (not including the +1 for the initial time steps. Consider it transitions only)
            trialResultsDict[key] = subHorizonTimeSteps
        elif key == "lastBelievedCollisionFreeTStep":
            #Need to check if this is after new end
            trialResultsDict[key] = min(trialResultsDict[key],subHorizonTimeSteps)
        else:
            #print(key) want everything up to our sub horizon (+1)
            trialResultsDict[key] = trialResultsDict[key][:subHorizonTimeSteps+1]

    #Update success of trials
    trialResultsDict = updateSuccessStatusOfTrialDataDict(trialResultsDict)

    rewards = trialResultsDict["rewards"]
    #Loop through and see when we become unsafe (die and stay dead logic)
    lastBelievedCollisionFreeTStep = -1
    for reward in rewards:
        if reward == 0:
            break
        lastBelievedCollisionFreeTStep += 1
    trialResultsDict["lastBelievedCollisionFreeTStep"] = lastBelievedCollisionFreeTStep
    with open(newTrialDataPath,'wb') as trialDataFile:
        pickle.dump(trialResultsDict,trialDataFile)
    #And update human readable copy
    trialDataTextPath = os.path.splitext(newTrialDataPath)[0]+'.txt'
    with open(trialDataTextPath, "w",encoding="UTF-8") as textFile:
        textFile.write(str(trialResultsDict))

    #Now we don't need to return anything, dict is saved
    return None # No process

def updateSuccessStatusOfTrialDataDict(trialResultsDict):
    """
    Recomputes success based on the last belief of this trial

    Parameters
    ----------
    trialResultsDict : dict
        trial data we will recompute the success on
    """

    #Get the beliefList last entry and true failure
    lastBelief = trialResultsDict["beliefList"][-1]
    failure = trialResultsDict["failureStateList"][-1]

    #Get predicted failure
    diagnosis = diagnoseFailure(lastBelief, trialResultsDict["possibleFailures"])

    if jnp.all(diagnosis == failure):
        trialResultsDict["success"] = 1
    else:
        trialResultsDict["success"] = 0

    return trialResultsDict


if __name__ == "__main__":
    setUpMultiPRocessing() #THIS IS ESSENTIAL TO AVOID HANGING!
    SAVED_DATA_DIRECTORY = None #SET TO THE DIRECTORY OF THE ORIGINAL EXPERIMENT YOU WANT TO TAKE A PARTIAL TRIAL FROM

    NEW_DATA_DIRECTORY =  None #SET TO THE DIRECTORY YOU WISH TO SAVE THE OUTPUT DATA TO

    computeAllPartialTrials(originalExperimentAbsoluteSavedDataPath=SAVED_DATA_DIRECTORY,newExperimentAbsoluteSavedDataPath=NEW_DATA_DIRECTORY, subHorizonTimeSteps=15)
