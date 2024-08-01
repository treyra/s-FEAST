"""
File for the functions need for saving the data
"""

import os
import time

import shutil
from importlib.metadata import version
import datetime
from hashlib import md5
from inspect import getsource

#import sys
import pickle
import errno
import numpy as onp
from tqdm import tqdm
import yaml

import jax.numpy as jnp


#FailurePy files for hashing to verify version
from failurePy.estimators import extendedKalmanFilterLinearSensing, kalmanFilter,kalmanFilterCommon, marginalizedFilter
from failurePy.utility.pipelineHelperMethods import diagnoseFailure


def processDataAverages(saveDirectoryAbsolutePath, experimentParamsDict,):
    """
    Logs data for future analysis

    Parameters
    ----------
    saveDirectoryAbsolutePath : str
        String of the absolute path to the saveDirectory for the results of the experiment. Auto determined from the experiment parameters, unless overwritten.
    experimentParamsDict : dict
        Dictionary containing all the relevant experiment parameters.
        Relevant contents as follows:
            nSimulationsPerTreeList : list, len(numTrials)
                The number of simulations performed before returning an action (when not running on time out mode).
                This parameter is an array, if longer then length 1, multiple trials are run, varying the number of simulations per tree.
            dt : float
                The time between time steps of the experiment
            nExperimentSteps : int
                How many time steps are in the experiment
            nTrialsPerPoint : int
                The number of repeated trials per configuration.
            nMaxComponentFailures : int
                Maximum number of simultaneous failures of components that can be considered
            nMaxPossibleFailures : int
                Maximum number of possible failures to consider. If larger than the number of possible unique failures, all possibilities are considered
            providedFailure : array, shape(numAct+numSen) (default=None)
                Provided failure (if any) to have each trial use
            systemF : function
                Function reference of the system to call to run experiment
            systemParametersTuple : tuple
                Tuple of system parameters needed
            solverFList : list
                List of solver functions to try
            solverParametersListsList : list
                List of lists of solver parameters. Included action list, failure scenarios
            solverNamesList: list
                List of names of solvers, for data logging
            estimatorF : function
                Estimator function to update the beliefs with. Takes batch of filters
            physicalStateSubEstimatorF : function
                Physical state estimator for use with the marginal filter, if any
            physicalStateJacobianF : function
                Jacobian of the model for use in estimating.
            physicalStateSubEstimatorSampleF : function
                Samples from the belief corresponding to this estimator
            beliefInitializationF : function
                Function that creates the initial belief
            rewardF : function
                Reward function to evaluate the beliefs with
            rngKeysOffset : int
                Offset to the initial PRNG used in generating the initial failure states and randomness in the trials. This is added to the trial number to allow for different trials to be preformed
            multiprocessingFlag : boolean
                Wether to use multi-processing or not
            saveTreeFlag : boolean
                Whether to save the tree or not (it can be quite large, so if not visualizing, it is best to set this to false)
            clobber : boolean
                Wether to overwrite existing data or not
    """


    #Loop through solvers
    for solverName in experimentParamsDict["solverNamesList"]:
        solverDirectoryPath = os.path.join(saveDirectoryAbsolutePath,solverName)

        #Loop through each number of sims per tree
        for nSimulationsPerTree in experimentParamsDict["nSimulationsPerTreeList"]:
            nSimPath =  os.path.join(solverDirectoryPath,str(nSimulationsPerTree))

            #Make arrays to average quantities over. We use numpy here because speed isn't a priority and flexibility is nice
            avgRewards = onp.zeros(experimentParamsDict["nExperimentSteps"]+1) #Need to add one because we track initial reward too!
            cumulativeAvgRewards = onp.zeros(experimentParamsDict["nExperimentSteps"]+1) #Need to add one because we track initial reward too!
            avgSuccessRate = 0
            avgWallClockTime = 0
            avgSteps = 0
            #Also take the variance!
            varRewards = onp.zeros(experimentParamsDict["nExperimentSteps"]+1) #Need to add one because we track initial reward too!
            cumulativeVarRewards = onp.zeros(experimentParamsDict["nExperimentSteps"]+1) #Need to add one because we track initial reward too!
            varWallClockTime = 0

            #Loop through each trial (Might be inefficient? Will check later)
            for nTrial in tqdm(range(experimentParamsDict["nTrialsPerPoint"])):
                nTrialPath =  os.path.join(nSimPath,str(nTrial+experimentParamsDict["rngKeysOffset"])) #Make directory according to the rngKeyUSed

                #Load in trial results (pickled)
                trialDataPath = os.path.join(nTrialPath,"trialData.dict")
                with open(trialDataPath, "rb") as trialDataFile:
                    trialResultsDict = pickle.load(trialDataFile)

                #Deal with nans, which can occur when the EKF goes unstable. We will treat these as 0 reward, and set success to zero for this trial
                trialRewards = trialResultsDict["rewards"]
                if onp.isnan(trialRewards[-1]):
                    trialRewards[onp.isnan(trialRewards)] = 0
                    trialResultsDict["success"] = 0

                #Add quantities to the average totals
                avgRewards = onp.add(avgRewards,trialResultsDict["rewards"])
                cumulativeAvgRewards = onp.add(cumulativeAvgRewards,onp.cumsum(trialResultsDict["rewards"]))
                #Might set success rate to nan if we don't want to count failing to converge against alg, so check for this
                if not onp.isnan(trialResultsDict["success"]):
                    avgSuccessRate += trialResultsDict["success"]
                avgWallClockTime += trialResultsDict["wallClockTime"]
                avgSteps += trialResultsDict["steps"]

            avgRewards = avgRewards/experimentParamsDict["nTrialsPerPoint"]
            cumulativeAvgRewards = cumulativeAvgRewards/experimentParamsDict["nTrialsPerPoint"]
            avgWallClockTime = avgWallClockTime/experimentParamsDict["nTrialsPerPoint"]

            #Compute variance as well (need to already know the average rewards, so have to loop again)
            for nTrial in tqdm(range(experimentParamsDict["nTrialsPerPoint"])):
                nTrialPath =  os.path.join(nSimPath,str(nTrial+experimentParamsDict["rngKeysOffset"])) #Make directory according to the rngKeyUSed

                #Load in trial results (pickled)
                trialDataPath = os.path.join(nTrialPath,"trialData.dict")
                with open(trialDataPath, "rb") as trialDataFile:
                    trialResultsDict = pickle.load(trialDataFile)

                varRewards = onp.add(varRewards, onp.square(onp.subtract(trialResultsDict["rewards"],avgRewards)))
                cumulativeVarRewards = onp.add(cumulativeVarRewards, onp.square(onp.subtract(onp.cumsum(trialResultsDict["rewards"]),cumulativeAvgRewards)))
                varWallClockTime += (trialResultsDict["wallClockTime"] - avgWallClockTime)**2

            #Take average
            experimentAverageDataDict = {
                "avgRewards" : avgRewards,
                "cumulativeAvgRewards": cumulativeAvgRewards,
                "avgSuccessRate" : avgSuccessRate/experimentParamsDict["nTrialsPerPoint"],
                "avgWallClockTime" : avgWallClockTime,
                "avgSteps" : avgSteps/experimentParamsDict["nTrialsPerPoint"],
                "varRewards" : varRewards/(experimentParamsDict["nTrialsPerPoint"]-1),
                "cumulativeVarRewards" : cumulativeVarRewards/(experimentParamsDict["nTrialsPerPoint"]-1),
                "varWallClockTime" : onp.float64(varWallClockTime)/(experimentParamsDict["nTrialsPerPoint"]-1), #Guarding against divide by zero
            }

            #And save averages
            #For now just pickle dump the full dictionary, consider more sophisticated saving later

            experimentDataPath = os.path.join(nSimPath,"averageData.dict")
            with open(experimentDataPath,'wb') as experimentDataFile:
                pickle.dump(experimentAverageDataDict,experimentDataFile)





def saveMetaData(saveDirectoryPath,experimentParamsDict):
    """
    Saves meta data for the experiment

    Parameters
    ----------
    saveDirectoryPath : str
        String of the absolute path to the saveDirectory for the results of the experiment. Auto determined from the experiment parameters, unless overwritten.
    experimentParamsDict : dict
        Dictionary containing all the relevant experiment parameters.
    """


    #Make yaml file
    metaData = makeMetaData(experimentParamsDict)

    #Write to file
    with open(os.path.join(saveDirectoryPath,"metaData.yaml"), "w", encoding = "utf-8") as metaDataFile:
        metaDataFile.write(metaData)
        #Overwrite if anything existed before (we checked with user already!)
        metaDataFile.truncate()

def copyConfigFileIfNeeded(saveDirectoryAbsolutePath,configFilePath):
    """
    Saves meta data for the experiment by copying config file. Doesn't copy if paths are the same

    Parameters
    ----------
    saveDirectoryAbsolutePath
        String of the absolute path to the saveDirectory for the results of the experiment. Auto determined from the experiment parameters, unless overwritten.
    configFilePath : String
        Relative (or absolute) path to the config file for the experiment
    """

    saveConfigFilePath = os.path.join(saveDirectoryAbsolutePath,"config.yaml")

    #Don't copy if a file already exists
    if os.path.exists(saveConfigFilePath):
        return
    try :
        shutil.copyfile(configFilePath,saveConfigFilePath)
    except shutil.SameFileError:
        pass #Don't need to do anything here, as we are saving the data to the folder the config already exists in

def saveFailurePyVersion(saveDirectoryAbsolutePath):
    """
    Saves the current failurePy version, to make future experiments more readily reproducible.

    Parameters
    ----------
    saveDirectoryAbsolutePath
        String of the absolute path to the saveDirectory for the results of the experiment. Auto determined from the experiment parameters, unless overwritten.
    """

    versionFilePath = os.path.join(saveDirectoryAbsolutePath,"version.txt")

    with open(versionFilePath, 'w',encoding="UTF-8") as versionFile:
        versionFile.write(version('failurePy')) #NEED TO MAKE SURE PIP UPDATES VERSION NUMBER, it won't be default in the -e format unless you manually re-install
        #Add time stamp as well
        versionFile.write(" " + str(datetime.datetime.now()))
        versionFile.write("\n\n")
        versionFile.write(getEstimatorsHash())

def getEstimatorsHash():
    """
    Method that provides hash of the estimation methods, to identify if they have changed (useful when testing different estimation ideas in hard code)

    Hardcoded now for repeatability
    """
    #Start hash
    estimatorHash = md5()
    estimatorHash.update(getsource(extendedKalmanFilterLinearSensing).encode(encoding = 'UTF-8'))
    estimatorHash.update(getsource(kalmanFilter).encode(encoding = 'UTF-8'))
    estimatorHash.update(getsource(kalmanFilterCommon).encode(encoding = 'UTF-8'))
    estimatorHash.update(getsource(marginalizedFilter).encode(encoding = 'UTF-8'))
    return estimatorHash.hexdigest()

def makeMetaData(experimentParamsDict):
    """
    Function that makes meta data file from the experimentParamsDict

    Parameters
    ----------
    experimentParamsDict : dict
        Dictionary containing all the relevant experiment parameters.

    Returns
    -------
    metaDataString : str
        String of all the meta data
    """

    return yaml.dump(experimentParamsDict, sort_keys=False)

def checkIfDataExists(dataPath):
    """
    Function that checks if data exists at the specified path

    Parameters
    ----------
    dataPath : string
        Absolute path to data to check for.

    Returns
    -------
    dataExistsFlag : boolean
        True if data exists
    """

    return os.path.isfile(dataPath)




def checkSaveDirectoryPathMakeIfNeeded(saveDirectoryAbsolutePath,experimentParamsDict):
    """
    Checks if the directory exists, making it if not, and prompting the user if it already exists and is incompatible with
    the current experiment

    Parameters
    ----------
    saveDirectoryAbsolutePath : str
        String of the absolute path to the saveDirectory for the results of the experiment. Auto determined from the experiment parameters, unless overwritten.
    experimentParamsDict : dict
        Dictionary containing all the relevant experiment parameters.
    """

    #Now we check if any data exists (by checking the meta data)
    if os.path.exists(saveDirectoryAbsolutePath):
        #Merge with existing data (this is blind for now)
        if experimentParamsDict["mergeData"]:
            return #Don't touch existing data

        #Check for clobber param
        if experimentParamsDict["clobber"]:
            shutil.rmtree(saveDirectoryAbsolutePath)
        else:
            #Check if there is anything other than config.yaml
            itemsInSaveDirectory = os.listdir(saveDirectoryAbsolutePath)
            for item in itemsInSaveDirectory:
                #This should only trigger in the case that we are saving in the directory the config file exists.
                if not os.path.isfile(os.path.join(saveDirectoryAbsolutePath,item)) or not item == "config.yaml":
                    dataExists = f"Data ({item}) already exists at {saveDirectoryAbsolutePath}! To overwrite, set <clobber> to true in the config yaml file."
                    raise FileExistsError(dataExists)


    #Now check for the folders to exist and make them if not
    makeDirectoryAndParents(saveDirectoryAbsolutePath)


def makeDirectoryAndParents(saveDirectoryPath):
    """
    Recursively makes directory, making parents as needed

    Parameters
    ----------
    saveDirectoryPath : str
        String of the absolute path to the saveDirectory for the results of the experiment. Auto determined from the experiment parameters, unless overwritten.
    """

    #Check if the directory exists
    if os.path.exists(saveDirectoryPath):
        return

    #Split off the parent path to check it
    parentPath = saveDirectoryPath.rsplit("/",1)

    #Check if the path used \\ instead.
    if parentPath[0] == saveDirectoryPath:
        parentPath = saveDirectoryPath.rsplit("\\",1)

    #If we still failed to split, we don't have a parent. Otherwise let's check that it exists or make it
    if parentPath[0] != saveDirectoryPath:
        makeDirectoryAndParents(parentPath[0])
        print(f"Making {saveDirectoryPath}")
        #Now make this directory
        os.mkdir(saveDirectoryPath)

def checkOrMakeDirectory(saveDirectoryPath,subDirectory):
    """
    Makes subdirectory (if it doesn't exist) in the saveDirectoryPath

    Parameters
    ----------
    saveDirectoryPath : str
        String of the absolute path to the saveDirectory for the results of the experiment. Auto determined from the experiment parameters, unless overwritten.
    subDirectory : str
        String of the subdirectory to make. Should NOT have further subDirectories under it.

    Returns
    -------
    subDirectoryPath : str
        Path to the new directory, or None if it failed
    returnCode : int
        Flag on the result of the operation. Possible values:
            2   : Directory already exists
            1   : Nominal success
            -1  : Failed to make directory because it already exists.
            -2  : subDirectory has further sub directories (\\ ir / characters in the string)
    """
    if "\\" in subDirectory or "/" in subDirectory:
        return None, -2

    subDirectoryPath = os.path.join(saveDirectoryPath,subDirectory)
    if os.path.exists(subDirectoryPath):
        return subDirectoryPath, 2

    #Attempt to make the directory
    try:
        os.mkdir(subDirectoryPath)
        return subDirectoryPath, 1
    #Catches directory already exists error. Raises if it is a different OSError (like out of space or permission denied)
    except OSError as exc:

        if exc.errno != errno.EEXIST:
            raise #Re-raises the exception
        return None, -1

def makeTrialResultDict(physicalStateList,failureStateList,beliefList,rewards,actionList,initialFailureParticles,success,timeStep,wctStartTime,saveTreeFlag,treeList,computeSafetyAtEnd=False):
    """
    Helper function that makes the trial result dictionary
    (since it has a lot of members and to minimize repeat code)
    """
    trialResultsDict = {"physicalStateList" : physicalStateList,
                        "failureStateList" : failureStateList,
                        "beliefList" : beliefList,
                        "rewards" : rewards,
                        "actionList" : actionList,
                        "possibleFailures" : initialFailureParticles, #Not changing name for backwards compatibility issue, should be okay because general faults will not be using this
                        "success" : success,
                        #0 indexed to 1 index on experiment time (NOTE the length of all of our lists is timeStep + 2)
                        "steps": timeStep+1, #Assuming nExperimentSteps is > 0, pylint: disable=undefined-loop-variable
                        "wallClockTime" : (time.time()-wctStartTime)/(timeStep+1)} #Assuming nExperimentSteps is > 0, pylint: disable=undefined-loop-variable

    if computeSafetyAtEnd:
        trialResultsDict = updateSuccessStatusAndSafetyOfTrialDataDict(trialResultsDict)

    #Now add None to the end of the trees (if saving the tree)
    if saveTreeFlag: #Only save tree if told to
        treeList.append(None)
        trialResultsDict["treeList"] = treeList
    return trialResultsDict

def updateSuccessStatusAndSafetyOfTrialDataDict(trialResultsDict):
    """
    Recomputes success based on the last belief of this trial using sub method and also
    computes when the system is first unsafe.

    Parameters
    ----------
    trialResultsDict : dict
        trial data we will recompute the success on
    """

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
    return trialResultsDict


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

    return trialResultsDict

def setUpDataOutput(saveDirectoryPath, experimentParamsDict, configFilePath):
    """
    Creates directories for logging data

    Parameters
    ----------
    saveDirectoryPath : str
        String of the absolute path to the saveDirectory for the results of the experiment. Auto determined from the experiment parameters, unless overwritten.
    experimentParamsDict : dict
        Dictionary containing all the relevant experiment parameters.
        Relevant parameters:
            nSimulationsPerTreeList : list, len(numTrials)
                The number of simulations performed before returning an action (when not running on time out mode).
                This parameter is an array, if longer then length 1, multiple trials are run, varying the number of simulations per tree.
            nTrialsPerPoint : int
                The number of repeated trials per configuration.
            solverNamesList: list
                List of names of solvers, for data logging
            rngKeysOffset : int
                Offset to the initial PRNG used in generating the initial failure states and randomness in the trials.
                This is added to the trial number to allow for different trials to be preformed
    configFilePath : String
        Relative path to the config file for the experiment
    """

    #Copy original config file over. Don't copy if they are already the same
    copyConfigFileIfNeeded(saveDirectoryPath,configFilePath)
    #Save version (this will make repeating experiments easier)
    saveFailurePyVersion(saveDirectoryPath)

    #Loop through each solver
    for solverName in experimentParamsDict["solverNamesList"]:
        #Check if the directory exists already (in case of merging compatible data )
        #Otherwise make directory (returnCode for future handling of existing directories.
        #Returns -1 if it fails, -2 if it fails due to SolverName having / or \\ in it, shouldn't happen)
        solverDirectoryPath,returnCode =  checkOrMakeDirectory(saveDirectoryPath,solverName)
        if not returnCode == 1:
            unexpectedDirectoryReturn = f"Checking the directory {solverDirectoryPath} gave an unexpected result, error code {returnCode}. Error handling not implemented yet"
            raise NotImplementedError(unexpectedDirectoryReturn) #Not trying to handle anything complicated yet.

        #Loop through each number of sims per tree we try and make sub folders for each
        for nSimulationsPerTree in experimentParamsDict["nSimulationsPerTreeList"]:
            #Check if the directory exists already (in case of merging compatible data )
            #Otherwise make directory (returnCode for future handling of existing directories. Returns -1 if it fails, -2 if it fails due to SolverName having / or \\ in it, shouldn't happen)
            nSimPath,returnCode =  checkOrMakeDirectory(solverDirectoryPath,str(nSimulationsPerTree))
            if not returnCode == 1:
                unexpectedDirectoryReturn = f"Checking the directory {solverDirectoryPath} gave an unexpected result, error code {returnCode}. Error handling not implemented yet"
                raise NotImplementedError(unexpectedDirectoryReturn) #Not trying to handle anything complicated yet.

            #Now we create folders for each trial (Might be inefficient? Will check later)
            for nTrial in range(experimentParamsDict["nTrialsPerPoint"]):
                #Check if the directory exists already (in case of merging compatible data )
                #Otherwise make directory (returnCode for future handling of existing directories. Returns -1 if it fails, -2 if it fails due to SolverName having / or \\ in it, shouldn't happen)
                dummyNTrialPath,returnCode =  checkOrMakeDirectory(nSimPath,str(nTrial+experimentParamsDict["rngKeysOffset"])) #Make directory according to the rngKeyUSed
                if not returnCode == 1:
                    unexpectedDirectoryReturn = f"Checking the directory {nSimPath} gave an unexpected result, error code {returnCode}. Error handling not implemented yet"
                    raise NotImplementedError(unexpectedDirectoryReturn) #Not trying to handle anything complicated yet.
                #We are now as low down as we need to go
    #We're done pre-allocating

def saveTrialResult(saveDirectoryPath, solverName, nSimulationsPerTree, nTrial,trialResultDict):
    """
    Method that saves a single experiment result

    Parameters
    ----------
    saveDirectoryPath : str
        String of the absolute path to the saveDirectory for the results of the experiment. Auto determined from the experiment parameters, unless overwritten.
    solverName : str
        Names of solver to log under
    nSimulationsPerTree :  int
        Number of simulations per tree to log under
    nTrial : int
        Trial number to log under (rng seed used)
    """
    #Save the data to the pre-created directories (made when setUpDataOutput is called)

    trialDataPath = getTrialDataPath(saveDirectoryPath, solverName, nSimulationsPerTree, nTrial)

    with open(trialDataPath,'wb') as trialDataFile:
        pickle.dump(trialResultDict,trialDataFile)

    #Dump as text file for human readability (trialDataPath ends in .dict, change to .txt)
    trialDataTextPath = os.path.splitext(trialDataPath)[0] + ".txt"
    with open(trialDataTextPath, "w",encoding="UTF-8") as textFile:
        textFile.write(str(trialResultDict))

def getTrialDataPath(saveDirectoryPath, solverName, nSimulationsPerTree, nTrial):
    """
    Helper function that makes path to save trialDataDict at
    """

    solverDirectoryPath = os.path.join(saveDirectoryPath,solverName)
    nSimPath =  os.path.join(solverDirectoryPath,str(nSimulationsPerTree))
    nTrialPath =  os.path.join(nSimPath,str(nTrial)) #Make directory according to the rngKeyUSed

    #For now just pickle dump the full dictionary, consider more sophisticated saving later
    return os.path.join(nTrialPath,"trialData.dict")
