"""
File for regenerating the average data from the results
"""
import os
import sys
import pickle
from tqdm import tqdm
import yaml

from failurePy.load.yamlLoader import loadExperimentParams,loadExperimentParamsFromYaml
from failurePy.load.yamlLoaderUtilityMethods import getInputDict
from failurePy.utility.saving import processDataAverages, checkSaveDirectoryPathMakeIfNeeded, setUpDataOutput


def reprocessDataAverages(configFilePath,absolutePathAboveSavedData=None):
    """
    Reprocesses the specified directory by recomputing the averages
    """

    #Get experiment params and save path
    experimentParamsDict, saveDirectoryPath = loadExperimentParamsFromYaml(configFilePath) # pylint: disable=unbalanced-tuple-unpacking
    #print(experimentParamsDict["nExperimentSteps"])
    configSaveDirectoryAbsoluteFlag = bool(saveDirectoryPath[0] == "/")

    if configSaveDirectoryAbsoluteFlag:
        print(saveDirectoryPath)
        processDataAverages(saveDirectoryPath, experimentParamsDict,)
    else:
        #Override if needed
        #savePath = ""
        print(saveDirectoryPath)
        processDataAverages(os.path.join(absolutePathAboveSavedData,saveDirectoryPath), experimentParamsDict,)

def reprocessTrialDataDicts(absoluteSavedDataPath,dictReprocessF):
    """
    Looks for all trialData.dict files in this directory and its subdirectories, then passes them to the
    specified reprocessing method. The changed dictionary overwrites the original file.

    Parameters
    ----------
    absoluteSavedDataPath : string
        Absolute path to directory where we should look for and modify trailData.dict files.
    dictReprocessF : function
        Function to reprocess the trial data
    """

    pbar = tqdm(total=getNumberOfTrials(absoluteSavedDataPath))

    #Use os.walk to search through every file here for saved data dictionaries
    for path, dummyDirectories, files in os.walk(absoluteSavedDataPath):
        #Look at each file here for saved data dictionary
        for file in files:
            #Found a dictionary to re-compute the successFlag for
            if file == "trialData.dict":
                #Load dictionary
                trialDataPath = os.path.join(path,"trialData.dict")
                with open(trialDataPath, "rb") as trialDataFile:
                    trialResultsDict = pickle.load(trialDataFile)
                #Reprocess!
                reprocessedTrialDataDict = dictReprocessF(trialResultsDict)
                #Guard against returning None (since that will destroy all the data)
                if reprocessedTrialDataDict is None:
                    invalidTrialDataDict = "trialDataDict is None. Check that the updated data was returned"
                    raise ValueError(invalidTrialDataDict)
                #Save again
                with open(trialDataPath,'wb') as trialDataFile:
                    pickle.dump(reprocessedTrialDataDict,trialDataFile)
                #And update human readable copy
                trialDataTextPath = os.path.splitext(trialDataPath)[0]+'.txt'
                with open(trialDataTextPath, "w",encoding="UTF-8") as textFile:
                    textFile.write(str(reprocessedTrialDataDict))
                #Update progress
                pbar.update(1)
    pbar.close()

def runNewExperimentsWithOldData(absoluteSavedDataPath,reprocessOrRunNewExpF,newExperimentAbsoluteSavedDataPath,inputUpdateDict):
    """
    Looks for all trialData.dict files in this directory and its subdirectories, then passes them to the
    specified reprocessing method. The changed dictionary overwrites the original file.

    Parameters
    ----------
    absoluteSavedDataPath : string
        Absolute path to directory where we should look for and modify trailData.dict files.
    dictReprocessF : function
        Function to reprocess the trial data
    newExperimentAbsoluteSavedDataPath : string
        Absolute path to directory where we should save modified (or rerun) trailData.dict files.
    """


    dummyDict = {"mergeData":False,"clobber":False}
    try:
        checkSaveDirectoryPathMakeIfNeeded(newExperimentAbsoluteSavedDataPath,dummyDict)
    except FileExistsError:
        print(f"Data already exists at {newExperimentAbsoluteSavedDataPath}! Overwrite?")
        if not confirmDataDeletion(newExperimentAbsoluteSavedDataPath,data="ALL DATA"):
            print("Aborting")
            return -1
        dummyDict["clobber"] = True
        checkSaveDirectoryPathMakeIfNeeded(newExperimentAbsoluteSavedDataPath,dummyDict)

    #This config file is the original experiments config
    configFilePath = os.path.join(absoluteSavedDataPath, "config.yaml")
    #Load raw yaml input
    inputDict = getInputDict(configFilePath)

    #Make changes, and dump in new directory, won't be sorted, but will ensure the saved config is right
    inputDict.update(inputUpdateDict)
    yamlContent = yaml.dump(inputDict, sort_keys=False, default_flow_style = False, allow_unicode = True, encoding = None)
    newConfigFilePath = os.path.join(newExperimentAbsoluteSavedDataPath, "config.yaml")
    with open(newConfigFilePath, "w", encoding="utf-8") as configFile:
        configFile.write(yamlContent)
        #Overwrite if anything existed before, there shouldn't be
        configFile.truncate()
    #print(inputDict.keys())

    experimentParamsDict, dummySaveDirectoryRelativePath = loadExperimentParams(inputDict) # pylint: disable=unbalanced-tuple-unpacking
    setUpDataOutput(newExperimentAbsoluteSavedDataPath, experimentParamsDict, configFilePath)


    pbar = tqdm(total=getNumberOfTrials(absoluteSavedDataPath))

    processList = []

    #Limit CPU cores to numProcesses! (To share better on servers)
    cpuMask = range(100)
    os.sched_setaffinity(os.getpid(), cpuMask)
    #Use os.walk to search through every file here for saved data dictionaries
    for path, dummyDirectories, files in os.walk(absoluteSavedDataPath):
        #Look at each file here for saved data dictionary
        for file in files:
            #Found a dictionary to re-compute the successFlag for
            if file == "trialData.dict":
                trialDataPath = os.path.join(path,"trialData.dict")
                #Check if we have a detached process
                processReturn =reprocessOrRunNewExpF(trialDataPath,inputDict,newExperimentAbsoluteSavedDataPath)
                if processReturn is not None:
                    processReturn.start()
                    if len(processList) <= 99:
                        processList.append(processReturn)
                        print(processReturn.is_alive(),processReturn.exitcode)
                    else: #This will be the 100th running, since it is started already
                        appendAndWaitUntilAProcessEnds(processReturn,processList,pbar)

                    continue #Don't update the pbar, we'll do that when popping processes
                #Update progress
                pbar.update(1)

    #Join all processes!
    for process in processList:
        print(process.is_alive(),process.exitcode)
        process.join()
        pbar.update(1)

    pbar.close()

    return 0

def appendAndWaitUntilAProcessEnds(process,processList,pbar):
    """
    Method that checks for finished process and joins then pops them.
    Will not return until this is done
    """

    atLeastOneProcessEnded = False
    processList.append(process)

    #Loop until we are done
    while not atLeastOneProcessEnded:
        #Go from the back to make popping work better
        #We know there are 100 processes, as we ended up in here
        for iProcess in range(99,0,-1):
            #Check each for 10 ms.
            processList[iProcess].join(.01)
            #If joined, it has ended
            if processList[iProcess].exitcode is not None:
                processList.pop(iProcess)
                atLeastOneProcessEnded = True
                pbar.update(1)


def confirmDataDeletion(directory,data="data"):
    """
    Confirms with the user before starting process that could delete data.

    Parameters
    ----------
    directory : string
        Path where the data will be deleted from.
    data : string (default="data")
        The specific data that can be removed

    Returns
    -------
    confirmation : boolean
        True if the user confirms, False otherwise.
    """

    sys.stdout.write(f"WARNING: This script can PERMANENTLY delete {data} in {directory}. Proceed [y/N]?")
    choice = input().lower()
    if choice in {"yes","y","ye"}:
        return True
    return False

def getNumberOfTrials(absoluteSavedDataPath):
    """
    Finds the total number of trials in the saved data
    """

    firstSolver = getSubFolders(absoluteSavedDataPath)[0]
    numSolvers = getNumberOfFoldersInDirectory(absoluteSavedDataPath)

    firstSolverDirectoryPath = os.path.join(absoluteSavedDataPath,firstSolver)
    firstNSimsPerTreeExperiment = getSubFolders(firstSolverDirectoryPath)[0]
    numNSimsPerTreeExperiments = getNumberOfFoldersInDirectory(firstSolverDirectoryPath)

    firstNSimsPerTreeExperimentPath = os.path.join(firstSolverDirectoryPath,firstNSimsPerTreeExperiment)
    numTrialsPerPoint = getNumberOfFoldersInDirectory(firstNSimsPerTreeExperimentPath)

    return numSolvers*numNSimsPerTreeExperiments*numTrialsPerPoint

def getSubFolders(directory):
    """
    Returns the names of the folders in a directory
    """

    return next(os.walk(directory))[1]

def getNumberOfFoldersInDirectory(directory):
    """
    Returns the number of sub folders in a directory
    """
    return len(getSubFolders(directory))
#Fixing a too-large tuple so will have a long method.
def reformatCache(experimentDataDirAbsolutePath,dieAndStayDead): # pylint: disable=too-many-statements
    """
    Translates from original pickle cache to dictionary based cache. Shouldn't be needed anymore
    """
    #newSafetyCachePath = os.path.join(experimentDataDirAbsolutePath,"safetyDataCache.dict")
    oldSafetyCachePath = os.path.join(experimentDataDirAbsolutePath,"safetyTupleCache.pickle")
    #load
    with open(oldSafetyCachePath,'rb') as oldCache:
        safetyTuple = pickle.load(oldCache)
    empiricalSafetyValuesIdx = 0
    avgEmpSafetyIdx = 1
    trueSafetyValuesIdx = 2
    avgTrueSafetyIdx = 3
    nInfSafetyValuesIdx = 4
    avgNInfSafetyIdx = 5
    empiricalFalseNegativesIdx = 6
    empiricalFalsePositivesIdx = 7
    empiricalFalseNegativesTotalsIdx = 8
    empiricalFalsePositivesTotalsIdx = 9
    nInfFalsePositivesIdx = 10
    nInfFalseNegativesIdx = 11
    nInfFalseNegativesTotalsIdx = 12
    nInfFalsePositivesTotalsIdx = 13
    nSimulationsPerTreesIdx = 14
    lastBelievedCollisionFreeTStepsIdx = 15
    avgBelievedLastCollisionFreeTStepsIdx = 16
    lastTrueCollisionFreeTStepIdx = 17
    avgBelievedLastCollisionFreeTStepsIdx = 18
    beliefAlphaSafetyIdx = 19
    avgBeliefAlphaSafetyIdx = 20
    beliefAlphaFalsePositivesIdx = 21
    beliefAlphaFalseNegativesIdx = 22

    beliefAlphaFalseNegativesTotalsIdx = 23
    beliefAlphaFalsePositivesTotalsIdx = 24
    #Convert to dict
    safetyDataDict = {}
    safetyDataDict["empiricalSafetyValues"] = safetyTuple[empiricalSafetyValuesIdx]
    safetyDataDict["avgEmpSafety"] = safetyTuple[avgEmpSafetyIdx]

    safetyDataDict["trueSafetyValues"] = safetyTuple[trueSafetyValuesIdx]
    safetyDataDict["avgTrueSafety"] = safetyTuple[avgTrueSafetyIdx]

    safetyDataDict["nInfSafetyValues"] = safetyTuple[nInfSafetyValuesIdx]
    safetyDataDict["avgNInfSafety"] = safetyTuple[avgNInfSafetyIdx]

    safetyDataDict["empiricalFalseNegatives"] = safetyTuple[empiricalFalseNegativesIdx]
    safetyDataDict["empiricalFalsePositives"] = safetyTuple[empiricalFalsePositivesIdx]

    safetyDataDict["empiricalFalseNegativesTotals"] = safetyTuple[empiricalFalseNegativesTotalsIdx]
    safetyDataDict["empiricalFalsePositivesTotals"] = safetyTuple[empiricalFalsePositivesTotalsIdx]

    safetyDataDict["nInfFalseNegatives"] = safetyTuple[nInfFalsePositivesIdx]
    safetyDataDict["nInfFalsePositives"] = safetyTuple[nInfFalseNegativesIdx]

    safetyDataDict["nInfFalseNegativesTotals"] = safetyTuple[nInfFalseNegativesTotalsIdx]
    safetyDataDict["nInfFalsePositivesTotals"] = safetyTuple[nInfFalsePositivesTotalsIdx]

    safetyDataDict["lastBelievedCollisionFreeTSteps"] = safetyTuple[lastBelievedCollisionFreeTStepsIdx]
    safetyDataDict["avgBelievedLastCollisionFreeTSteps"] = safetyTuple[avgBelievedLastCollisionFreeTStepsIdx]

    safetyDataDict["lastTrueCollisionFreeTSteps"] = safetyTuple[lastTrueCollisionFreeTStepIdx]
    safetyDataDict["avgTrueLastCollisionFreeTSteps"] = safetyTuple[avgBelievedLastCollisionFreeTStepsIdx] #note wrong!

    safetyDataDict["beliefAlphaSafetyValues"] = safetyTuple[beliefAlphaSafetyIdx]
    safetyDataDict["avgBeliefAlphaSafety"] = safetyTuple[avgBeliefAlphaSafetyIdx]

    safetyDataDict["beliefAlphaFalsePositives"] = safetyTuple[beliefAlphaFalsePositivesIdx]
    safetyDataDict["beliefAlphaFalseNegatives"] = safetyTuple[beliefAlphaFalseNegativesIdx]

    safetyDataDict["beliefAlphaFalseNegativesTotals"] = safetyTuple[beliefAlphaFalseNegativesTotalsIdx]
    safetyDataDict["beliefAlphaFalsePositivesTotals"] = safetyTuple[beliefAlphaFalsePositivesTotalsIdx]

    safetyDataDict["nSimulationsPerTrees"] = safetyTuple[nSimulationsPerTreesIdx]
    safetyDataDict["numTimeSteps"] = len(safetyTuple[0][0,0,0])

    #To determine if this cache matches later
    safetyDataDict["dieAndStayDead"] = dieAndStayDead

    #Save this
    cacheData(experimentDataDirAbsolutePath,safetyDataDict)

def cacheData(experimentDataDirAbsolutePath,safetyDataDict,cacheName=None):
    """
    Saves safety data to save time re-loading it
    """
    safetyCachePath = getCachePath(experimentDataDirAbsolutePath,cacheName)

    with open(safetyCachePath,'wb') as safetyCacheFile:
        pickle.dump(safetyDataDict,safetyCacheFile)

def getCachePath(experimentDataDirAbsolutePath,cacheName=None):
    """
    Creates path for cache file
    """
    if cacheName is not None:
        return os.path.join(experimentDataDirAbsolutePath,f"{cacheName}.dict")
    return os.path.join(experimentDataDirAbsolutePath,"safetyDataCache.dict")


if __name__ == "__main__":
    #CONFIG_FILE_PATH = "../config/chebyshevSafetyLinearModelTest.yaml"
    #reprocessDataAverages(CONFIG_FILE_PATH,os.path.dirname(os.getcwd()))

    SAVED_DATA_DIRECTORY = "/mnt/i/jplFeastSavedData/SavedData/nonLinearSafetyTestTrueCrashScenario10s90Confidence1000LowSigma"
    reprocessDataAverages(os.path.join(SAVED_DATA_DIRECTORY,"config.yaml"),os.path.dirname(os.path.dirname(SAVED_DATA_DIRECTORY)))
