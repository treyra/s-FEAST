"""
Module containing various functions for post processing data, such as performing safety metric analysis
"""


import os

import pickle
from tqdm import tqdm
import numpy as onp
import matplotlib.pyplot as plt

import jax.random as jaxRandom

from failurePy.load.yamlLoaderUtilityMethods import getInputDict
from failurePy.load.safetyLoader import loadSafetyModulatedRewardComponents
from failurePy.load.systemConstructors import loadAndBuildSingleAgentSystem
from failurePy.load.yamlLoader import loadEstimatorAndBelief
from failurePy.rewards.safetyConstraint import makeProbabilisticAlphaSafetyFunctionEvaluation

# Data loader for reward plotting


def loadDataSummaryFromSaveDirectory(savedDataDirPath, experimentName,solverNames,cumulativeFlag=False,baselineExpName=None,altRewardFlag=True,): #Alt way to plot only some experimentIndexes=None):
    """
    Method to load saved data that is reused by several functions

    This is an alternate method from loadDataSummary in failurePy.visualization.visualization
    due data location being required parameter, both provide same returns.

    Parameters
    ----------
    savedDataDirPath : str
        Absolute path of the saved data.
    experimentName : str
        Name of the experiment (top level saved data directory)
    solverNames : list
        List of the names of each solver that was used
    cumulativeFlag : boolean (default=False)
        If true, plot the cumulative reward
    baselineExpName : str (default=None)
        If provided, loads in additional solver experiments that are all baselines (1 nSim per tree, always 1)
    altRewardFlag : str (default=False)
        If true, looks for alternative reward data instead of the default. Incompatible with cumulative sum currently. Does not check for existence first

    Returns
    -------
    nSimulationsPerTrees : array, shape(numNSimulationsPerTrees)
        The number of simulations performed before returning an action (when not running on time out mode).
        This parameter is an array, if longer then length 1, multiple trials are run, varying the number of simulations per tree.
    nTrialsPerPoint : int
        The number of repeated trials per configuration.
    avgRewards : array, shape(numSolvers,numNSimulationsPerTrees,numExperimentSteps+1)
        The average rewards for each solver and number of simulations per tree, for each time step (including initial time)
    avgSuccessRates : array, shape(numSolvers,numNSimulationsPerTrees)
        The average success rate for each solver and number of simulations per tree
    avgWallClockTimes : array, shape(numSolvers,numNSimulationsPerTrees)
        The average time for each solver and number of simulations per tree
    avgSteps : array, shape(numSolvers,numNSimulationsPerTrees)
        The average steps for each solver and number of simulations per tree
    sigmaRewards : array, shape(numSolvers,numNSimulationsPerTrees)
        The 1 sigma bounds for the rewards
    """

    #First we need to collect the data
    experimentDataDirPath = os.path.join(savedDataDirPath,experimentName)

    #Check if there is a baseline directory to load from too
    if baselineExpName is not None:
        baselineExperimentDataDirPath = os.path.join(savedDataDirPath,baselineExpName)
        baselineSolverNames = next(os.walk(baselineExperimentDataDirPath))[1]
        #Hack to sort baselines consistently (will format when plotting legend)
        if "greedy" in baselineSolverNames and "cbf" in baselineSolverNames and "scp" in baselineSolverNames:
            baselineSolverNames = ["greedy", "cbf", "scp"]


        numBaselines = len(baselineSolverNames)
        solverNames += baselineSolverNames #Updates solverNames by reference!
    else:
        numBaselines = 0

    #We take for granted that the number of simulations per trees is the same for each solver, as is the number of trials per point. If this is None, haven't found them yet
    nSimulationsPerTrees = None
    nTrialsPerPoint = None

    for iSolver, solverName in enumerate(solverNames):
        solverDirectoryPath = getSolverDirectoryPath(iSolver,solverNames,numBaselines,baselineExperimentDataDirPath,experimentDataDirPath,solverName)

        if nSimulationsPerTrees is None:
            nSimulationsPerTrees = getNSimulationsPerTrees(solverDirectoryPath)

        for jNSimsPerTreeExperiment, nSimulationsPerTree in enumerate(nSimulationsPerTrees):
            #Initialize average data arrays if we haven't yet
            if nTrialsPerPoint is None:
                nSimPath =  os.path.join(solverDirectoryPath,str(nSimulationsPerTree))
                nTrialsPerPoint = len(onp.array(next(os.walk(nSimPath))[1]))

                #Load first data dict
                experimentAverageDataDict = loadExperimentAverageDataDict(nSimPath)

                #Now initialize the average data now that we know the number of experiments
                avgRewards = onp.zeros((len(solverNames),len(nSimulationsPerTrees),len(experimentAverageDataDict["avgRewards"])))
                avgSuccessRates = onp.zeros((len(solverNames),len(nSimulationsPerTrees)))
                avgWallClockTimes = onp.zeros((len(solverNames),len(nSimulationsPerTrees)))
                avgSteps = onp.zeros((len(solverNames),len(nSimulationsPerTrees)))
                sigmaRewards = onp.zeros((len(solverNames),len(nSimulationsPerTrees),len(experimentAverageDataDict["avgRewards"])))
            #Baselines load a little different (SUPER HACKY)
            elif iSolver > 0 and jNSimsPerTreeExperiment == 0:
                nSimPath =  os.path.join(solverDirectoryPath,str(1)) #Always 1 for baselines
                experimentAverageDataDict = loadExperimentAverageDataDict(nSimPath)
            #Fill baselines with nans for other "num sims per tre" SUPER HACKY here, just trying to quickly plot.
            elif iSolver > 0:
                avgSuccessRates[iSolver,jNSimsPerTreeExperiment] += onp.nan
                avgWallClockTimes[iSolver,jNSimsPerTreeExperiment] += onp.nan
                avgSteps[iSolver,jNSimsPerTreeExperiment] += onp.nan
                avgRewards[iSolver,jNSimsPerTreeExperiment] += onp.nan
                sigmaRewards[iSolver,jNSimsPerTreeExperiment] += onp.nan
                continue #Don't load anything
            #Otherwise just load
            else:
                #Load data dict
                nSimPath =  os.path.join(solverDirectoryPath,str(nSimulationsPerTree))
                experimentAverageDataDict = loadExperimentAverageDataDict(nSimPath)


            #print(experimentAverageDataDict)

            #In either case, get data
            loadExperimentAverageData(experimentAverageDataDict,avgSuccessRates,avgWallClockTimes,avgSteps,avgRewards,sigmaRewards,cumulativeFlag,
                  altRewardFlag,iSolver,jNSimsPerTreeExperiment)

    return nSimulationsPerTrees, nTrialsPerPoint, avgRewards, avgSuccessRates, avgWallClockTimes, avgSteps, sigmaRewards

def getSolverDirectoryPath(iSolver,solverNames,numBaselines,baselineExperimentDataDirPath,experimentDataDirPath,solverName):
    """
    Helper method to generate solver path
    """
    if iSolver >= len(solverNames) - numBaselines:
        solverDirectoryPath = os.path.join(baselineExperimentDataDirPath,solverName)
    else:
        solverDirectoryPath = os.path.join(experimentDataDirPath,solverName)

    if not os.path.exists(solverDirectoryPath):
        raise FileNotFoundError(f"Directory {solverDirectoryPath} not found, check if the correct save directory and experiment are given")

    return solverDirectoryPath

def getNSimulationsPerTrees(solverDirectoryPath):
    """
    Helper method to load NSimulations per trees from folder names
    """
    #Get nSimulationsPerTrees using os.walk to read the directory names
    nSimulationsPerTrees = onp.array(next(os.walk(solverDirectoryPath))[1])
    nSimulationsPerTrees = nSimulationsPerTrees.astype(int)
    nSimulationsPerTrees = onp.sort(nSimulationsPerTrees)
    return nSimulationsPerTrees

def getTrialNumbers(nSimPath):
    """
    Helper method to load nTrialsPerPoint from the folders present
    """
    #Get  nTrialsPerPoint
    trialNumbers = onp.array(next(os.walk(nSimPath))[1])
    nTrialsPerPoint = len(trialNumbers)
    #Sorts the trial numbers so we have consistency (incase that matters)
    return onp.sort(trialNumbers),nTrialsPerPoint

def loadExperimentAverageDataDict(experimentPath):
    """
    Helper method to load average data from a given experiment path
    """

    experimentDataPath = os.path.join(experimentPath,"averageData.dict")
    with open(experimentDataPath,'rb') as experimentDataFile:
        experimentAverageDataDict = pickle.load(experimentDataFile)
    return experimentAverageDataDict

def loadTrialDataDict(trialPath):
    """
    Helper method to load trial data dict
    """
    trialDataPath = os.path.join(trialPath, "trialData.dict")
    with open(trialDataPath, "rb") as trialDataFile:
        trialDataDict = pickle.load(trialDataFile)
    return trialDataDict


def loadExperimentAverageData(experimentAverageDataDict,avgSuccessRates,avgWallClockTimes,avgSteps,avgRewards,sigmaRewards,cumulativeFlag,
                  altRewardFlag,iSolver,jNSimsPerTreeExperiment):
    """
    Helper method to load the data for a given trial, and make adjustments if needed
    """

    avgSuccessRates[iSolver,jNSimsPerTreeExperiment] = experimentAverageDataDict["avgSuccessRate"]
    avgWallClockTimes[iSolver,jNSimsPerTreeExperiment] = experimentAverageDataDict["avgWallClockTime"]
    avgSteps[iSolver,jNSimsPerTreeExperiment] = experimentAverageDataDict["avgSteps"]
    if cumulativeFlag:
        #If EKF diverged = failure (also need to make sure success set to 0)
        experimentAverageDataDict["cumulativeAvgRewards"][onp.isnan(experimentAverageDataDict["cumulativeAvgRewards"])] = 0

        avgRewards[iSolver,jNSimsPerTreeExperiment] = experimentAverageDataDict["cumulativeAvgRewards"]
        sigmaRewards[iSolver,jNSimsPerTreeExperiment] = onp.sqrt(experimentAverageDataDict["cumulativeVarRewards"])
    elif altRewardFlag:
        #If EKF diverged = failure (also need to make sure success set to 0)
        experimentAverageDataDict["avgAltRewards"][onp.isnan(experimentAverageDataDict["avgAltRewards"])] = 0

        avgRewards[iSolver,jNSimsPerTreeExperiment] = experimentAverageDataDict["avgAltRewards"]
        sigmaRewards[iSolver,jNSimsPerTreeExperiment] = onp.sqrt(experimentAverageDataDict["varAltRewards"])
    else:
        #If EKF diverged = failure (also need to make sure success set to 0)
        experimentAverageDataDict["avgRewards"][onp.isnan(experimentAverageDataDict["avgRewards"])] = 0

        avgRewards[iSolver,jNSimsPerTreeExperiment] = experimentAverageDataDict["avgRewards"]
        sigmaRewards[iSolver,jNSimsPerTreeExperiment] = onp.sqrt(experimentAverageDataDict["varRewards"])

def trialSuccessRatesAndFaultSpaceErrors(savedDataDirAbsolutePath, experimentName,solverNames,perTimeStep=False,faultDistances=False,stride=10, #pylint: disable=too-many-branches,too-many-statements
                                         clearCache=False): #Letting this be a do everything method, and conditionally create arrays pylint: disable=too-many-branches,too-many-statements,too-many-nested-blocks,possibly-used-before-assignment
    """
    Finds the success rates (correct diagnoses at the end) as well as the fault space distances for each trial
    """
    #First we need to collect the data
    experimentDataDirAbsolutePath = os.path.join(savedDataDirAbsolutePath,experimentName)

    #Check for caches
    if cacheDict := getFaultSpaceErrorCache(experimentDataDirAbsolutePath, perTimeStep, stride, clearCache):
        return cacheDict

    nSimulationsPerTrees = None
    nTrialsPerPoint = None

    for iSolver, solverName in enumerate(solverNames): #pylint: disable=too-many-nested-blocks
        solverDirectoryPath = getSolverDirectoryPath(iSolver,solverNames,numBaselines=0,baselineExperimentDataDirPath=None,experimentDataDirPath=experimentDataDirAbsolutePath,solverName=solverName)

        if nSimulationsPerTrees is None:
            nSimulationsPerTrees = getNSimulationsPerTrees(solverDirectoryPath)

        for jNSimsPerTreeExperiment, nSimulationsPerTree in enumerate(nSimulationsPerTrees):
            #Initialize average data arrays if we haven't yet
            if nTrialsPerPoint is None:
                nSimPath =  os.path.join(solverDirectoryPath,str(nSimulationsPerTree))
                trialNumbers, nTrialsPerPoint = getTrialNumbers(nSimPath)
                #Now initialize the data now that we know the number of experiments
                if perTimeStep:
                    kTrialPath = os.path.join(nSimPath, str(trialNumbers[0]))
                    trialDataDict = loadTrialDataDict(kTrialPath)
                    numExperimentTimeSteps = len(trialDataDict["beliefList"])
                    numTimeStepsToAnalyze = min(int(numExperimentTimeSteps/stride) + 1, numExperimentTimeSteps) # Only add one when stride !=1
                    diagnosisSuccess = onp.zeros((len(solverNames),len(nSimulationsPerTrees),nTrialsPerPoint,numTimeStepsToAnalyze))
                    diagnosisErrors = onp.zeros((len(solverNames),len(nSimulationsPerTrees),nTrialsPerPoint,numTimeStepsToAnalyze))
                    diagnosisConfidences = onp.zeros((len(solverNames),len(nSimulationsPerTrees),nTrialsPerPoint,numTimeStepsToAnalyze))
                    if faultDistances: #Kind of expensive to compute, lots of looping
                        avgFaultDistances = onp.zeros((len(solverNames),len(nSimulationsPerTrees),nTrialsPerPoint,numTimeStepsToAnalyze))
                        minFaultDistances = onp.zeros((len(solverNames),len(nSimulationsPerTrees),nTrialsPerPoint,numTimeStepsToAnalyze))
                    else:
                        avgFaultDistances = None
                        minFaultDistances = None

                else:
                    diagnosisSuccess = onp.zeros((len(solverNames),len(nSimulationsPerTrees),nTrialsPerPoint))
                    diagnosisErrors = onp.zeros((len(solverNames),len(nSimulationsPerTrees),nTrialsPerPoint))
                    diagnosisConfidences = onp.zeros((len(solverNames),len(nSimulationsPerTrees),nTrialsPerPoint))
                    if faultDistances: #Kind of expensive to compute, lots of looping
                        avgFaultDistances = onp.zeros((len(solverNames),len(nSimulationsPerTrees),nTrialsPerPoint))
                        minFaultDistances = onp.zeros((len(solverNames),len(nSimulationsPerTrees),nTrialsPerPoint))
                    else:
                        avgFaultDistances = None
                        minFaultDistances = None
            #Load data for each trial
            for kTrial,trialNumber in tqdm(enumerate(trialNumbers)):
                kTrialPath = os.path.join(nSimPath, str(trialNumber))
                trialDataDict = loadTrialDataDict(kTrialPath)
                if perTimeStep:
                    failureParticles = trialDataDict["possibleFailures"] #Initial possible failures
                    trueFault = trialDataDict["failureStateList"][0] #Assumed unchanging
                    jAnalysisStep = 0
                    for iTimeStep in range(numExperimentTimeSteps):
                        #Check if we re-weighted
                        if len(trialDataDict["beliefList"][iTimeStep]) == 3:
                            failureParticles = trialDataDict["beliefList"][iTimeStep][2]
                        #Check if we should analyze this point
                        if iTimeStep % stride != 0:
                            continue
                        #Fix zero looking right because all equal weight
                        if iTimeStep == 0:
                            #Kinda hacky, but should work without loss of generality
                            # Just set diagnosis to second particles, as this is randomly selected, so should average out
                            faultWeights = onp.array([0,1])
                        else:
                            faultWeights = trialDataDict["beliefList"][iTimeStep][0] #Fault weights are 0th position in the belief tuple

                        #Get the metrics
                        diagnosis = failureParticles[onp.argmax(faultWeights)]
                        #We assume the failure state doesn't change, but taking last to be general incase this changes
                        diagnosisErrors[iSolver,jNSimsPerTreeExperiment,kTrial,jAnalysisStep] = onp.linalg.norm(trueFault - diagnosis) #pylint: disable=possibly-used-before-assignment
                        diagnosisSuccess[iSolver,jNSimsPerTreeExperiment,kTrial,jAnalysisStep] = (diagnosisErrors[iSolver,jNSimsPerTreeExperiment,kTrial,jAnalysisStep]  < .01) #pylint: disable=possibly-used-before-assignment
                        diagnosisConfidences[iSolver,jNSimsPerTreeExperiment,kTrial,jAnalysisStep] = onp.max(faultWeights) #pylint: disable=possibly-used-before-assignment
                        if faultDistances: #Kind of expensive to compute, lots of looping
                            avgFaultDistance = 0
                            minFaultDistance = onp.inf
                            for faultParticle in failureParticles:
                                faultDistance = onp.linalg.norm(faultParticle-trueFault)
                                avgFaultDistance += faultDistance
                                if 0 < faultDistance < minFaultDistance:
                                    minFaultDistance = faultDistance
                            #Note when one of the particle is the true particle, this will be lower than when none are
                            avgFaultDistances[iSolver,jNSimsPerTreeExperiment,kTrial,jAnalysisStep] = avgFaultDistance/(len(failureParticles)) #pylint: disable=possibly-used-before-assignment
                            minFaultDistances[iSolver,jNSimsPerTreeExperiment,kTrial,jAnalysisStep] = minFaultDistance #pylint: disable=possibly-used-before-assignment
                        jAnalysisStep += 1

                else:
                    #If we re-sample the failure particles, these are included in the belief tuple as the 2 index, so check for these,
                    # otherwise use possible failures, the initial failure particles
                    #Loop backwards!
                    for iTimeStep in range(len(trialDataDict["beliefList"])):
                        if len(trialDataDict["beliefList"][-iTimeStep-1]) == 3:
                            failureParticles = trialDataDict["beliefList"][-iTimeStep-1][2]
                            break
                    #In python else is only entered if we DON'T break
                    else:
                        failureParticles = trialDataDict["possibleFailures"]

                    #While converging to the first fault indicates correct diagnosis when the fault is in the failure state, since we have access to the fault directly, will use general case
                    #Will consider a distance of <.01 in fault space as converged
                    #The diagnosis is the most likely fault in the final belief
                    faultWeights = trialDataDict["beliefList"][-1][0] #Fault weights are 0th position in the belief tuple
                    diagnosis = failureParticles[onp.argmax(faultWeights)]
                    #We assume the failure state doesn't change, but taking last to be general incase this changes
                    diagnosisErrors[iSolver,jNSimsPerTreeExperiment,kTrial] = onp.linalg.norm((trueFault := trialDataDict["failureStateList"][-1]) - diagnosis)
                    diagnosisSuccess[iSolver,jNSimsPerTreeExperiment,kTrial] = (diagnosisErrors[iSolver,jNSimsPerTreeExperiment,kTrial]  < .01)
                    diagnosisConfidences[iSolver,jNSimsPerTreeExperiment,kTrial] = onp.max(faultWeights)
                    if faultDistances: #Kind of expensive to compute, lots of looping
                        avgFaultDistance = 0
                        minFaultDistance = onp.inf
                        for faultParticle in failureParticles:
                            faultDistance = onp.linalg.norm(faultParticle-trueFault)
                            avgFaultDistance += faultDistance
                            if 0 < faultDistance < minFaultDistance:
                                minFaultDistance = faultDistance
                        #Note when one of the particle is the true particle, this will be lower than when none are
                        avgFaultDistances[iSolver,jNSimsPerTreeExperiment,kTrial] = avgFaultDistance/(len(failureParticles))
                        minFaultDistances[iSolver,jNSimsPerTreeExperiment,kTrial] = minFaultDistance

    #Cache data so we only need to do this once
    faultDataDict = makeFaultDataDict(diagnosisSuccess,diagnosisErrors,diagnosisConfidences,avgFaultDistances, minFaultDistances,stride)
    if perTimeStep:
        cacheData(experimentDataDirAbsolutePath, faultDataDict, cacheName="faultSpaceErrorPerTimeStepDataCache.dict")
    else:
        cacheData(experimentDataDirAbsolutePath, faultDataDict, cacheName="faultSpaceFinalErrorDataCache.dict")

    return faultDataDict

def makeFaultDataDict(diagnosisSuccess,diagnosisErrors,diagnosisConfidences,avgFaultDistances, minFaultDistances,stride):
    """
    Makes a dictionary of the computed data for caching and to return
    """
    return {"diagnosisSuccess": diagnosisSuccess,
            "diagnosisErrors": diagnosisErrors,
            "diagnosisConfidences": diagnosisConfidences,
            "avgFaultDistances": avgFaultDistances,
            "minFaultDistances": minFaultDistances,
            "stride": stride} #only matters on per time step data

def getFaultSpaceErrorCache(experimentDataDirAbsolutePath, perTimeStep, stride, clearCachedData=False): # pylint: disable=too-many-nested-blocks
    """
    Checks for already existing safety data cache
    """
    if not clearCachedData: # pylint: disable=too-many-nested-blocks
        try:
            if perTimeStep:
                faultSpaceErrorCachePath = getCachePath(experimentDataDirAbsolutePath, "faultSpaceErrorPerTimeStepDataCache.dict")
            else:
                faultSpaceErrorCachePath = getCachePath(experimentDataDirAbsolutePath, "faultSpaceFinalErrorDataCache.dict")
            with open(faultSpaceErrorCachePath, "rb") as safetyCacheFile:
                cacheDict = pickle.load(safetyCacheFile)
            #Check if we are consistent on per time step or not
            # Check consistency in time scale (only applies if we are per time step)
            if perTimeStep and cacheDict["stride"] != stride:
                #Okay if we can get subset of date with this stride.
                # This is the case if the current stride is larger than the cached one and is divided evenly by the caches one
                if stride % cacheDict["stride"] == 0:
                    #In this case, get the sub set
                    cacheStride = int(stride / cacheDict["stride"])
                    for key in cacheDict.keys():
                        if key == "stride" or cacheDict[key] is None:
                            continue
                        cacheDict[key] =  cacheDict[key][:,:,:,::cacheStride]
                else:
                    mismatchedCache = (f"Cache at {faultSpaceErrorCachePath} has an inconsistent stride ({cacheDict['stride']})" +
                                    f" than the provided stride {stride}. Use a compatible stride or set clearCachedData to True")
                    raise ValueError(mismatchedCache)
        except FileNotFoundError:
            cacheDict = None
    else:
        cacheDict = None
    return cacheDict

def plotComparisonMetricsAvgsAlongTime(savedDataDirPath, experimentNameRoot,solverNames,dataAnalysisF,metricStr,stride=10,clearCache=False):
    """
    Function that plots several experiments against each other that all share the same experimentNameRoot.
    Takes in one of these functions as the analysis function, and plots one of their average metrics along experiment time

    Assumes same size time step (and should have same time length for best comparison)
    """
    print("Loading Experiments")
    #First we need to find all matching experiments
    savedExperiments = next(os.walk(savedDataDirPath))[1] #Get all experiment folder strings
    comparisonExperiments = []
    print("Looking for matches")
    for savedExperiment in tqdm(savedExperiments):
        if savedExperiment.startswith(experimentNameRoot):
            comparisonExperiments.append(savedExperiment)

    if len(comparisonExperiments) == 0:
        noExperimentsFound = f"No experiments found matching {experimentNameRoot} in {savedDataDirPath}."
        raise FileNotFoundError(noExperimentsFound)
    print("Getting comparison data")
    #Get data
    comparisonData = []
    maxNumTimeSteps = 0
    experimentLabels = []
    for experimentName in tqdm(comparisonExperiments):
        dataAnalysisFReturnDict = dataAnalysisF(savedDataDirPath,experimentName,solverNames,perTimeStep=True,stride=stride,clearCache=clearCache)
        if not metricStr in dataAnalysisFReturnDict.keys():
            selectedMetricIdxToLarge = f"The selected metricStr {metricStr} is not in the metrics provided by {dataAnalysisF} ({dataAnalysisFReturnDict.keys()})"
            raise ValueError(selectedMetricIdxToLarge)
        comparisonData.append(dataAnalysisFReturnDict[metricStr])

        if len(comparisonData[-1]) > 1 or len(comparisonData[-1][0]) >1:
            tooManyVariables = f"Currently can only compare with one solver and one nSimsPerTree. Received {len(comparisonData[-1])} and {len(comparisonData[-1][0])}"
            raise ValueError(tooManyVariables)
        comparisonData[-1] = comparisonData[-1][0,0]

        maxNumTimeSteps = max(maxNumTimeSteps, len(comparisonData[-1][0]))
        experimentLabels.append(experimentName[len(experimentNameRoot):])


    #Finally plot
    dummyFig, ax = plt.subplots(nrows=1,ncols=1,figsize=(15,10))
    timeSteps = onp.arange(maxNumTimeSteps) * stride
    legendHandles = []
    for iExperiment, data in enumerate(comparisonData):
        handle, = ax.plot(timeSteps,onp.average(data,axis=0),label=experimentLabels[iExperiment])
        legendHandles.append(handle)
    ax.legend(handles=legendHandles)

    plt.show()

def plotMetricsBySimLevelAvgsAlongTime(savedDataDirPath, experimentName,solverNames,dataAnalysisF,metricStr,stride=10,clearCache=False):
    """
    Function that plots the different planning levels as a function of time.
    Takes in one of these functions as the analysis function, and plots one of their average metrics along experiment time

    Assumes same size time step (and should have same time length for best comparison)
    """


    #Get data
    numTimeSteps = 0
    dataAnalysisFReturnDict = dataAnalysisF(savedDataDirPath,experimentName,solverNames,perTimeStep=True,stride=stride,clearCache=clearCache)
    if not metricStr in dataAnalysisFReturnDict.keys():
        selectedMetricIdxToLarge = f"The selected metricStr {metricStr} is not in the metrics provided by {dataAnalysisF} ({dataAnalysisFReturnDict.keys()})"
        raise ValueError(selectedMetricIdxToLarge)
    comparisonData = dataAnalysisFReturnDict[metricStr]

    if len(comparisonData) > 1:
        tooManyVariables = f"Currently can only simulation levels with one solver. Received {len(comparisonData)}"
        raise ValueError(tooManyVariables)
    comparisonData = comparisonData[0]

    numTimeSteps = len(comparisonData[0,0])


    solverDirectoryPath = getSolverDirectoryPath(0,solverNames,numBaselines=0,baselineExperimentDataDirPath=None,
                                                 experimentDataDirPath=os.path.join(savedDataDirPath,experimentName),
                                                 solverName=solverNames[0])

    nSimulationsPerTrees = getNSimulationsPerTrees(solverDirectoryPath)

    #Finally plot
    dummyFig, ax = plt.subplots(nrows=1,ncols=1,figsize=(15,10))
    timeSteps = onp.arange(numTimeSteps) * stride
    legendHandles = []
    for iNumSimulationsPerTree in tqdm(range(len(comparisonData))):
        handle, = ax.plot(timeSteps,onp.average(comparisonData[iNumSimulationsPerTree],axis=0),label=f"N = {nSimulationsPerTrees[iNumSimulationsPerTree]}")
        legendHandles.append(handle)
    ax.legend(handles=legendHandles)

    plt.show()


# Compute safety metrics across experiment results.
# We will leverage the fact that the only way to get a reward = 0 is if the belief was determined to be unsafe


# First need to load data (This is a pretty manual function so disabling pylint)
def loadAndComputeSafetyData(experimentDataDirAbsolutePath, solverNamesList, dieAndStayDead=True,
                             sampleBeliefFlag=False, cacheName="safetyDataCache", clearCachedData=False, recomputeLastCollisionFree=False,requestedTSteps=None):
    """
    Returns safety metrics for plotting
    """
    # Need to load safety constraint evaluation to evaluate if belief is safe or not.
    # Actually we don't! We can leverage reward = 0 only when the belief was evaluated not to be alpha-safe (could look into whether it is overestimated... sure!)

    # Check for cached data. Then we don't need to recompute (might need to recompute missing data)
    cacheDict = getSafetyCache(experimentDataDirAbsolutePath, cacheName, dieAndStayDead, clearCachedData)

    if sampleBeliefFlag:
        safetyFunctionEvaluationF, safetyFunctionF, alphaSafetyFunctionEvaluationF = loadExactAndSampleSafetyFunctionF(experimentDataDirAbsolutePath, returnAlphaSafetyFunctionEvaluationF=True)
    else:
        safetyFunctionEvaluationF, safetyFunctionF = loadExactAndSampleSafetyFunctionF(experimentDataDirAbsolutePath) # pylint: disable=unbalanced-tuple-unpacking
        alphaSafetyFunctionEvaluationF = None
    initialized = False
    # Returning as a dict now
    safetyDataDict = {}
    #Variables which will be replaced when we first loop through
    nSimulationsPerTrees = None
    partialCache = None
    nTrialsPerPoint = None
    trialNumbers = None

    rngKey = jaxRandom.PRNGKey(0)

    # Loop through solvers
    for iSolver, solverName in enumerate(solverNamesList):
        solverDirectoryPath = os.path.join(experimentDataDirAbsolutePath, solverName)
        if not os.path.exists(solverDirectoryPath):
            raise FileNotFoundError("Directory not found, check if the correct save directory and experiment are given")

        if nSimulationsPerTrees is None:
            safetyDataDict["nSimulationsPerTrees"] = getNSimulationsPerTrees(solverDirectoryPath)

            # Check for consistency one more time, then assume cache matches
            partialCache, returnFlag, sampleBeliefFlag = checkAndApplyCache(cacheDict,safetyDataDict,experimentDataDirAbsolutePath,
                                                                            recomputeLastCollisionFree,sampleBeliefFlag)
            #If cache already has all the data, just return it
            if returnFlag:
                return returnRequestedDataDictRange(cacheDict, requestedTSteps)

        for jNSimsPerTreeExperiment, nSimulationsPerTree in enumerate(safetyDataDict["nSimulationsPerTrees"]):
            print(f"Computing safety for N = {nSimulationsPerTree}")
            nSimPath = os.path.join(solverDirectoryPath, str(nSimulationsPerTree))
            # Initialize average data arrays if we haven't yet, since determining safetyDataDict["nSimulationsPerTrees"] programmatically
            if nTrialsPerPoint is None:
                # Get  nTrialsPerPoint
                trialNumbers, nTrialsPerPoint = getTrialNumbers(nSimPath)

            for kTrial, trialNumber in tqdm(enumerate(trialNumbers)):
                kTrialPath = os.path.join(nSimPath, str(trialNumber))
                trialDataDict = loadTrialDataDict(kTrialPath)

                # Set up outputs if needed
                if not initialized:
                    numTimeSteps = len(trialDataDict["beliefList"])
                    initializeSafetyDataDict(safetyDataDict,partialCache,numTimeSteps,len(solverNamesList),nTrialsPerPoint,recomputeLastCollisionFree)
                    initialized = True

                # In any case, get data
                physicalStateList = trialDataDict["physicalStateList"]
                beliefList = trialDataDict["beliefList"]
                rewards = trialDataDict["rewards"]
                # If EKF diverged = failure (also need to make sure success set to 0)
                rewards[onp.isnan(rewards)] = 0
                # Old name, not as informative
                # lastBelievedCollisionFreeTStep = trialDataDict["lastCollisionFreeTStep"]
                lastBelievedCollisionFreeTStep = trialDataDict["lastBelievedCollisionFreeTStep"]

                #Loop over each time step (all saved in safetyDataDict, so no returns)
                safetyAtEachTimeStep(safetyDataDict,iSolver,jNSimsPerTreeExperiment,kTrial,numTimeSteps,partialCache,
                         dieAndStayDead,physicalStateList,beliefList,rewards,safetyFunctionF,safetyFunctionEvaluationF,
                         alphaSafetyFunctionEvaluationF,sampleBeliefFlag,rngKey)

                #Compute averages and trial wide data
                safetyOverTrial(safetyDataDict,iSolver,jNSimsPerTreeExperiment,kTrial,partialCache,sampleBeliefFlag,recomputeLastCollisionFree,lastBelievedCollisionFreeTStep)

            # Average collision free-ness
            if not partialCache or recomputeLastCollisionFree:
                safetyDataDict["avgBelievedLastCollisionFreeTSteps"][iSolver, jNSimsPerTreeExperiment] = onp.average(
                        safetyDataDict["lastBelievedCollisionFreeTSteps"][iSolver, jNSimsPerTreeExperiment, :])
                safetyDataDict["avgTrueLastCollisionFreeTSteps"][iSolver, jNSimsPerTreeExperiment] = onp.average(
                        safetyDataDict["lastTrueCollisionFreeTSteps"][iSolver, jNSimsPerTreeExperiment, :])

    # Useful to know
    safetyDataDict["numTimeSteps"] = numTimeSteps

    # To determine if this cache matches later
    safetyDataDict["dieAndStayDead"] = dieAndStayDead
    cacheData(experimentDataDirAbsolutePath, safetyDataDict, cacheName)
    return returnRequestedDataDictRange(safetyDataDict, requestedTSteps)

def returnRequestedDataDictRange(safetyDataDict, requestedTSteps):
    """
    Helper function that only returns some time steps if a shorter range is requested
    """


    #Allow for only considering some of the data
    if requestedTSteps is not None:
        timeStepsInData = len(safetyDataDict["empiricalSafetyValues"][0,0,0])
        if timeStepsInData > requestedTSteps:
            for key in safetyDataDict.keys():
                if len(onp.shape(safetyDataDict[key]))>3 and not "totals" in key.lower(): #Totals sums over all time steps
                    safetyDataDict[key] = safetyDataDict[key][:,:,:,:requestedTSteps+1]
                elif key == "numTimeSteps":
                    safetyDataDict[key] = requestedTSteps+1

        elif timeStepsInData <  requestedTSteps:
            tooManyTimeStepsRequested = f"Saved data only has {timeStepsInData} time steps. {requestedTSteps} requested"
            raise ValueError(tooManyTimeStepsRequested)

    return safetyDataDict

def checkAndApplyCache(cacheDict,safetyDataDict,experimentDataDirAbsolutePath,recomputeLastCollisionFree,sampleBeliefFlag):
    """
    Helper method that inspects what type of cache we have
    """

    if cacheDict is not None:
        if not onp.all(safetyDataDict["nSimulationsPerTrees"] == cacheDict["nSimulationsPerTrees"]):
            mismatchedCache = (f"Safety cache under {experimentDataDirAbsolutePath} has different value of nSimulationsPerTrees ({safetyDataDict['nSimulationsPerTrees']}) " +
                "than the computed value, indicating a change in the data. Use a different cache name or set clearCachedData to True")
            raise ValueError(mismatchedCache)

        # Check if we need to do anything, if true assume all values present, so no need to compute anything
        # (if sampleBeliefFlag = False, then we don't care about missing info)
        returnFlag = not recomputeLastCollisionFree and ("beliefAlphaSafetyValues" in cacheDict or not sampleBeliefFlag)


        if "beliefAlphaSafetyValues" in cacheDict or not sampleBeliefFlag:
            partialCache = True
            sampleBeliefFlag = False  # Only need to recompute last collision free times
            safetyDataDict = cacheDict
        else:
            # Only need to do the belief sampling (may make this more complicated later?)
            partialCache = True
            safetyDataDict = cacheDict
    else:
        partialCache = False
        returnFlag = False

    return partialCache, returnFlag, sampleBeliefFlag

def safetyAtEachTimeStep(safetyDataDict,iSolver,jNSimsPerTreeExperiment,kTrial,numTimeSteps,partialCache,
                         dieAndStayDead,physicalStateList,beliefList,rewards,safetyFunctionF,safetyFunctionEvaluationF,
                         alphaSafetyFunctionEvaluationF,sampleBeliefFlag,rngKey):
    """
    Helper method that performs the safety analysis at each time step
    """
    # Safety at each point (May need to vectorize this...) Note, t=0 is initialization so should always be safe
    for mTimeStep in range(numTimeSteps):
        if not partialCache:
            # Check if we crashed before
            if dieAndStayDead and mTimeStep > 0:
                estimatedAlive = safetyDataDict["empiricalSafetyValues"][iSolver, jNSimsPerTreeExperiment, kTrial, mTimeStep - 1]
                trulyAlive = safetyDataDict["trueSafetyValues"][iSolver, jNSimsPerTreeExperiment, kTrial, mTimeStep - 1]
                nInfAlive = safetyDataDict["nInfSafetyValues"][iSolver, jNSimsPerTreeExperiment, kTrial, mTimeStep - 1]

                # Check for false life.
                if estimatedAlive != trulyAlive:
                    if estimatedAlive > trulyAlive:
                        safetyDataDict["empiricalFalsePositives"][iSolver,jNSimsPerTreeExperiment,kTrial,mTimeStep,1,] = 1
                    else:
                        safetyDataDict["empiricalFalseNegatives"][iSolver,jNSimsPerTreeExperiment,kTrial,mTimeStep,1,] = 1
                if nInfAlive != trulyAlive:
                    if nInfAlive > trulyAlive:
                        safetyDataDict["nInfFalsePositives"][iSolver,jNSimsPerTreeExperiment,kTrial, mTimeStep,1,] = 1
                    else:
                        safetyDataDict["nInfFalseNegatives"][iSolver,jNSimsPerTreeExperiment,kTrial,mTimeStep,1,] = 1

            else:
                estimatedAlive = 1
                trulyAlive = 1
                nInfAlive = 1

            # Safe unless reward = 0  ! (And not already dead)
            estimatedSafety = onp.sign(rewards[mTimeStep])
            safetyDataDict["empiricalSafetyValues"][iSolver, jNSimsPerTreeExperiment, kTrial, mTimeStep] = estimatedSafety * estimatedAlive
            # Can check true safety (negative return is safe, positive or zero is unsafe!)
            trueSafetyFUnctionSign = onp.sign(safetyFunctionF(physicalStateList[mTimeStep]))
            trueSafety = (0.5 - 0.5 * trueSafetyFUnctionSign) * onp.abs(trueSafetyFUnctionSign)
            safetyDataDict["trueSafetyValues"][iSolver, jNSimsPerTreeExperiment, kTrial, mTimeStep] = trueSafety * trulyAlive
            # Can approximate nInf by letting it go to high value (will try with N=1000 here)
            rngKey, rngSubKey = jaxRandom.split(rngKey)
            nInfSafety = safetyFunctionEvaluationF(beliefList[mTimeStep], rngSubKey)
            safetyDataDict["nInfSafetyValues"][iSolver, jNSimsPerTreeExperiment, kTrial, mTimeStep] = nInfSafety * nInfAlive

            # Check for false +/-
            falsePositiveNegativeCheck(safetyDataDict,estimatedSafety,nInfSafety,trueSafety,iSolver,jNSimsPerTreeExperiment,kTrial, mTimeStep)

        # Always do this part, partial cache or no.
        # Assuming this might be expensive, seems to be a 1/2 slow down (but 2000 samples may be excessive) This and nInf are probably main costs.
        if sampleBeliefFlag:
            sampleBeliefAlphaSafety(safetyDataDict,dieAndStayDead,trulyAlive,alphaSafetyFunctionEvaluationF,beliefList,trueSafety,iSolver,jNSimsPerTreeExperiment,kTrial, mTimeStep, rngKey)

def falsePositiveNegativeCheck(safetyDataDict,estimatedSafety,nInfSafety,trueSafety,iSolver,jNSimsPerTreeExperiment,kTrial,mTimeStep):
    """
    Helper function to compute false +/-
    """
    if estimatedSafety != trueSafety:
        if estimatedSafety > trueSafety:
            safetyDataDict["empiricalFalsePositives"][iSolver,jNSimsPerTreeExperiment,kTrial, mTimeStep,0,] = 1
        else:
            safetyDataDict["empiricalFalseNegatives"][iSolver,jNSimsPerTreeExperiment, kTrial,mTimeStep, 0,] = 1
    if nInfSafety != trueSafety:
        if nInfSafety > trueSafety:
            safetyDataDict["nInfFalsePositives"][iSolver,jNSimsPerTreeExperiment,kTrial,mTimeStep,0,] = 1
        else:
            safetyDataDict["nInfFalseNegatives"][iSolver,jNSimsPerTreeExperiment,kTrial,mTimeStep,0,] = 1

def sampleBeliefAlphaSafety(safetyDataDict,dieAndStayDead,trulyAlive,alphaSafetyFunctionEvaluationF,beliefList,trueSafety,iSolver,jNSimsPerTreeExperiment,kTrial, mTimeStep, rngKey):
    """
    Helper method that computes the safety estimate we get by just sampling from the belief a specified number of times (1000 by hard coding)
    """
    if dieAndStayDead and mTimeStep > 0:
        alphaAlive = safetyDataDict["beliefAlphaSafetyValues"][iSolver, jNSimsPerTreeExperiment, kTrial, mTimeStep - 1]
        # False life?
        if alphaAlive != trulyAlive:
            if alphaAlive > trulyAlive:
                safetyDataDict["beliefAlphaFalsePositives"][iSolver,jNSimsPerTreeExperiment,kTrial,mTimeStep,1,] = 1
            else:
                safetyDataDict["beliefAlphaFalseNegatives"][iSolver,jNSimsPerTreeExperiment,kTrial,mTimeStep,1,] = 1
    else:
        alphaAlive = 1

    beliefAlphaSafety = alphaSafetyFunctionEvaluationF( beliefList[mTimeStep], rngKey)
    # print(trueSafety,beliefAlphaSafety,alphaAlive)
    safetyDataDict["beliefAlphaSafetyValues"][iSolver, jNSimsPerTreeExperiment, kTrial, mTimeStep] = beliefAlphaSafety * alphaAlive
    # false +/-
    if beliefAlphaSafety != trueSafety:
        if beliefAlphaSafety > trueSafety:
            safetyDataDict["beliefAlphaFalsePositives"][iSolver,jNSimsPerTreeExperiment,kTrial,mTimeStep,0,] = 1
        else:
            safetyDataDict["beliefAlphaFalseNegatives"][iSolver,jNSimsPerTreeExperiment,kTrial,mTimeStep,0,] = 1


def safetyOverTrial(safetyDataDict,iSolver,jNSimsPerTreeExperiment,kTrial,partialCache,sampleBeliefFlag,recomputeLastCollisionFree,lastBelievedCollisionFreeTStep):
    """
    Helper method that performs the safety analysis for the trial as a whole
    """
    # Totals and Averages
    if not partialCache:
        safetyDataDict["empiricalFalsePositivesTotals"][ iSolver, jNSimsPerTreeExperiment, kTrial] = onp.sum(
                safetyDataDict["empiricalFalsePositives"][iSolver, jNSimsPerTreeExperiment, kTrial, :, :],axis=0,)
        safetyDataDict["empiricalFalseNegativesTotals"][iSolver, jNSimsPerTreeExperiment, kTrial] = onp.sum(
                safetyDataDict["empiricalFalseNegatives"][iSolver, jNSimsPerTreeExperiment, kTrial, :, :],axis=0,)

        safetyDataDict["nInfFalseNegativesTotals"][iSolver, jNSimsPerTreeExperiment, kTrial] = onp.sum(
                safetyDataDict["nInfFalseNegatives"][iSolver, jNSimsPerTreeExperiment, kTrial, :, :],axis=0,)
        safetyDataDict["nInfFalsePositivesTotals"][iSolver, jNSimsPerTreeExperiment, kTrial] = onp.sum(
                safetyDataDict["nInfFalsePositives"][iSolver, jNSimsPerTreeExperiment, kTrial, :, :],axis=0,)

        safetyDataDict["avgEmpSafety"][iSolver, jNSimsPerTreeExperiment, kTrial] = onp.average(
                safetyDataDict["empiricalSafetyValues"][iSolver, jNSimsPerTreeExperiment, kTrial, :])
        safetyDataDict["avgTrueSafety"][iSolver, jNSimsPerTreeExperiment, kTrial] = onp.average(
                safetyDataDict["trueSafetyValues"][iSolver, jNSimsPerTreeExperiment, kTrial, :])
        safetyDataDict["avgNInfSafety"][iSolver, jNSimsPerTreeExperiment, kTrial] = onp.average(
                safetyDataDict["nInfSafetyValues"][iSolver, jNSimsPerTreeExperiment, kTrial, :])

    # Collision free
    if not partialCache or recomputeLastCollisionFree:
        # print("believed",lastBelievedCollisionFreeTStep)
        safetyDataDict["lastBelievedCollisionFreeTSteps"][iSolver, jNSimsPerTreeExperiment, kTrial] = lastBelievedCollisionFreeTStep
        lastTrueCollisionFreeTStep = -1
        for trueSafetyValue in safetyDataDict["trueSafetyValues"][iSolver, jNSimsPerTreeExperiment, kTrial]:
            if trueSafetyValue == 0:
                # print("true",lastTrueCollisionFreeTStep)
                break
            lastTrueCollisionFreeTStep += 1
        safetyDataDict["lastTrueCollisionFreeTSteps"][iSolver, jNSimsPerTreeExperiment, kTrial] = lastTrueCollisionFreeTStep

    if sampleBeliefFlag:
        safetyDataDict["beliefAlphaFalseNegativesTotals"][iSolver, jNSimsPerTreeExperiment, kTrial] = onp.sum(
                safetyDataDict["beliefAlphaFalseNegatives"][ iSolver, jNSimsPerTreeExperiment, kTrial, :, :],axis=0,)
        safetyDataDict["beliefAlphaFalsePositivesTotals"][iSolver, jNSimsPerTreeExperiment, kTrial] = onp.sum(
                safetyDataDict["beliefAlphaFalsePositives"][iSolver, jNSimsPerTreeExperiment, kTrial, :, :],axis=0, )

        safetyDataDict["avgBeliefAlphaSafety"][iSolver, jNSimsPerTreeExperiment, kTrial] = onp.average(
                safetyDataDict["beliefAlphaSafetyValues"][iSolver, jNSimsPerTreeExperiment, kTrial, :])

def initializeSafetyDataDict(safetyDataDict,partialCache,numTimeSteps,numSolvers,nTrialsPerPoint,recomputeLastCollisionFree):
    """
    Helper method to create all the output arrays
    """
    if not partialCache:

        numNSimulationsPerTree = len(safetyDataDict["nSimulationsPerTrees"])

        # Now initialize the data now that we know the number of experiments
        safetyDataDict["empiricalSafetyValues"] = onp.zeros((numSolvers,numNSimulationsPerTree,nTrialsPerPoint,numTimeSteps,))
        safetyDataDict["avgEmpSafety"] = onp.zeros((numSolvers, numNSimulationsPerTree, nTrialsPerPoint))
        safetyDataDict["trueSafetyValues"] = onp.zeros((numSolvers, numNSimulationsPerTree,nTrialsPerPoint,numTimeSteps,))
        safetyDataDict["avgTrueSafety"] = onp.zeros((numSolvers, numNSimulationsPerTree, nTrialsPerPoint))

        safetyDataDict["nInfSafetyValues"] = onp.zeros((numSolvers,numNSimulationsPerTree,nTrialsPerPoint,numTimeSteps,))
        safetyDataDict["avgNInfSafety"] = onp.zeros((numSolvers, numNSimulationsPerTree, nTrialsPerPoint))

        safetyDataDict["empiricalFalseNegatives"] = onp.zeros((numSolvers,numNSimulationsPerTree,nTrialsPerPoint,numTimeSteps,2,))
        safetyDataDict["empiricalFalsePositives"] = onp.zeros((numSolvers,numNSimulationsPerTree,nTrialsPerPoint,numTimeSteps,2,))

        safetyDataDict["empiricalFalseNegativesTotals"] = onp.zeros((numSolvers, numNSimulationsPerTree, nTrialsPerPoint, 2))
        safetyDataDict["empiricalFalsePositivesTotals"] = onp.zeros((numSolvers, numNSimulationsPerTree, nTrialsPerPoint, 2))

        safetyDataDict["nInfFalseNegatives"] = onp.zeros((numSolvers,numNSimulationsPerTree,nTrialsPerPoint,numTimeSteps,2,))
        safetyDataDict["nInfFalsePositives"] = onp.zeros((numSolvers,numNSimulationsPerTree,nTrialsPerPoint,numTimeSteps,2,))

        safetyDataDict["nInfFalseNegativesTotals"] = onp.zeros((numSolvers, numNSimulationsPerTree, nTrialsPerPoint, 2))
        safetyDataDict["nInfFalsePositivesTotals"] = onp.zeros((numSolvers, numNSimulationsPerTree, nTrialsPerPoint, 2))
    # Calculate for new run or specified recompute
    if not partialCache or recomputeLastCollisionFree:
        safetyDataDict["lastBelievedCollisionFreeTSteps"] = onp.zeros((numSolvers, numNSimulationsPerTree, nTrialsPerPoint))
        safetyDataDict["avgBelievedLastCollisionFreeTSteps"] = onp.zeros((numSolvers, numNSimulationsPerTree))

        safetyDataDict["lastTrueCollisionFreeTSteps"] = onp.zeros((numSolvers, numNSimulationsPerTree, nTrialsPerPoint))
        safetyDataDict["avgTrueLastCollisionFreeTSteps"] = onp.zeros((numSolvers, numNSimulationsPerTree))
    # Do this always partial cached or no cache

    safetyDataDict["beliefAlphaSafetyValues"] = onp.zeros((numSolvers,numNSimulationsPerTree,nTrialsPerPoint,numTimeSteps,))
    safetyDataDict["avgBeliefAlphaSafety"] = onp.zeros((numSolvers, numNSimulationsPerTree, nTrialsPerPoint))

    safetyDataDict["beliefAlphaFalsePositives"] = onp.zeros((numSolvers,numNSimulationsPerTree,nTrialsPerPoint,numTimeSteps,2,))
    safetyDataDict["beliefAlphaFalseNegatives"] = onp.zeros((numSolvers,numNSimulationsPerTree,nTrialsPerPoint,numTimeSteps,2,))

    safetyDataDict["beliefAlphaFalseNegativesTotals"] = onp.zeros((numSolvers, numNSimulationsPerTree, nTrialsPerPoint, 2))
    safetyDataDict["beliefAlphaFalsePositivesTotals"] = onp.zeros((numSolvers, numNSimulationsPerTree, nTrialsPerPoint, 2))

def getSafetyCache(experimentDataDirAbsolutePath, cacheName="safetyDataCache", dieAndStayDead=True, clearCachedData=False):
    """
    Checks for already existing safety data cache
    """
    if not clearCachedData:
        try:
            safetyCachePath = getCachePath(experimentDataDirAbsolutePath, cacheName)
            with open(safetyCachePath, "rb") as safetyCacheFile:
                cacheDict = pickle.load(safetyCacheFile)
            # Check consistency
            if cacheDict["dieAndStayDead"] != dieAndStayDead:
                mismatchedCache = (f"Cache at {safetyCachePath} has different value of dieAndStayDead ({not dieAndStayDead})" +
                                   " than the provided value. Use a different cache name or set clearCachedData to True")
                raise ValueError(mismatchedCache)
        except FileNotFoundError:
            cacheDict = None
    else:
        cacheDict = None
    return cacheDict

def cacheData(experimentDataDirAbsolutePath, dataDict, cacheName=None):
    """
    Saves analyzed data to save time re-loading it
    """
    cachePath = getCachePath(experimentDataDirAbsolutePath, cacheName)

    with open(cachePath, "wb") as cacheFile:
        pickle.dump(dataDict, cacheFile)


def getCachePath(experimentDataDirAbsolutePath, cacheName="safetyDataCache"):
    """
    Creates path for cache file. Default is safety cache
    """

    return os.path.join(experimentDataDirAbsolutePath, f"{cacheName}.dict")



def loadExactAndSampleSafetyFunctionF(experimentDataDirAbsolutePath, returnAlphaSafetyFunctionEvaluationF=False):
    """
    Returns the sample-based safety function evaluation method.
    """

    configFilePath = os.path.join(experimentDataDirAbsolutePath, "config.yaml")

    inputDict = getInputDict(configFilePath)

    # Could be more efficient getting components here, but seems okay for now
    silent = True
    dummySystemF,dummySystemParametersTuple,dummyPhysicalStateJacobianF,dummyDim,linear,dummyDt,dummySigmaW,dummySigmaV, dummyNumAct = loadAndBuildSingleAgentSystem(
                inputDict, providedFailure=None,generalFaultFlag=False, silent=silent) #Even if we have general fault system, don't need it for safety analysis, as systemParametersTuple isn't affected

    #unneeded for any estimator or reward right now
    # To get the number of states, this is length of covariance matrix, which is the -2 element of the systemParametersTuple
    #covarianceQ = systemParametersTuple[-2]
    #numState = len(covarianceQ)

    # Load estimator and belief initialization function
    dummyEstimatorF,dummyPhysicalStateSubEstimatorF,physicalStateSubEstimatorSampleF, dummyBeliefInitializationF = loadEstimatorAndBelief(
            inputDict, linear, generalFaultFlag=False, silent=silent)#Even if we have general fault system, don't need it for safety analysis,
                                                                    # as physicalStateSubEstimatorSampleF isn't affected

    # Going to approximate infinite samples by an order of magnitude higher, to justify claim that we are close.
    safetyFunctionEvaluationF, safetyFunctionF = loadSafetyModulatedRewardComponents(
            inputDict, physicalStateSubEstimatorSampleF, numSamples=1000)

    if returnAlphaSafetyFunctionEvaluationF:
        alphaSafetyFunctionEvaluationF = makeProbabilisticAlphaSafetyFunctionEvaluation(safetyFunctionF,physicalStateSubEstimatorSampleF,numSamples=2000,alpha=0.95,)
        return (safetyFunctionEvaluationF,safetyFunctionF,alphaSafetyFunctionEvaluationF,)

    return safetyFunctionEvaluationF, safetyFunctionF


def getActions(experimentDataDirPath, solverName, lastTimeStep=None):
    """
    Method for getting the actions a solver takes throughout the course of the experiment for each trial

    Useful for comparing how aggressive a solver is
    """

    solverDirectoryPath = os.path.join(experimentDataDirPath, solverName)
    nSimulationsPerTrees = getNSimulationsPerTrees(solverDirectoryPath)
    nSimPath = os.path.join(solverDirectoryPath, str(nSimulationsPerTrees[0]))
    # Initialize average data arrays
    # Get  nTrialsPerPoint
    trialNumbers, nTrialsPerPoint = getTrialNumbers(nSimPath)
    kTrialPath = os.path.join(nSimPath, str(trialNumbers[0]))
    trialDataDict = loadTrialDataDict(kTrialPath)
    # Get number of experiment steps
    numTimeSteps = len(trialDataDict["actionList"])
    # Want to be able to cap to shorter length if specified
    if lastTimeStep is not None and lastTimeStep < numTimeSteps - 1:
        numTimeSteps = lastTimeStep + 1

    actions = onp.zeros((len(nSimulationsPerTrees),nTrialsPerPoint,numTimeSteps,len(trialDataDict["actionList"][0]),))
    numActiveActuators = onp.zeros((len(nSimulationsPerTrees), nTrialsPerPoint, numTimeSteps))

    # Now loop through and process
    for jNSimsPerTreeExperiment, nSimulationsPerTree in enumerate(nSimulationsPerTrees):
        print(nSimulationsPerTree)
        # if nSimulationsPerTree < 200:
        #    continue
        nSimPath = os.path.join(solverDirectoryPath, str(nSimulationsPerTree))

        for kTrial, trialNumber in tqdm(enumerate(trialNumbers)):
            kTrialPath = os.path.join(nSimPath, str(trialNumber))
            trialDataDict = loadTrialDataDict(kTrialPath)

            actions[jNSimsPerTreeExperiment, kTrial] = onp.array(trialDataDict["actionList"][:numTimeSteps])
            numActiveActuators[jNSimsPerTreeExperiment, kTrial] = onp.sum(actions[jNSimsPerTreeExperiment, kTrial], axis=1)

    return actions, numActiveActuators
