"""
Main file for setting up, running, and logging data for experiments
"""
import os
import time
#import pickle
import argparse
#Multiprocessing components to make things run faster
import multiprocessing as mp
from queue import Empty #Can't get this directly from mp.Queue
import numpy as onp #Following Jax conventions
import multiprocess.context as ctx
import jax.numpy as jnp

import jax
from jax import random as jaxRandom
from tqdm import tqdm
#failurePy imports, these need to be installed by pip to work``
from failurePy.load.yamlLoader import loadExperimentParamsFromYaml, loadExperimentParams
from failurePy.utility.saving import checkSaveDirectoryPathMakeIfNeeded, processDataAverages, checkIfDataExists,makeTrialResultDict,setUpDataOutput,saveTrialResult,getTrialDataPath #,saveMetaData
from failurePy.visualization.renderPlanarVisWrapper import visualizeFirstTrajectory
from failurePy.solvers.randomPolicy import solveForNextAction as randomPolicyF
from failurePy.solvers.randomPolicy import distributedSolveForNextAction as distributedRandomPolicyF
from failurePy.utility.tqdmMultiprocessing import initializeTqdm
from failurePy.utility.pipelineHelperMethods import getExperimentParameters, generateAllPossibleFailures, checkFailureIsValid, diagnoseFailure
from failurePy.estimators.generalFaultSampling import binaryToDegradationFaults #Check fault initialization method
from failurePy.utility.computeAlternateReward import computeSquareSumFailureBeliefRewardAlternativeThroughout


def main(configFilePath, saveData=True, visualizeFirstTrajectoryFlag=True):
    """
    Main method of the code sets up, runs, logs and cleans up experiment

    Parameters
    ----------
    configFilePath : String
        Relative path to the config file for the experiment
    saveData : boolean (default=True)
        Over-rides data saving behavior when False to not write out any data, avoiding potential overwrite when testing or profiling
    visualizeFirstTrajectory  : boolean (default=True)
        Whether to visualize the first experiment result or not
    """


    #First load in the experiment parameters. These determine the save path, so this is returned as well
    experimentParamsDict, saveDirectoryPath = loadExperimentParamsFromYaml(configFilePath) #We won't use extra data here pylint: disable=unbalanced-tuple-unpacking

    #Make absolute save directory if it isn't already
    if not saveDirectoryPath[0] == "/":
        saveDirectoryPath = os.path.join(os.getcwd(),saveDirectoryPath)

    #Check if save directory is clean
    checkSaveDirectoryPathMakeIfNeeded(saveDirectoryPath,experimentParamsDict)

    #Setup data output
    setUpDataOutput(saveDirectoryPath, experimentParamsDict, configFilePath)

    #Check for multiprocessing. If so, we will recurse into sub processes
    if experimentParamsDict["multiprocessingFlag"]:
        runMultiprocessingSubExperiments(experimentParamsDict["virtualConfigDictList"],saveDirectoryPath,
                                         nTrialsPerPoint=experimentParamsDict["nTrialsPerPoint"],numExperiments=len(experimentParamsDict["nSimulationsPerTreeList"]))
    else:
        runExperimentsAndLog(experimentParamsDict,saveDirectoryPath,saveData)

    #Log data (potentially do something with a return code at a later date?)
    processDataAverages(saveDirectoryPath,experimentParamsDict)

    computeSquareSumFailureBeliefRewardAlternativeThroughout(saveDirectoryPath,force=True)

    if visualizeFirstTrajectoryFlag:
        #Display, only if asked (never multiprocessing)
        visualize(experimentParamsDict,saveDirectoryPath)

def runMultiprocessingSubExperiments(virtualConfigDictList,saveDirectoryPath,nTrialsPerPoint,numExperiments):
    """
    Splits the experiments to run into many sub experiments for parallelization using multiprocessing.
    Each sub experiment has a virtual configuration dictionary, that can be passed to yamlLoader to get the needed parameters,
    it has already been configured for the number of cores desired.

    virtualConfigDictList : list
        Each member of the list contains the configuration parameters for their sub-experiment. This is the same information
        we would get from the config file, but multiprocessing is set to False, and the rngKeysOffset and nTrialsPerPoint are
        adjusted to split the load across cores
    saveDirectoryPath : str
        String of the absolute path to the saveDirectory for the results of the experiment. Auto determined from the experiment parameters, unless overwritten.
        Now using pickle to save the data in it's original dictionary format, rather than creating directories for each data type saved
    totalTrials : int
        Total number of trials that we are doing across all processes
    """

    #Get number of cores available, always leave specified number free (in multiprocessing flag!)
    numProcesses = len(virtualConfigDictList)

    #Limit CPU cores to numProcesses! (To share better on servers)
    cpuMask = range(numProcesses)
    os.sched_setaffinity(os.getpid(), cpuMask)

    #Make progress bar and queue
    tqdmQueue = mp.Queue() #Infinite Queue
    tqdmProcess = mp.Process(target=multiprocessingTqdmUpdater,args=(tqdmQueue,nTrialsPerPoint,numExperiments),daemon=True) #Incase we ctrl+c (should still kill them but can't hurt)
    tqdmProcess.start()

    #And multiprocess!
    processList = []
    for iProcess in range(numProcesses):
        #Make progress bar (only first process gets a real one, rest are None) (daemon incase we ctrl+c (should still kill them but can't hurt))
        process = mp.Process(target=multiprocessingMain,args=(virtualConfigDictList[iProcess],saveDirectoryPath,tqdmQueue),daemon=True)
        process.start()
        processList.append(process)

    for process in processList:
        process.join()

    #No need to track progress
    tqdmProcess.join()

def multiprocessingTqdmUpdater(tqdmQueue,nTrialsPerPoint,numExperiments):
    """
    Helper function that updates the tqdm progress bar
    """

    progressBar = initializeTqdm(nTrialsPerPoint)
    experiment = 0
    trial = 0
    numExtraUpdates = 0

    while experiment < numExperiments:
        #Listening loop
        while trial < nTrialsPerPoint:
            #Get all progress counts that are available, then the exception breaks us out
            numUpdates = 0
            try:
                while True:
                    numUpdates += tqdmQueue.get_nowait()
            except Empty:
                #This is our indication that we have all the updates, so update now!
                trial += numUpdates
                #Check if any processes went on to the next experiment already (note this is approximate)
                if trial > nTrialsPerPoint:
                    numExtraUpdates = trial - nTrialsPerPoint
                    numUpdates = numUpdates-numExtraUpdates

                progressBar.update(numUpdates)
            #Sleep a little to avoid wasting too much effort listening
            time.sleep(.1) #Don't need updates more than every 10th of a second

        #Make next manual bar for next experiment
        progressBar  = initializeTqdm(nTrialsPerPoint)
        #Add any extra updates
        progressBar.update(numExtraUpdates)
        trial = numExtraUpdates
        experiment += 1

def multiprocessingMain(virtualInputDict,saveDirectoryPath,tqdmQueue=None, disableTqdm=True):
    """
    Helper function that loads the experimentParamsDict and runs all of the experiments and trials.

    Parameters
    ----------
    virtualInputDict : dict
        Contains the configuration parameters for this sub-experiment. This is the same information we would
        get from the config file, but multiprocessing is set to False, and the rngKeysOffset and nTrialsPerPoint are adjusted
        to split the load across cores
    saveDirectoryPath : str
        String of the absolute path to the saveDirectory for the results of the experiment. Auto determined from the experiment parameters, unless overwritten.
        Now using pickle to save the data in it's original dictionary format, rather than creating directories for each data type saved
    tqdmQueue : multiprocessing queue (default=None)
        Queue object to send progress updates through, should be provided unless disableTqdm is set to False
    disableTqdm : boolean (default=True)
        Replaces tqdm with custom tqdmQueue object if the tqdmQueue is provided
    """
    #Get the experiments parameters. We won't use the relative save directory path, as we will use the one for the full experiment
    experimentParamsDict, dummyRelativeSaveDirectoryPath = loadExperimentParams(virtualInputDict,silent=True) #We won't use extra data here pylint: disable=unbalanced-tuple-unpacking
    #And run! We always save data
    runExperimentsAndLog(experimentParamsDict,saveDirectoryPath,saveData=True, #Always save data
                         disableTqdm=disableTqdm, tqdmQueue=tqdmQueue)


def runExperimentsAndLog(experimentParamsDict,saveDirectoryPath,saveData=True,disableTqdm=False, tqdmQueue=None,):
    """
    Method that contains the main experiment loop.
    To avoid memory leaks, we log the results of each trial as they are produced

    Parameters
    ----------
    experimentParamsDict : dict
        Dictionary containing all the relevant experiment parameters.
        The contents should be as follows:
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
            nMaxFailureParticles : int
                Maximum number of failure particles to consider at once. If larger than the number of possible unique failures, all possibilities are considered
            providedFailure : array, shape(numAct+numSen) (default=None)
                Provided failure (if any) to have each trial use
            systemF : function
                Function reference of the system to call to run experiment
            systemParametersTuple : tuple
                Tuple of system parameters needed. See the model being used for details. (ie, linearModel)
            solverFList : list
                List of solver functions to try
            solverParametersTuplesList : list
                List of tuples of solver parameters. Included action list, failure scenarios
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
                Offset to the initial PRNG used in generating the initial failure states and randomness in the trials.
                This is added to the trial number to allow for different trials to be preformed
            multiprocessingFlag : int
                Wether to use multi-processing (if number is set other than 0) or not (if False/0)
            saveTreeFlag : boolean
                Whether to save the tree or not (it can be quite large, so if not visualizing, it is best to set this to false)
            numWarmStart : int (default=0)
                Checks if we should run the solver a few times to compile first, and if so how many. Only does so on first trial. Currently only implemented for non-multiprocessing
            clobber : boolean
                Wether to overwrite existing data or not
            plottingBounds : array, shape(2,2) (default=None)
                Bounds of the plotting axis
            resolution : int (default=200)
                How high of a resolution the safe zone should be drawn in when showing the safety function.
            virtualConfigDictList : list
                List of input dictionaries for each subExperiment when multiprocessing
            networkFlag : bool
                Whether we are in a distributed network or not
            generalFaultDict : dict
                If we are using a general fault model this is a dictionary with the following values. Otherwise it is None
                failureParticleResampleF : function
                    Function that resamples the particles when needed
                failureParticleResampleCheckF : function
                    Function that determines if resampling is needed
                failureParticleInitializationF : function
                    Function that creates the initial failure particles
            filterDivergenceHandlingMethod : string
                How to handle if the filter diverges mid trial. None if it should not be.
    saveDirectoryPath : str
        String of the absolute path to the saveDirectory for the results of the experiment. Auto determined from the experiment parameters, unless overwritten.
        Now using pickle to save the data in it's original dictionary format, rather than creating directories for each data type saved
    saveData : boolean (default=True)
        Over-rides data saving behavior when False to not write out any data, avoiding potential overwrite when testing or profiling
    disableTqdm : boolean (default=False)
        When true, turns off tqdm bar for a manual bar. tqdmQueue should be provided
    tqdmQueue : queue (default=None)
        Queue to pass updates to the manual progress bar
    """
    #Get basic parameters of the system
    numState, numAct, numSen, numNFailures, numWarmStart, numAgents = getExperimentParameters(experimentParamsDict)

    #Loop over solvers
    for iSolver in range(len(experimentParamsDict["solverFList"])):

        #Loop over nSimulationsPerTrees
        for nSimulationsPerTree in experimentParamsDict["nSimulationsPerTreeList"]:

            #Loop over trials
            for jTrial in tqdm(range(experimentParamsDict["nTrialsPerPoint"]),disable=disableTqdm):
                #Only warm start first trial
                warmStart = 0 if jTrial != 0 else numWarmStart
                #Eventually might allow us to have data already existing
                if experimentParamsDict["mergeData"] and checkIfDataExists(getTrialDataPath(saveDirectoryPath, experimentParamsDict["solverNamesList"][iSolver], nSimulationsPerTree, jTrial)):
                    raise NotImplementedError("Merging is not implemented at this time")
                    #continue #Don't overwrite
                initializeRunAndSaveTrial(experimentParamsDict,numState, numAct, numSen,numNFailures,saveDirectoryPath,nSimulationsPerTree,
                                            iSolver,jTrial+experimentParamsDict["rngKeysOffset"],saveData,numWarmStart=warmStart,numAgents=numAgents)
                #Manual multiprocessing bar
                if disableTqdm:
                    tqdmQueue.put_nowait(1)

def initializeRunAndSaveTrial(experimentParamsDict,numState, numAct, numSen,numNFailures,saveDirectoryPath,nSimulationsPerTree,iSolver,rngSeed,
                              saveData=True,numWarmStart=0,numAgents=1):
    """
    Helper method that performs the initialize, run and saving for each trial, to enable parallelization

    Parameters
    ----------
    experimentParamsDict : dict
        Dictionary containing all the relevant experiment parameters.
    numState : int
        Number of states
    numAct : int
        Number of actuators
    numSen : int
        Number of sensors
    numNFailures : array, shape(nMaxComponentFailures+1)
        Number of failure combinations for each level of failed components
    saveDirectoryPath : str
        String of the absolute path to the saveDirectory for the results of the experiment. Auto determined from the experiment parameters, unless overwritten.
    nSimulationsPerTree : int
        Number of max simulations per tree for the solver to search
    iSolver : int
        Index of the solver being run
    rngSeed : int
        Index of the trial being run (plus offset)
    saveData : boolean (default=True)
        Over-rides data saving behavior when False to not write out any data, avoiding potential overwrite when testing or profiling
    numWarmStart : int (default=0)
        Checks if we should run the solver a few times to compile first, and if so how many. Only does so on first trial. Currently only implemented for non-multiprocessing
    numAgents : int (default=1)
        How many agents are present (distinguishes single from multi-agent)
    """

    #Set random policy whenever nSimulationsPerTree = 0
    if nSimulationsPerTree == 0:
        if numAgents == 1:
            solverF = randomPolicyF
        else:
            solverF = distributedRandomPolicyF
    else:
        solverF = experimentParamsDict["solverFList"][iSolver]

    #Make initial rngKey and split
    rngKey = jaxRandom.PRNGKey(rngSeed)
    rngKey,rngSubKey = jaxRandom.split(rngKey)
    initializationDict, initialFailureParticles = initializeTrial(numState, numAct, numSen,
                                                        experimentParamsDict["providedFailure"],numNFailures,
                                                        experimentParamsDict["nMaxFailureParticles"],experimentParamsDict["beliefInitializationF"],
                                                        experimentParamsDict["rewardF"],experimentParamsDict["generalFaultDict"],rngSubKey,
                                                        initialState=experimentParamsDict["initialState"],numAgents=numAgents)
    #print("Trial initialized")
    #Run trial
    if numAgents == 1:
        runTrialF = runSingleAgentTrial
    else:
        runTrialF = runNetworkTrial
    trialResultDict = runTrialF(initializationDict,nSimulationsPerTree,experimentParamsDict["nExperimentSteps"],
                            experimentParamsDict["systemF"],experimentParamsDict["systemParametersTuple"],
                            solverF,experimentParamsDict["solverParametersTuplesList"][iSolver],#Get the list of solver params corresponding to this solver
                            experimentParamsDict["estimatorF"],experimentParamsDict["physicalStateSubEstimatorF"],experimentParamsDict["physicalStateJacobianF"],
                            experimentParamsDict["physicalStateSubEstimatorSampleF"],experimentParamsDict["rewardF"],initialFailureParticles,experimentParamsDict["generalFaultDict"],
                            experimentParamsDict["diagnosisThreshold"],experimentParamsDict["filterDivergenceHandlingMethod"], experimentParamsDict["saveTreeFlag"],
                            rngKey, numWarmStart=numWarmStart) #pass in PRNG key for this trial
                            #Future TODO: make computeSuccessAtEnd configurable!
    #print("Trial run")
    #Save this result directly (if turned on)
    if saveData:
        saveTrialResult(saveDirectoryPath, experimentParamsDict["solverNamesList"][iSolver], nSimulationsPerTree, rngSeed, trialResultDict)

def runSingleAgentTrial(initializationDict,nSimulationsPerTree,nExperimentSteps,systemF,systemParametersTuple,solverF,solverParametersTuple,estimatorF,
             physicalStateSubEstimatorF,physicalStateJacobianF,physicalStateSubEstimatorSampleF,rewardF,initialFailureParticles, generalFaultDict, diagnosisThreshold,filterDivergenceHandlingMethod,
             saveTreeFlag, rngKey, computeSuccessAtEnd=False,computeSafetyAtEnd=True, numWarmStart=0): #These last two parameters currently hardcoded TODO. # pylint: disable=too-many-branches
    """
    Method that runs one trial of a given system and solver, from given initialization

    Parameters
    ----------
    initializationDict : dict
        Initial physical state, failure state, beliefTuple, reward
    nSimulationsPerTree : int
        Number of max simulations per tree for the solver to search
    nExperimentSteps : int
        How many time steps are in the experiment
    systemF : function
        Function reference of the system to call to run experiment
    systemParametersTuple : tuple
        Tuple of system parameters needed. See the model being used for details. (ie, linearModel)
    solverF : function
        Solver function to select next action with
    estimatorF : function
        Estimator function to update the beliefs with. Takes batch of filters
    physicalStateSubEstimatorF : function
        Physical state estimator for use with the marginal filter, if any
    physicalStateJacobianF : function
        Jacobian of the model for use in estimating.
    physicalStateSubEstimatorSampleF : function
        Samples from the belief corresponding to this estimator
    rewardF : function
        Reward function to evaluate the beliefs with
    solverParametersTuple : tuple
        List of solver parameters needed. Contents are:
            availableActions : array, shape(maxNumActions,numAct)
                Array of actions that can be taken. First action is always null action
            discretization : float
                Discretization level or scheme
            maxSimulationTime : float
                Max simulation time (can be infinite). NOTE: Currently implemented by breaking loop after EXCEEDING time, NOT a hard cap
            explorationParameter : float
                Weighting on exploration vs. exploitation
            nMaxDepth : int
                Max depth of the tree
            discountFactor : float
                Discount on future rewards, should be in range [0,1]
    initialFailureParticles : array, shape(nMaxFailureParticles,numAct+numSen)
        List of (initial) possible failures to be considered by the solver. If not exhaustive, these are a random subset. Belief failure weights are are over these possibilities
        These may change if the estimator allows for re-sampling the fault particles
    generalFaultDict : dict
        If we are using a general fault model this is a dictionary with the following values. Otherwise it is None
        failureParticleResampleF : function
            Function that resamples the particles when needed
        failureParticleResampleCheckF : function
            Function that determines if resampling is needed
        failureParticleInitializationF : function
            Function that creates the initial failure particles
    diagnosisThreshold : float
        Level of the reward to consider high enough to return an answer.
        Overload: If set to a value above 1, use diagnosisThreshold-1 for evaluating filter divergences (if appropriate)
    filterDivergenceHandlingMethod : string
        How to handle if the filter diverges mid trial. None if it should not be.
    saveTreeFlag : boolean
        Whether to save the tree or not (it can be quite large, so if not visualizing, it is best to set this to false)
    rngKey : JAX PRNG key
        Key for the PRNG, to be used to create pseudorandomness
    computeSuccessAtEnd : boolean
        Checks if we should assign nan or not if ends without converging (ie, for safety where we don't want to break early)
    computeSafetyAtEnd : boolean
        Checks if we should evaluate safety of the trial at the end (to avoid needing to do it later), supersedes computeSuccessAtEnd, which may be removed, as it also checks this.
    numWarmStart : int (default=0)
        Checks if we should run the solver a few times to compile first, and if so how many. Only does so on first trial. Currently only implemented for non-multiprocessing

    Returns
    -------
    trialResultsDict : dict
        dict containing:
            physicalStateList : list
                List of the (realized) physical states of the system
            failureStateList : list
                List of the (unchanging) true failure state
            beliefList : list
                List of the beliefs at each time step (time steps determined by nExperimentSteps and dt, which is currently set in the system model)
            rewardList : list
                List of the rewards at each time step
            actionList : list
                List of the actions taken at each time step
            treeList: list
                List of the tree data at each time step. Each element is a tuple with the nodeList and the valuesRewardsVisitsArray for the tree
            initialFailureParticles : array, shape(nMaxFailureParticles,numAct+numSen)
                List of possible failures to be considered by the solver. If not exhaustive, these are a random subset. Belief failure weights are are over these possibilities

    """
    physicalStateList,failureStateList,beliefList,actionList,observationList,treeList,rewards = setUpTrialOutputs(initializationDict,nExperimentSteps,saveTreeFlag)

    #Warm start by running the solver a few times to compile everything
    if numWarmStart:
        print("Warm start")
        for dummyIndex in range(numWarmStart):
            #Still use rng keys so burn ins are different
            rngKey, rngSubKey = jaxRandom.split(rngKey)
            solverF(beliefList[-1],solverParametersTuple,initialFailureParticles,systemF,systemParametersTuple,rewardF,estimatorF,
                        physicalStateSubEstimatorF,physicalStateJacobianF,physicalStateSubEstimatorSampleF,nSimulationsPerTree,rngSubKey)
        print("Warm start done")

    #Get time for logging later
    wctStartTime = time.time()
    success = 0 #Will be set to 1 if returns the correct failure
    currentFailureParticles = initialFailureParticles #Updated by general fault model

    #for timeStep in tqdm(range(nExperimentSteps)): #Used to eyeball wall clock time
    for timeStep in range(nExperimentSteps):
        #Reaches here
        #Compute action to take
        rngKey, rngSubKey = jaxRandom.split(rngKey) #Keys are consumed on use, so keep splitting rngKey first to avoid consumption
        nextAction,tree = solverF(beliefList[-1],solverParametersTuple,currentFailureParticles,systemF,systemParametersTuple,rewardF,estimatorF,
                                  physicalStateSubEstimatorF,physicalStateJacobianF,physicalStateSubEstimatorSampleF,nSimulationsPerTree,rngSubKey)

        #Simulate system forward
        rngKey, rngSubKey = jaxRandom.split(rngKey) #Keys are consumed on use, so keep splitting rngKey first to avoid consumption
        nextPhysicalState,nextFailureState,nextObservation = systemF(physicalStateList[-1],failureStateList[-1],nextAction,rngSubKey,systemParametersTuple)
        #print("Ground Truth, state, failure, obs")
        #print(nextPhysicalState,nextFailureState,nextObservation)
        #err, some reach this, not all?
        #Update beliefs
        nextBelief = estimatorF(nextAction,nextObservation,beliefList[-1],currentFailureParticles,systemF,systemParametersTuple,physicalStateSubEstimatorF,physicalStateJacobianF)

        #NOTE: This is a future feature that is not used yet
        #Check resample condition for general faults (hybrid case means resample could be none. Still thinking about best way here)
        if generalFaultDict and generalFaultDict["failureParticleResampleCheckF"] and  generalFaultDict["failureParticleResampleCheckF"](nextBelief):
            #Resample based on given function
            sampleRngKey, rngSubKey = jaxRandom.split(sampleRngKey)
            #Here we expect the tuple to have the current failure particles included
            actionHistory = tuple(actionList) + (nextAction,)
            observationHistory = tuple(observationList) + (nextObservation,)
            #print(actionHistory)
            #print(observationHistory)
            print(timeStep)
            nextBelief = generalFaultDict["failureParticleResampleF"](nextBelief+(currentFailureParticles,),timeStep,estimatorF,beliefList[0],actionHistory,
                                                                      observationHistory, systemF,systemParametersTuple,physicalStateSubEstimatorF,physicalStateJacobianF,rngSubKey)
            currentFailureParticles = nextBelief[2]
        #Does not reach here
        #Evaluate Reward, we now allow for probabilistic rewards (usually b/ of safety constraint)
        rngKey, rngSubKey = jaxRandom.split(rngKey)
        appendTimeStepToTrialResults(physicalStateList,nextPhysicalState,failureStateList,nextFailureState,beliefList,nextBelief,
                                 rewards,rewardF,rngSubKey,actionList,nextAction,observationList,nextObservation,timeStep,saveTreeFlag,tree,treeList)

        #We will let it break here, if want to use jit, have to get more creative?
        #Check if reward over threshold
        if rewards[timeStep+1] > diagnosisThreshold:
            success = terminateEarly(nextBelief,currentFailureParticles,nextFailureState,timeStep,nExperimentSteps,rewards,saveTreeFlag,treeList)
            break
        #Check for filter divergence (can do it here, as will fix it before the next action)
        if jnp.isnan(rewards[timeStep+1]) and filterDivergenceHandlingMethod:
            handleFilterDivergence(filterDivergenceHandlingMethod,rewards,timeStep,diagnosisThreshold,beliefList)

    #Set success to nan if we don't converge. Rationale here is this allows us to check for this later, and then count as a failure (or not) as desired.
    #In python else is only entered if we DON'T break
    else:
        if computeSuccessAtEnd and not computeSafetyAtEnd: #Avoid computing twice
            diagnosis = diagnoseFailure(nextBelief,currentFailureParticles)
            #If the last reward is nan, we diverged so this is always a failure
            if not jnp.isnan(rewards[-1]) and jnp.all(diagnosis == nextFailureState):
                success = 1
        else:
            success = jnp.nan

    #If we want to give credit for being right but not confident, use this instead
    ##In python else is only entered if we DON'T break. We want to check for success if we time out still! (useful in the case of safety where we can't break early)
    #else:
    #    diagnosis = diagnoseFailure(nextBelief,currentFailureParticles)
    #    if jnp.all(diagnosis == nextFailureState):
    #        success = 1
    #    else:
    #        success = jnp.nan

    #return results!
    return makeTrialResultDict(physicalStateList,failureStateList,beliefList,rewards,actionList,initialFailureParticles,success,timeStep,wctStartTime,saveTreeFlag,treeList,computeSafetyAtEnd)

def runNetworkTrial(initializationDict,nSimulationsPerTree,nExperimentSteps,systemF,systemParametersTuple,solverF,solverParametersTuple,estimatorF, # pylint: disable=unused-argument
             physicalStateSubEstimatorF,physicalStateJacobianF,physicalStateSubEstimatorSampleF,rewardF,initialFailureParticles, generalFaultDict, diagnosisThreshold,filterDivergenceHandlingMethod, # pylint: disable=unused-argument
             saveTreeFlag, rngKey, computeSuccessAtEnd=False,computeSafetyAtEnd=True, numWarmStart=0): # pylint: disable=unused-argument
    """
    Networked systems runner
    """
    futureCapability = "The distributed version of s-FEAST is intended future work, but is not currently implemented"
    raise NotImplementedError(futureCapability)

def handleFilterDivergence(filterDivergenceHandlingMethod,rewards,timeStep,diagnosisThreshold,beliefList):
    """
    Method to handle filter divergence if needed.
    """
    failureWeightsIdx = 0
    filtersIdx = 1
    if filterDivergenceHandlingMethod == "acceptDiagnosisBeforeNan":
        #Check if we are above the threshold we set for accepting the diagnosis if needed (otherwise, let filter diverge, we failed)
        if rewards[timeStep] > diagnosisThreshold-1:
            print(f"Over confidence threshold {diagnosisThreshold-1}")
            #Set *failure* belief to our diagnosis
            diagnosisIdx  = jnp.argmax(beliefList[timeStep][failureWeightsIdx])
            diagnosis = jnp.zeros(len(beliefList[timeStep][failureWeightsIdx]))
            diagnosis = diagnosis.at[diagnosisIdx].set(1)
            beliefList[timeStep+1] = (diagnosis,beliefList[timeStep+1][filtersIdx]) #Note this won't handle general fault particles, TODO
        else:
            print(f"Under confidence threshold {diagnosisThreshold-1}, will diverge")
    else:
        unrecognizedFilterDivergenceHandlingMethod = f"The filter divergence handling method {filterDivergenceHandlingMethod} is not recognized or implemented."
        raise ValueError(unrecognizedFilterDivergenceHandlingMethod)

def setUpTrialOutputs(initializationDict,nExperimentSteps,saveTreeFlag):
    """
    Helper method to set up trials for running
    """

    #Initialize data to be returned
    physicalStateList = [initializationDict["physicalState"]]
    failureStateList = [initializationDict["failureState"]]
    beliefList = [initializationDict["beliefTuple"]]
    actionList = [initializationDict["firstAction"]]
    observationList = [initializationDict["firstObservation"]]
    #breaks if using solver w/0 action list like cbf: actionList = [solverParametersTuple[idxAvailableActionList][idxNullAction]] #First action in the list of available actions is null action
    treeList = None
    if saveTreeFlag: #Only save tree if told to
        treeList = [] #We will add None to end, b/ tree thinkings about next action from current state

    #we'll use onp arrays for flexibility and to make processing later easier for the data we want to average
    rewards = onp.zeros(nExperimentSteps+1)
    rewards[0] = initializationDict["reward"]

    return physicalStateList,failureStateList,beliefList,actionList,observationList,treeList,rewards

def appendTimeStepToTrialResults(physicalStateList,nextPhysicalState,failureStateList,nextFailureState,beliefList,nextBelief,
                                    rewards,rewardF,rngKey,actionList,nextAction,observationList,nextObservation,timeStep,saveTreeFlag,tree,treeList):
    """
    Helper function that appends all the data for a time step to the trial results
    """

    #Evaluate Reward, we now allow for probabilistic rewards (usually b/ of safety constraint)
    nextReward = rewardF(nextBelief,rngKey)

    #Add everything to our data lists
    physicalStateList.append(nextPhysicalState)
    failureStateList.append(nextFailureState) #Potential optimization: only save this once? Doesn't change by assumption. Leaving since it is small compared to rest of data
    beliefList.append(nextBelief)
    rewards[timeStep+1] = nextReward
    actionList.append(nextAction)
    observationList.append(nextObservation)
    #Only save tree if told to, as it is very large compared to the other data
    if saveTreeFlag:
        treeList.append(tree)

#Future TODO: Need to adjust for general fault with changing failure particles, currently not supported
def terminateEarly(nextBelief,currentFailureParticles,nextFailureState,timeStep,nExperimentSteps,rewards,saveTreeFlag,treeList):
    """
    Helper method handling early termination.
    """
    #Now add None to the end of the trees (if saving the tree)
    if saveTreeFlag: #Only save tree if told to
        treeList.append(None)
    diagnosis = diagnoseFailure(nextBelief,currentFailureParticles) #Future TODO: Need to adjust for general fault with changing failure particles
    if jnp.all(diagnosis == nextFailureState):
        success = 1
    else:
        success = 0

    # Check if we terminated early, get extra reward if so. NOTE: it gets the extra reward EVEN if the answer is wrong.
    # The reasoning here is that the POMDP has no way of knowing "correctness" and the POMDP reward is only on convergence, so this should only be on convergence
    if timeStep < nExperimentSteps - 1:
        rewards[timeStep+2:] = 1 #Set to max reward
    return success

#Can't jit while there is a if statement.
def initializeTrial(numState, numAct, numSen,providedFailure,numNFailures,nMaxFailureParticles,beliefInitializationF,rewardF,generalFaultDict,rngKey,# pylint: disable=too-many-branches
                    initialState=None,initialUncertainty=.001,numAgents=1):
    """
    Method that creates the initial physical and failure state, beliefTuple and reward. Also generates the possible failures.
    Both true and possible failures are randomized, currently nothing else is.

    Parameters
    ----------
    numState : int
        Number of states
    numAct : int
        Number of actuators
    numSen : int
        Number of sensors
    providedFailure : array, shape(numAct+numSen) (default=None)
        Provided failure (if any) to have each trial use
    numNFailures : array, shape(nMaxComponentFailures+1)
        Number of failure combinations for each level of failed components
    nMaxFailureParticles : int
        Maximum number of possible failures to consider at once. If larger than the number of possible unique failures, all possibilities are considered
    beliefInitializationF : function
        Function that creates the initial belief
    estimatorF : function
        Estimator function to update the beliefs with. Takes batch of filters
    physicalStateSubEstimatorF : function
        Physical state estimator for use with the marginal filter, if any
    rewardF : function
        Reward function to evaluate the beliefs with
    generalFaultDict : dict
        If we are using a general fault model this is a dictionary with the following values. Otherwise it is None
        failureParticleResampleF : function
            Function that resamples the particles when needed
        failureParticleResampleCheckF : function
            Function that determines if resampling is needed
        failureParticleInitializationF : function
            Function that creates the initial failure particles
    rngKey : JAX PRNG key
        Key for the PRNG, to be used to create pseudorandomness
    initialState : array, shape(numState) (default=None)
        Initial state if any provided. If not, defaults to origin.
    initialUncertainty : float (default=.001)
        Initial state uncertainty magnitude, if any
    numAgents : int (default=1)
        How many agents are present (distinguishes single from multi-agent which is currently not supported)

    Returns
    -------
    initializationDict : dict
        Initial physical state, failure state, beliefTuple, reward, firstAction (=0), firstObservation (=0)
    initialFailureParticles : array, shape(nMaxFailureParticles,numAct+numSen)
        List of possible failures to be considered by the solver. If not exhaustive, these are a random subset. Belief failure weights are are over these possibilities
    """

    #enumerated fault case. hybridFaultFlag is checked again later. This is written so if the first condition is true (generalFaultDict is None)
    #Then hybridFaultFlag is None or False. If the second condition is true (necessitates first is false), then hybridFaultFlag is True
    #If both are false, hybridFaultFlag is False.
    if (hybridFaultFlag := generalFaultDict) is None or (hybridFaultFlag := generalFaultDict["failureParticleInitializationF"] == binaryToDegradationFaults): #pylint: disable=comparison-with-callable
        #Generate possible failures (THIS ONLY APPLIES TO ENUMERATED BINARY FAULTS!)
        #########################################
        #First compute total possible number
        nFailureCombs = jnp.sum(numNFailures)
        rngKey,rngSubKey = jaxRandom.split(rngKey)
        if nMaxFailureParticles < nFailureCombs:
            #Really slow when there is a large number of possible failures, so will just randomly generate if too large (and assume repeats odds are low so regen easily)
            if nFailureCombs > 10000:
                while True:
                    initialFailureParticlesIdxes = jaxRandom.randint(rngSubKey, minval=0, maxval=nFailureCombs, shape=(nMaxFailureParticles,))
                    dummyVals, counts = jnp.unique(initialFailureParticlesIdxes,return_counts=True)
                    if jnp.max(counts) == 1:
                        break
            else:
                #Returns un ordered! So don't need to randomly pick from these, can always take the first.
                initialFailureParticlesIdxes = jaxRandom.choice(rngSubKey, nFailureCombs, (nMaxFailureParticles,),replace=False)
        else:
            #Need to scramble, because taking first idex as true failure
            initialFailureParticlesIdxes = jaxRandom.permutation(rngSubKey, jnp.arange(nFailureCombs,dtype=int))
        #Now generate the possible failures
        initialFailureParticles = generateAllPossibleFailures(numAct,numSen,numNFailures,initialFailureParticlesIdxes,numAgents)

        #Only validate single agent failures for now
        if numAgents==1:
            #Validate that the system is solvable (ie, no double sensing failure, as then we can't solve at all)
            for iFailure,possibleFailure in enumerate(initialFailureParticles):
                #Check if not valid and update  NOTE: can get duplicate failures, not observed to be an issue
                newFailureIfNeeded = checkFailureIsValid(possibleFailure,numSen)
                if newFailureIfNeeded is not None:
                    initialFailureParticles = initialFailureParticles.at[iFailure].set(newFailureIfNeeded)


    #General Fault Case
    else:
        #Start with totally general case where every component could be partially broken, determined by failureParticleInitializationF, which
        #could be biased to nominal components.
        #Future TODO: Configure to use nMaxComponentFailures?
        rngKey,rngSubKey = jaxRandom.split(rngKey)
        #nMaxFailureParticles is the number of particles we start with. numFallibleComponents = numAct+numSen
        initialFailureParticles = generalFaultDict["failureParticleInitializationF"](nMaxFailureParticles, numAct,numSen,rngSubKey,providedFailure)


    #Add in the providedFailure if we didn't generate it (replaces first failure always, since this is random, this is fine)
    #In either case, always select the first failure (no loss of generality), unless general fault with the initial failure not in the failure particles
    #NOTE: (WARNING) if we ever try to learn, need to be careful not to learn that the first failure is always the answer
    if providedFailure is not None:
        failureState = providedFailure

        if generalFaultDict is None or generalFaultDict["trueFaultInInitialParticlesFlag"]:
            #This checks if any array matches (will fail in general case almost certainly, and almost always in 0/1)
            #If it does, then we already have a particle for the general case and shouldn't add a duplicate
            if not jnp.any(jnp.all(providedFailure == initialFailureParticles,axis=1)):
                initialFailureParticles = initialFailureParticles.at[0].set(providedFailure)
            #We're done, as now true fault is in the initial particle set
        #In the general fault case where we don't want the initial fault in the particle set, we're done

    #Incase with no provided failure where we want the true fault in the initial particle set, this is all we need to do.
    elif generalFaultDict is None or generalFaultDict["trueFaultInInitialParticlesFlag"]:
        failureState = initialFailureParticles[0]

    #Finally, in this case, in general fault not in initial particle set
    else:
        #Fault isn't known in general!
        rngKey,rngSubKey = jaxRandom.split(rngKey)
        failureState = generalFaultDict["failureParticleInitializationF"](2, numAct+numSen,rngSubKey)[0]

    #Finally, check if 0/1 true fault or degradation is present
    if hybridFaultFlag:
        rngKey,rngSubKey = jaxRandom.split(rngKey)
        failureState = generalFaultDict["failureParticleInitializationF"](failureState,rngSubKey)



    #Create Initial state
    #####################################
    initializationDict = {}

    #Set default physical state if one isn't provided
    if initialState is not None:
        if len(initialState) != numState:
            inconsistentInitialState = f"The provided initial state ({initialState}) is inconsistent with the expected number of states ({numState})"
            raise ValueError(inconsistentInitialState)
        initializationDict["physicalState"] = initialState
    else:
        initializationDict["physicalState"] = jnp.zeros(numState)

    initializationDict["failureState"] = failureState

    #Create Belief #Future addition TODO, add ability for adversarial or other initial priors
    #print(initializationDict["physicalState"])
    initializationDict["beliefTuple"] = beliefInitializationF(initializationDict["physicalState"],initialFailureParticles,initialUncertainty)

    #Generate Initial reward, we now allow for probabilistic rewards (usually b/ of safety constraint).
    # Don't need to split because last use of this key
    initializationDict["reward"] = rewardF(initializationDict["beliefTuple"],rngKey)

    initializationDict["firstAction"] = jnp.zeros(numAct)
    initializationDict["firstObservation"] = jnp.zeros(numSen)

    return initializationDict, initialFailureParticles

def visualize(experimentParamsDict,saveDirectoryPath):
    """
    Visualizes

    Parameters
    ----------
    experimentParams : dict
        Relevant parameters are:
            nSimulationsPerTreeList : list, len(numTrials)
                The number of simulations performed before returning an action (when not running on time out mode).
                This parameter is an array, if longer then length 1, multiple trials are run, varying the number of simulations per tree.
            dt : float
                The time between time steps of the experiment
            solverNamesList: list
                List of names of solvers, for data logging
            networkFlag : bool
                Whether we are in a distributed network or not
    saveDirectoryPath : str
        String of the absolute path to the saveDirectory for the results of the experiment. Auto determined from the experiment parameters, unless overwritten.
    """

    outputFilePath = os.path.join(saveDirectoryPath,"render.pdf")
    if experimentParamsDict["networkFlag"]:
        futureCapability = "The distributed version of s-FEAST is intended future work, but is not currently implemented"
        raise NotImplementedError(futureCapability)
    #else:
    visualizeFirstTrajectory(saveDirectoryPath,experimentParamsDict,outputFilePath)

def getCommandLineArgs():
    """
    Helper function to parse command line arguments
    """

    #Add --noSave argument for testing code w/o potentially overwriting data
    parser = argparse.ArgumentParser(description='Generate and process data for given trial configurations')
    parser.add_argument('configFilePath',help='Relative (or absolute) path to the config file')
    parser.add_argument('--noSave', dest='save', action='store_false', default=True,
                        help='Flag to disable saving data')
    commandLineArgs = parser.parse_args()
    return commandLineArgs

def setUpMultiPRocessing():
    """
    Helper function to setup multiprocessing
    """
    #JAX is internally threaded, so if we do any multi-processing, need to set it to spawn new processes
    mp.set_start_method('forkserver')
    ctx._force_start_method('forkserver') #Recommended by JAX, but should look at a better way in the future. pylint: disable=protected-access


#Entry point to code, called by "python pipeline.py"
if __name__ == '__main__':
    COMMAND_LINE_ARGS_INPUT = getCommandLineArgs()
    setUpMultiPRocessing()
    #Debug turn off jit
    #with jax.disable_jit():
    #    with jax.default_device(jax.devices("cpu")[0]):
    #       jax.config.update("jax_debug_nans", True)
    #       main(COMMAND_LINE_ARGS_INPUT.configFilePath,COMMAND_LINE_ARGS_INPUT.save)
    #Context manager that disables GPU (faster in current s-FEAST algorithm) This is the proper way to do this now, setting GPU to be invisible raises an error
    with jax.default_device(jax.devices("cpu")[0]):
        main(COMMAND_LINE_ARGS_INPUT.configFilePath,COMMAND_LINE_ARGS_INPUT.save)
