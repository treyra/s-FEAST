"""
Main file for setting up, running, and logging data for experiments simulated to run in real-time.
Similar to pipeline, and future work is to reduce/eliminate redundancy, for now ignoring.

Currently only supports binary faults!

Imports as much as possible from pipeline.py
"""
# pylint: disable=duplicate-code
import os
import time
import inspect

#Multiprocessing components to make things run faster (implement later)
#from itertools import repeat
#import multiprocessing as mp
import numpy as onp #Following Jax conventions

import jax.numpy as jnp
#import jax
#from functools import partial
from jax import random as jaxRandom
from tqdm import tqdm

#failurePy imports, these need to be installed by pip to work
from failurePy.load.yamlLoader import loadExperimentParamsFromYaml
from failurePy.load.yamlHardwareLoader import loadRealTimeParams
from failurePy.load.yamlLoaderUtilityMethods import raiseIncompatibleSpecifications
from failurePy.utility.saving import checkSaveDirectoryPathMakeIfNeeded, processDataAverages, checkIfDataExists,setUpDataOutput,saveTrialResult,makeTrialResultDict,getTrialDataPath #,saveMetaData
from failurePy.solvers.randomPolicy import solveForNextAction as randomPolicyF
#For determining system type
from failurePy.pipeline import visualize, initializeTrial, getCommandLineArgs, setUpMultiPRocessing,terminateEarly
from failurePy.utility.pipelineHelperMethods import getExperimentParameters, diagnoseFailure


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

    #Cuda not installed on Jimmy desktop
    #Toggle CUDA (GPU) off
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    #First load in the experiment parameters. These determine the save path, so this is returned as well
    experimentParamsDict, saveDirectoryPath = loadExperimentParamsFromYaml(configFilePath) #We won't use extra data here pylint: disable=unbalanced-tuple-unpacking

    #Make absolute save directory if it isn't already
    if not saveDirectoryPath[0] == "/":
        saveDirectoryPath = os.path.join(os.getcwd(),saveDirectoryPath)

    #Check for compatibility, real-time solver is different
    checkRealTimeSolverCompatibility(experimentParamsDict)

    #Get initial action (if any). Loading here instead of in loadExperimentParams as this is only for real-time implementations
    initialAction = loadRealTimeParams(configFilePath)

    #Check if save directory is clean
    checkSaveDirectoryPathMakeIfNeeded(saveDirectoryPath,experimentParamsDict)

    #Setup data output
    setUpDataOutput(saveDirectoryPath, experimentParamsDict, configFilePath)

    #Next run experiment
    runExperimentsAndLog(experimentParamsDict,initialAction,saveDirectoryPath,saveData)

    #Log data (potentially do something with a return code at a later date?)
    processDataAverages(saveDirectoryPath,experimentParamsDict)

    if visualizeFirstTrajectoryFlag:
        #Clean up / display
        visualize(experimentParamsDict,saveDirectoryPath)

def runExperimentsAndLog(experimentParamsDict,initialAction,saveDirectoryPath,saveData=True):
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
            nMaxPossibleFailures : int
                Maximum number of possible failures to consider. If larger than the number of possible unique failures, all possibilities are considered
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
    initialAction : array, len(numAct)
        Initial action to take (zeros if none provided)
    saveDirectoryPath : str
        String of the absolute path to the saveDirectory for the results of the experiment. Auto determined from the experiment parameters, unless overwritten.
        Now using pickle to save the data in it's original dictionary format, rather than creating directories for each data type saved
    saveData : boolean (default=True)
        Over-rides data saving behavior when False to not write out any data, avoiding potential overwrite when testing or profiling
    """

    #Get basic parameters of the system
    numState, numAct, numSen, numNFailures, numWarmStart, numAgents = getExperimentParameters(experimentParamsDict)

    if numAgents > 1:
        realtimeMultiAgent = "Real time distributed systems are not currently supported"
        raise NotImplementedError(realtimeMultiAgent)

    #Validate initial action
    if initialAction is not None:
        if len(initialAction) != numAct:
            raiseIncompatibleSpecifications(f"initialAction: {initialAction}",f"numAct: {numAct}")
    else:
        initialAction = jnp.zeros(numAct)

    #Loop over solvers
    for iSolver in range(len(experimentParamsDict["solverFList"])):

        #Loop over nSimulationsPerTrees
        for nSimulationsPerTree in experimentParamsDict["nSimulationsPerTreeList"]:

            #Check if we're multiprocessing (Speed up for parallelizing over solvers and experiments seems minimal)
            #This relies on the processes only being spawned at opening of pool, so we don't need to recompile every time, otherwise its a major slow down.
            if experimentParamsDict["multiprocessingFlag"]:
                raise NotImplementedError("Not Implemented for real time pipeline yet")

            #else: #Will implement multiprocessing later
            #Loop over trials
            for jTrial in tqdm(range(experimentParamsDict["nTrialsPerPoint"])):
                #Only warm start first trial
                if numWarmStart and jTrial != 0:
                    warmStart=0
                else:
                    warmStart=numWarmStart
                #Now allowing to have data already existing
                if experimentParamsDict["mergeData"] and checkIfDataExists(getTrialDataPath(saveDirectoryPath, experimentParamsDict["solverNamesList"][iSolver], nSimulationsPerTree, jTrial)):
                    continue #Don't overwrite
                initializeRunAndSaveTrial(experimentParamsDict,initialAction,numState, numAct, numSen,numNFailures,saveDirectoryPath,nSimulationsPerTree,
                                            iSolver,jTrial+experimentParamsDict["rngKeysOffset"],saveData,numWarmStart=warmStart)


#def initializeRunAndSaveTrialWrapper(args):
#    """
#    Wrapper of initializeRunAndSaveTrial that allows for all the args to be passed as one iterable for multiprocessing
#    """
#
#    initializeRunAndSaveTrial(*args)


def initializeRunAndSaveTrial(experimentParamsDict,initialAction,numState, numAct, numSen,numNFailures,saveDirectoryPath,nSimulationsPerTree,iSolver,rngSeed,
                              saveData=True,numWarmStart=0):
    """
    Helper method that performs the initialize, run and saving for each trial, to enable parallelization

    Parameters
    ----------
    experimentParamsDict : dict
        Dictionary containing all the relevant experiment parameters.
    initialAction : array, len(numAct)
        Initial action to take (zeros if none provided)
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
    """

    #Set random policy whenever nSimulationsPerTree = 0
    if nSimulationsPerTree == 0:
        solverF = randomPolicyF
    else:
        solverF = experimentParamsDict["solverFList"][iSolver]

    #Make initial rngKey and split
    rngKey = jaxRandom.PRNGKey(rngSeed)
    rngKey,rngSubKey = jaxRandom.split(rngKey)
    initializationDict, possibleFailures = initializeTrial(numState, numAct, numSen,
                                                           experimentParamsDict["providedFailure"],numNFailures,
                                                           experimentParamsDict["nMaxPossibleFailures"],experimentParamsDict["beliefInitializationF"],
                                                           experimentParamsDict["rewardF"],
                                                           None, #No generalFualt dict
                                                           rngSubKey,initialState=experimentParamsDict["initialState"])

    initializationDict["initialAction"] = initialAction

    #Run trial
    trialResultDict = runTrial(initializationDict,nSimulationsPerTree,experimentParamsDict["nExperimentSteps"],
                               experimentParamsDict["systemF"],experimentParamsDict["systemParametersTuple"],
                               solverF,experimentParamsDict["solverParametersTuplesList"][iSolver],#Get the list of solver params corresponding to this solver
                               experimentParamsDict["estimatorF"],experimentParamsDict["physicalStateSubEstimatorF"],experimentParamsDict["physicalStateJacobianF"],
                               experimentParamsDict["physicalStateSubEstimatorSampleF"],experimentParamsDict["rewardF"],possibleFailures,
                               experimentParamsDict["diagnosisThreshold"], experimentParamsDict["saveTreeFlag"], rngKey, numWarmStart=numWarmStart) #pass in PRNG key for this trial
                               #Future TODO: make computeSuccessAtEnd configurable!

    #Save this result directly (if turned on)
    if saveData:
        saveTrialResult(saveDirectoryPath, experimentParamsDict["solverNamesList"][iSolver], nSimulationsPerTree, rngSeed, trialResultDict)


#Add leap frogging actions
def runTrial(initializationDict,nSimulationsPerTree,nExperimentSteps,systemF,systemParametersTuple,solverF,solverParametersTuple,estimatorF,
             physicalStateSubEstimatorF,physicalStateJacobianF,physicalStateSubEstimatorSampleF,rewardF,possibleFailures, diagnosisThreshold,
             saveTreeFlag, rngKey, computeSuccessAtEnd=False, numWarmStart=0): # pylint: disable too-many-branches
    """
    Method that runs one trial of a given system and solver, from given initialization.
    Runs in simulated real-time, so plans next action while current action is running.

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
    possibleFailures : array, shape(nMaxPossibleFailures,numAct+numSen)
        List of possible failures to be considered by the solver. If not exhaustive, these are a random subset. Belief failure weights are are over these possibilities
    diagnosisThreshold : float
        Level of the reward to consider high enough to return an answer.
    saveTreeFlag : boolean
        Whether to save the tree or not (it can be quite large, so if not visualizing, it is best to set this to false)
    rngKey : JAX PRNG key
        Key for the PRNG, to be used to create pseudorandomness
    computeSuccessAtEnd : boolean
        Checks if we should assign nan or not if ends without converging (ie, for safety where we don't want to break early)
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
            possibleFailures : array, shape(nMaxPossibleFailures,numAct+numSen)
                List of possible failures to be considered by the solver. If not exhaustive, these are a random subset. Belief failure weights are are over these possibilities

    """

    #Explicitly define indexes of parameters we'll get later for clarity
    idxAvailableActionList = 0
    idxNullAction = 0

    #Initialize data to be returned
    physicalStateList = [initializationDict["physicalState"]]
    failureStateList = [initializationDict["failureState"]]
    beliefList = [initializationDict["beliefTuple"]]
    actionList = [initializationDict["initialAction"]] #Now can have a pre specified action
    treeList = None #Only save tree if told to
    if saveTreeFlag:
        treeList = [] #We will add None to end, b/ tree thinkings about next action from current state

    #we'll use onp arrays for flexibility and to make processing later easier for the data we want to average
    rewards = onp.zeros(nExperimentSteps+1)
    rewards[0] = initializationDict["reward"]


    #Warm start by running the solver a few times to compile everything
    if numWarmStart:
        print("Warm start")
        warmStartSolverParametersTuple = solverParametersTuple[0:2] + (jnp.inf,) + solverParametersTuple[3:]
        for dummyIndex in range(numWarmStart):
            #Still use rng keys so burn ins are different
            rngKey, rngSubKey = jaxRandom.split(rngKey)
            #Set max sim time to inf for warm starts
            solverF(beliefList[-1],warmStartSolverParametersTuple,possibleFailures,systemF,systemParametersTuple,rewardF,estimatorF,
                        physicalStateSubEstimatorF,physicalStateJacobianF,physicalStateSubEstimatorSampleF,nSimulationsPerTree,
                        rngKey,currentAction=actionList[-1])
        print("Warm start done")

    #Get time for logging later
    wctStartTime = time.time()
    success = 0 #Will be set to 1 if returns the correct failure
    #for timeStep in tqdm(range(nExperimentSteps)): #Used to eyeball wall clock time
    for timeStep in range(nExperimentSteps):
        #Compute NEXT action to take
        rngKey, rngSubKey = jaxRandom.split(rngKey) #Keys are consumed on use, so keep splitting rngKey first to avoid consumption
        nextAction,tree = solverF(beliefList[-1],solverParametersTuple,possibleFailures,systemF,systemParametersTuple,rewardF,estimatorF,
                                  physicalStateSubEstimatorF,physicalStateJacobianF,physicalStateSubEstimatorSampleF,nSimulationsPerTree,
                                  rngSubKey,currentAction=actionList[-1])

        #Simulate system forward using CURRENT action
        rngKey, rngSubKey = jaxRandom.split(rngKey) #Keys are consumed on use, so keep splitting rngKey first to avoid consumption
        nextPhysicalState,nextFailureState,nextObservation = systemF(physicalStateList[-1],failureStateList[-1],actionList[-1],rngSubKey,systemParametersTuple)
        #print("Ground Truth, state, failure, obs")
        #print(nextPhysicalState,nextFailureState,nextObservation)

        #Update beliefs
        nextBelief = estimatorF(actionList[-1],nextObservation,beliefList[-1],possibleFailures,systemF,systemParametersTuple,physicalStateSubEstimatorF,physicalStateJacobianF)
        #Evaluate Reward, we now allow for probabilistic rewards (usually b/ of safety constraint)
        rngKey, rngSubKey = jaxRandom.split(rngKey)
        nextReward = rewardF(nextBelief,rngSubKey)

        #Add everything to our data lists
        physicalStateList.append(nextPhysicalState)
        failureStateList.append(nextFailureState) #Potential optimization: only save this once? Doesn't change by assumption. Leaving since it is small compared to rest of data
        beliefList.append(nextBelief)
        rewards[timeStep+1] = nextReward
        if saveTreeFlag: #Only save tree if told to
            treeList.append(tree)

        #We will let it break here, if want to use jit, have to get more creative?
        #Check if reward over threshold
        if nextReward > diagnosisThreshold:
            success = terminateEarly(nextBelief,possibleFailures,nextFailureState,timeStep,nExperimentSteps,rewards,saveTreeFlag,treeList)
            break

        #Didn't break, so need next action (if below iteration count, otherwise breaking anyways and will apply null action)
        if timeStep+1 < nExperimentSteps:
            #Save NEXT action to apply next time (it'll be the current action next loop)
            actionList.append(nextAction)

    #Set success to nan if we don't converge. Rational here is this allows us to check for this later, and then count as a failure (or not) as desired.
    #In python else is only entered if we DON'T break
    else:
        if computeSuccessAtEnd:
            diagnosis = diagnoseFailure(nextBelief,possibleFailures)
            if jnp.all(diagnosis == nextFailureState):
                success = 1
        else:
            success = jnp.nan

    #If we want to give credit for being right but not confident, use this instead
    ##In python else is only entered if we DON'T break. We want to check for success if we time out still! (useful in the case of safety where we can't break early)
    #else:
    #    diagnosis = diagnoseFailure(nextBelief,possibleFailures)
    #    if jnp.all(diagnosis == nextFailureState):
    #        success = 1
    #    else:
    #        success = jnp.nan

    #Don't take the last action when we break, do nothing instead.
    # (technically last computation is wasted, but on hardware this is accurate (won't know we don't need it before hand) and if we time out it's okay)
    actionList.append(solverParametersTuple[idxAvailableActionList][idxNullAction])

    #return results!
    return makeTrialResultDict(physicalStateList,failureStateList,beliefList,rewards,actionList,possibleFailures,success,timeStep,wctStartTime,saveTreeFlag,treeList)

def checkRealTimeSolverCompatibility(experimentParamsDict):
    """
    Method that checks for known inconsistencies in the experimentParameters with real-time simulation
    This method will be expanded as more inconsistencies are identified.
    Raises error with inconsistent parameters when identified

    Parameters
    ----------
    experimentParamsDict : dict
        Dictionary containing all the relevant experiment parameters.
    """

    #Check solver is real-time, take in extra parameter
    solverFList = experimentParamsDict["solverFList"]
    for solverF in solverFList:
        methodSignature = inspect.signature(solverF)
        if not "currentAction" in methodSignature.parameters:
            raiseIncompatibleSpecifications(str(solverF),"real-time simulation framework, as it does not accept a currentAction parameter.")


#Entry point to code, called by "python pipeline.py"
if __name__ == '__main__':

    COMMAND_LINE_ARGS_INPUT = getCommandLineArgs()
    setUpMultiPRocessing()

    ##Debug turn off jit
    #import jax
    #with jax.disable_jit():
    #    from jax.config import config
    #    config.update("jax_debug_nans", True)
    #    main(commandLineArgs.configFilePath,commandLineArgs.save)

    main(COMMAND_LINE_ARGS_INPUT.configFilePath,COMMAND_LINE_ARGS_INPUT.save)
