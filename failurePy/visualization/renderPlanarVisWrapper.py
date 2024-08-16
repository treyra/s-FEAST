"""
File that wraps renderPlanarVis.py to use visualization code with new data structure with minimal re-writing.
"""
import os
import subprocess
import pickle
import numpy as onp #Will convert to onp arrays here (after all experiments are done)
import matplotlib.pyplot as plt
from tqdm import tqdm
from jax import random as jaxRandom
from jax import numpy as jnp
from failurePy.visualization import renderPlanarVis, renderPlanarVisGeneralFault
from failurePy.utility.saving import checkOrMakeDirectory

def visualizeFirstTrajectory(saveDirectoryPath,experimentParamsDict,outputFilePath=None,regenTree=0):
    """
    Function that takes experiment ResultsList and grabs the first trial result to visualize

    Parameters
    ----------
    saveDirectoryPath : str
        String of the absolute path to the saveDirectory for the results of the experiment. Auto determined from the experiment parameters, unless overwritten.
    experimentParams : dict
        Relevant parameters are:
            nSimulationsPerTreeList : list, len(numTrials)
                The number of simulations performed before returning an action (when not running on time out mode).
                This parameter is an array, if longer then length 1, multiple trials are run, varying the number of simulations per tree.
            dt : float
                The time between time steps of the experiment
            solverNamesList: list
                List of names of solvers, for data logging
            rngKeysOffset : int
                Offset to the initial PRNG used in generating the initial failure states and randomness in the trials. This is added to the trial number to allow for different trials to be preformed
    outputFilePath : str (default=None)
        Path to where the output file should be saved. If it doesn't exist, default used
    regenTree : int (default=0)
        Recreates a plausible tree given the belief and simulation level. No tree drawn when one already exists
        or when the value is 0
    """

    visualizeNthTrajectory(saveDirectoryPath,experimentParamsDict,outputFilePath,regenTree=regenTree)

#dt is universal, so making exception
def visualizeNthTrajectory(saveDirectoryPath,experimentParamsDict,outputFilePath,hardwareFlag=False,regenTree=0): # pylint: disable=invalid-name
    """
    Function that takes experiment ResultsList and grabs the nth trial result to visualize

    Parameters
    ----------
    saveDirectoryPath : str
        String of the absolute path to the saveDirectory for the results of the experiment. Auto determined from the experiment parameters, unless overwritten.
    experimentParams : dict
        Relevant parameters are:
            nSimulationsPerTreeList : list, len(numTrials)
                The number of simulations performed before returning an action (when not running on time out mode).
                This parameter is an array, if longer then length 1, multiple trials are run, varying the number of simulations per tree.
            dt : float
                The time between time steps of the experiment
            solverNamesList: list
                List of names of solvers, for data logging
            rngKeysOffset : int
                Offset to the initial PRNG used in generating the initial failure states and randomness in the trials. This is added to the trial number to allow for different trials to be preformed
    outputFilePath : str (default=None)
        Path to where the output file should be saved. If it doesn't exist, default used
    hardwareFlag : bool (default=False)
        Flag to indicate this is date from the hardware.
        NOTE: this isn't quite working yet unfortunately, mostly because we don't have the needed data from hardware
    regenTree : int (default=0)
        Recreates a plausible tree given the belief and simulation level. No tree drawn when one already exists
        or when the value is 0
    """

    solverName = experimentParamsDict["solverNamesList"][0]
    nSimulationsPerTree = experimentParamsDict["nSimulationsPerTreeList"][0]
    nTrial=experimentParamsDict["rngKeysOffset"]
    dt=experimentParamsDict["dt"]
    safetyFunctionF=experimentParamsDict["safetyFunctionF"]
    plottingBounds=experimentParamsDict["plottingBounds"]
    resolution=experimentParamsDict["resolution"]

    #Get data and pass on
    trialResultsDict = loadNthTrialResults(saveDirectoryPath,solverName,nSimulationsPerTree,nTrial,regenTree,experimentParamsDict)

    if hardwareFlag: #Currently bounds and resolution not configurable here, since hardcoded for now
        visualizeHardwareTrajectory(trialResultsDict,dt,outputFilePath,safetyFunctionF)
    else:
        visualizeSingleTrajectory(trialResultsDict,dt,outputFilePath,safetyFunctionF,plottingBounds,resolution)

def loadNthTrialResults(saveDirectoryPath,solverName,nSimulationsPerTree,nTrial,regenTree=0,experimentParamsDict=None):
    """
    Loads the data from the specified trial and returns it

    Parameters
    ----------
    saveDirectoryPath : str
        String of the absolute path to the saveDirectory for the results of the experiment. Auto determined from the experiment parameters, unless overwritten.
    solverName : str
        The solver to get the data from
    nSimulationsPerTree: int
        The  number of simulations per tree to get data from.
    nTrial : int
        The trial to get the data from
    regenTree : int (default=0)
        Recreates a plausible tree given the belief and simulation level. No tree drawn when one already exists
        or when the value is 0
    experimentParams : dict (default=None)
        Experiment parameters needed when regenerating tree

    Returns
    -------
    trialResultsDict : dict
        Dictionary with the stored trial results
    """
    solverDirectoryPath = os.path.join(saveDirectoryPath,solverName)
    nSimPath =  os.path.join(solverDirectoryPath,str(nSimulationsPerTree))
    nTrialPath =  os.path.join(nSimPath,str(nTrial))
    trialDataPath = os.path.join(nTrialPath,"trialData.dict")
    with open(trialDataPath, "rb") as trialDataFile:
        trialResultsDict = pickle.load(trialDataFile)

    #Check if we need to make a new tree
    existingTreeList = False
    if "treeList" in trialResultsDict:
        dataTreeList = trialResultsDict["treeList"]
        if len(dataTreeList) != 1 or dataTreeList[0] is not None:
            existingTreeList = True

    if regenTree and not existingTreeList:
        #Check for cache
        cacheFilePath = os.path.join(nTrialPath,"regenTreeCache.pckl")
        if os.path.exists(cacheFilePath):
            with open(cacheFilePath, "rb") as rootNodeDataFile:
                rootNodeList = pickle.load(rootNodeDataFile)
        else:
            if experimentParamsDict is None:
                experimentParamsDictRequired = "experimentParamsDict must be provided to regenerate the tree"
                raise ValueError(experimentParamsDictRequired)
            rootNodeList = regenTreeFromData(trialResultsDict,experimentParamsDict,numSimulations=regenTree)
            #Cache for future plotting
            with open(cacheFilePath, "wb") as rootNodeDataFile:
                pickle.dump(rootNodeList,rootNodeDataFile)
        trialResultsDict["treeList"] = rootNodeList

    return trialResultsDict

#dt is universal, so making exception
def visualizeSingleTrajectory(trialDataDict,dt=1,outputFilePath=None,safetyFunctionF=None,plottingBounds=None,resolution=200,showFig=True): # pylint: disable=invalid-name
    """
    Function that renders the trajectory of one trial, using the wrapped renderPlanarVis.py functions.

    Parameters
    ----------
    trialDataDict : dict
        Dict with trial results nested. All arrays are jax jnp arrays
            physicalStateList : list
                List of the (realized) physical states of the system
            failureStateList : list
                List of the (unchanging) true failure state
            beliefList : list
                List of the beliefs at each time step (time steps determined by nExperimentSteps and dt, which is currently set in the system model)
                ASSUMES MARGINALIZED FILTER BELIEF AND KALMAN BELIEF
            rewardList : list
                List of the rewards at each time step
            actionList : list
                List of the actions taken at each time step
            treeList: list
                List of the tree data at each time step. Each element is a tuple with the nodeList and the valuesRewardsVisitsArray for the tree
            possibleFailures : array, shape(nMaxPossibleFailures,numAct+numSen)
                List of possible failures to be considered by the solver. If not exhaustive, these are a random subset. Belief failure weights are are over these possibilities
    dt : float (default=1)
        The time between time steps of the experiment
    outputFilePath : str (default=None)
        Path to where the output file should be saved. If it doesn't exist, default used
    safetyFunctionF : function (default=None)
        Constraints that physical states must satisfy to be safe. Returns 1 if safe, 0 if not.
    plottingBounds : array, shape(2,2) (default=None)
        Bounds of the axis
    resolution : int (default=200)
        How high of a resolution the safe zone should be drawn in.
    showFig : boolean (default=True)
        Shows the trajectory when rendering finishes
    """

    #We need to convert arrays to numpy arrays

    #print("making physical states")
    physicalStates = onp.zeros((len(trialDataDict["physicalStateList"]),len(trialDataDict["physicalStateList"][0])))

    numPhysicalStates =len(physicalStates[0])
    rotationFlag = bool(numPhysicalStates > 4)


    #Check if we can render with existing tools

    #Conditions we can render are dim == 2 (numState=4) or (dim==3 (numState=6/8(wheels)) and linear)
    if not ((numPhysicalStates == 4 and not rotationFlag) or (numPhysicalStates == 6 and rotationFlag) or (numPhysicalStates == 8 and rotationFlag)):
        print("Cannot visualize this experiment configuration")
        print(numPhysicalStates,rotationFlag)
        return

    #print("making actions")
    actions = onp.zeros((len(trialDataDict["actionList"]),len(trialDataDict["actionList"][0])))
    #print("making possible failures")
    possibleFailures = onp.zeros((len(trialDataDict["possibleFailures"]),len(trialDataDict["possibleFailures"][0])))

    #print("wrapping physical states")
    for iState,physicalState in enumerate(trialDataDict["physicalStateList"]):
        physicalStates[iState] = onp.array(physicalState)

    #print("wrapping actions")
    for iAction,action in enumerate(trialDataDict["actionList"]):
        actions[iAction] = onp.array(action)

    #print("wrapping possible failures")
    for iFailure,possibleFailure in enumerate(trialDataDict["possibleFailures"]):
        possibleFailures[iFailure] = onp.array(possibleFailure)

    #print("wrapping failure state")
    failureState = onp.array(trialDataDict["failureStateList"][0])

    #Display success code
    print(f"Success code: {trialDataDict['success']}")

    times = onp.arange(0,dt*len(physicalStates),dt)
    #times = onp.arange(0,dt*25,dt) #Visualize only some times

    if "treeList" in trialDataDict:
        rootNodeList = trialDataDict["treeList"]
        #Check for nones, some inconsistency with real time code here
        if rootNodeList[0] is None:
            rootNodeList = None
    else:
        rootNodeList = None

    #No longer need to wrap the belief
    beliefList = trialDataDict["beliefList"]

    renderPlanarVisWrapper(physicalStates,failureState,actions,beliefList,times,rootNodeList,possibleFailures,plottingBounds=plottingBounds,rotationFlag=rotationFlag,
                            outputFilePath=outputFilePath,safetyFunctionF=safetyFunctionF,resolution=resolution,showFig=showFig)

#dt is universal, so making exception
def visualizeHardwareTrajectory(trialDataDict,dt=1,outputFilePath=None): # pylint: disable=invalid-name
    """
    Function that renders the trajectory of hardware experiment, using the wrapped renderPlanarVis.py functions.
    Current hardware data isn't enough to use this

    Parameters
    ----------
    trialDataDict : dict
        Dict with trial results nested. All arrays are jax jnp arrays
            physicalStateList : list
                List of the (realized) physical states of the system
            failureStateList : list
                List of the (unchanging) true failure state
            beliefList : list
                List of the beliefs at each time step (time steps determined by nExperimentSteps and dt, which is currently set in the system model)
                ASSUMES MARGINALIZED FILTER BELIEF AND KALMAN BELIEF
            rewardList : list
                List of the rewards at each time step
            actionList : list
                List of the actions taken at each time step
            treeList: list
                List of the tree data at each time step. Each element is a tuple with the nodeList and the valuesRewardsVisitsArray for the tree
            possibleFailures : array, shape(nMaxPossibleFailures,numAct+numSen)
                List of possible failures to be considered by the solver. If not exhaustive, these are a random subset. Belief failure weights are are over these possibilities
    dt : float (default=1)
        The time between time steps of the experiment
    outputFilePath : str (default=None)
        Path to where the output file should be saved. If it doesn't exist, default used
    """

    #We will interpolate to make the transition seem smoother
    interpolationFactor = 10

    physicalStates, actions, possibleFailures = wrapStatesActionsFailures(trialDataDict,interpolationFactor)

    #print("wrapping failure state")
    failureState = onp.array(trialDataDict["failureStateList"][0])

    #Display success code
    print(f"Success code: {trialDataDict['success']}")

    times = onp.arange(0,dt*trialDataDict["physicalStateList"],dt)

    if "treeList" in trialDataDict:
        originalRootNodeList = trialDataDict["treeList"]
        #Need to show tree for interpolated states
        rootNodeList = []
        for rootNode in originalRootNodeList:
            #Check for initial None, as previous versions of visualization showed the tree to get to a state,
            #we want to show tree to get to NEXT state
            if rootNode is None:
                continue

            for jInterpolatedState in range(interpolationFactor): # pylint: disable=unused-variable
                rootNodeList.append(rootNode)

        rootNodeList.append(None) #No tree for final state
    else:
        rootNodeList = None

    rotationFlag = bool(len(physicalStates[0]) > 4)

    #No longer need to wrap the belief
    beliefList = trialDataDict["beliefList"]

    #We're using a lot of arguments because we're using functional programming, could try to wrap these up later.
    renderPlanarVisWrapper(physicalStates,failureState,actions,beliefList,times,rootNodeList,possibleFailures, # pylint: disable=too-many-function-args
                           rotationFlag,outputFilePath)

def wrapStatesActionsFailures(trialDataDict,interpolationFactor):
    """
    Sub module that wraps the physicalState, actions and failures for use with our drawing functions
    """

    physicalStates = onp.zeros((interpolationFactor*len(trialDataDict["physicalStateList"])+2,len(trialDataDict["physicalStateList"][0])))

    actions = onp.zeros((interpolationFactor*len(trialDataDict["actionList"])+1,len(trialDataDict["actionList"][0])))

    possibleFailures = onp.zeros((len(trialDataDict["possibleFailures"]),len(trialDataDict["possibleFailures"][0])))

    for iPhysicalState,physicalState in enumerate(trialDataDict["physicalStateList"]):
        #interpolate all but the last state
        if iPhysicalState < len(trialDataDict["physicalStateList"]):
            for jInterpolatedState in range(interpolationFactor):
                interpolatedPhysicalState = interpolatePhysicalStates(physicalState,trialDataDict["physicalStateList"][iPhysicalState+1],jInterpolatedState/interpolationFactor)

            physicalStates[iPhysicalState*interpolationFactor+jInterpolatedState] = interpolatedPhysicalState
        #Add the last state at the end
        else:
            physicalStates[-1] = physicalState

    #Actions held constant, first action is always 0 (as initial state had no control)
    for iAction,action in enumerate(trialDataDict["actionList"]):
        if iAction == 0:
            actions[iAction] = onp.array(action)
        else:
            actions[1+interpolationFactor*(iAction-1):1+interpolationFactor*(iAction)] = onp.array(action)

    #print("wrapping possible failures")
    for iPossibleFailure,possibleFailure in enumerate(trialDataDict["possibleFailures"]):
        possibleFailures[iPossibleFailure] = onp.array(possibleFailure)

    return physicalStates, actions, possibleFailures

def regenTreeFromData(trialDataDict,experimentParamsDict,numSimulations): # Not going to break this up, since single use. pylint: disable=too-many-branches,too-many-statements
    """
    Function that uses the trial data to recreate the tree for visualization
    what a plausible tree looked like with the level of planning specified.
    Useful for hardware tests where the trees are not save for performance reasons

    Parameters
    ----------
    trialDataDict : dict
        Dict with trial results nested. All arrays are jax jnp arrays
            physicalStateList : list
                List of the (realized) physical states of the system
            failureStateList : list
                List of the (unchanging) true failure state
            beliefList : list
                List of the beliefs at each time step (time steps determined by nExperimentSteps and dt, which is currently set in the system model)
                ASSUMES MARGINALIZED FILTER BELIEF AND KALMAN BELIEF
            rewardList : list
                List of the rewards at each time step
            actionList : list
                List of the actions taken at each time step
            treeList: list
                Will always be None here, as we are recreating
            possibleFailures : array, shape(nMaxPossibleFailures,numAct+numSen)
                List of possible failures to be considered by the solver. If not exhaustive, these are a random subset. Belief failure weights are are over these possibilities
    experimentParams : dict
        Relevant parameters are:
            dt : float
                The time between time steps of the experiment
            solverNamesList: list
                List of names of solvers, for determining what type of tree search to perform
    numSimulations : int
        Number of simulations this recreated tree should have

    Returns
    -------
    treeList: list
        List of the tree data at each time step. Each element is a tuple with the nodeList and the valuesRewardsVisitsArray for the tree
    """

    #Assume only 1 solver
    if len(experimentParamsDict["solverNamesList"]) != 1:
        tooManySolvers = f"Regenerating the tree only works with one solver. Given {experimentParamsDict['solverNamesList']}"
        raise ValueError(tooManySolvers)
    solverName = experimentParamsDict["solverNamesList"][0]
    #Logic dependent imports
    if solverName == "SFEAST":
        from failurePy.solvers.sFEAST import solveForNextAction as solverF # pylint: disable=import-outside-toplevel
        realtime=False
    elif solverName == "realTimeSFEAST":
        from failurePy.solvers.realTimeSFEAST import solveForNextAction as solverF # pylint: disable=import-outside-toplevel
        realtime=True
    else:
        solverNameNotRecognized = f"the solver name specified '{solverName}' isn't recognized by the regenTreeFromData function."
        raise ValueError(solverNameNotRecognized)

    #Wrap as needed

    actions = jnp.zeros((len(trialDataDict["actionList"]),len(trialDataDict["actionList"][0])))
    for iAction,action in enumerate(trialDataDict["actionList"]):
        actions = actions.at[iAction].set(jnp.array(action))

    #beliefs = jnp.zeros((len(trialDataDict["beliefList"]),len(trialDataDict["beliefList"][0]),len(trialDataDict["beliefList"][0][0])))
    #for iBelief,belief in enumerate(trialDataDict["actionList"]):
    #    beliefs = beliefs.at[iBelief].set(jnp.array(belief))

    treeList = []

    rngKey = jaxRandom.PRNGKey(0)
    rngKey,rngSubKey = jaxRandom.split(rngKey)
    maxTries = 100

    solverParametersTuple = experimentParamsDict["solverParametersTuplesList"][0]
    #Remove timeout
    solverParametersTuple = solverParametersTuple[0:2] +  (jnp.inf,) + solverParametersTuple[3:]
    possibleFailures = jnp.array(trialDataDict["possibleFailures"])
    systemF = experimentParamsDict["systemF"]
    systemParametersTuple = experimentParamsDict["systemParametersTuple"]
    rewardF = experimentParamsDict["rewardF"]
    estimatorF = experimentParamsDict["estimatorF"]
    physicalStateSubEstimatorF = experimentParamsDict["physicalStateSubEstimatorF"]
    physicalStateJacobianF = experimentParamsDict["physicalStateJacobianF"]
    physicalStateSubEstimatorSampleF = experimentParamsDict["physicalStateSubEstimatorSampleF"]
    #At each time step, we will run the solver
    for iBelief, beliefTuple in enumerate(trialDataDict["beliefList"]):
        if iBelief == 0:
            treeList.append(None) #No tree for the first time step
            continue
        #Turn into Jax arrays
        beliefTuple = (jnp.array(beliefTuple[0]),jnp.array(beliefTuple[1]))
        if realtime:
            currentAction = actions[iBelief]
            nextAction = actions[iBelief+1]
        else:
            #CHECK
            nextAction = actions[iBelief]
        numRegenTries = 0
        print(f"Starting Regen #{iBelief}")
        while numRegenTries < maxTries:
            print(numRegenTries)
            rngKey,rngSubKey = jaxRandom.split(rngKey)
            if realtime:
                selectedAction, rootNode = solverF(beliefTuple, solverParametersTuple, possibleFailures, systemF, systemParametersTuple, rewardF, estimatorF,
                            physicalStateSubEstimatorF,physicalStateJacobianF,physicalStateSubEstimatorSampleF,numSimulations,rngSubKey,currentAction)
            else:
                selectedAction, rootNode = solverF(beliefTuple, solverParametersTuple, possibleFailures, systemF, systemParametersTuple, rewardF, estimatorF,
                            physicalStateSubEstimatorF,physicalStateJacobianF,physicalStateSubEstimatorSampleF,numSimulations,rngSubKey)
            if jnp.all(selectedAction==nextAction):
                treeList.append(rootNode)
                print("Found")
                numRegenTries = maxTries #Should break
            else:
                numRegenTries += 1
                print(selectedAction)
                print(nextAction)
                print(selectedAction-nextAction)
                if numRegenTries == maxTries:
                    treeList.append(rootNode)

    return treeList

def renderPlanarVisWrapper(physicalStates,failureState,actions,beliefList,times,rootNodeList,possibleFailures,plottingBounds=None,rotationFlag=False,
                            outputFilePath=None,hardwareFlag=False,safetyFunctionF=None,resolution=200,showFig=True):
    """
    Function that uses the renderPlanarVis.py functions to visualize a trajectory.

    Currently only works for 2DOF and 3DOF

    Parameters
    ----------
    physicalStates : array, shape(nTimeSteps,numState)
        Numpy array with the physical state at each time
    failureState : array, shape(numAct+numSen)
        True failure state
    actions : array, shape(nTimeSteps,numAct)
        Numpy array with the action at each time
    beliefList : list
        List of tuples representing the beliefs. Each tuple contains the array of weights on the failures and the filter array (wrapped correctly)
    times : array, shape(nTimeSteps)
        Time at each step of the experiment
    rootNodeList : list
        List of root nodes
    plottingBounds : array, shape(2,2) (default=None)
        Bounds of the axis
    rotationFlag : boolean
        Flag on what type of system we're visualizing
    outputFilePath : str (default=None)
        Path to where the output file should be saved. If it doesn't exist, default used
    hardwareFlag : boolean (default=False)
        Whether or not this is the s/c sim hardware (if true, overrides rotating3DOFFlag in terms of hardware set up)
    safetyFunctionF : function (default=None)
        Constraints that physical states must satisfy to be safe. Returns 1 if safe, 0 if not.
    resolution : int (default=200)
        How high of a resolution the safe zone should be drawn in.
    """

    #NOTE: we've switched to the format of u_i taking x_{i-1} to x_i
    #plottingBounds now can be set in config file
    #dim,defaultScale,defaultPlottingBounds = getSpacecraftPlottingParams(hardwareFlag,rotationFlag)
    dim,defaultPlottingBounds = getSpacecraftPlottingParams(hardwareFlag,rotationFlag)
    if plottingBounds is None:
        plottingBounds = defaultPlottingBounds
    #if scale is None:
    #    scale = defaultScale


    maxTime = times[-1]

    # make ground truth belief
    trueWeights = []
    for possibleFailureState in possibleFailures:
        if (possibleFailureState == failureState).all():
            trueWeights.append(1)
        else:
            trueWeights.append(0)
    trueBelief = (trueWeights, "groundTruth")

    # plot ground truth
    dummyFig, ax = plt.subplots() #ax is used a lot with matplotlib so pylint: disable=invalid-name
    numAct = len(actions[0])
    #Currently always have more actions than sensors so this works, but is a HACK
    if len(failureState) > 2*numAct:
        spacecraftDrawingF = renderPlanarVisGeneralFault.drawSpacecraft
    else:
        spacecraftDrawingF = renderPlanarVis.drawSpacecraft

    #Draw unsafe region for ground truth too
    if safetyFunctionF is not None:
        renderPlanarVis.plotUnsafeRegions(ax,safetyFunctionF,plottingBounds,resolution=resolution)
    spacecraftDrawingF(
        onp.zeros((dim*2,1)), onp.zeros((numAct,1)),
        trueBelief, possibleFailures, None, ax, plottingBounds,
        legendFlag=True,rotationFlag=rotationFlag)
    ax.set_title("Ground Truth")

    for iTimeStep, time in tqdm(enumerate(times)):
        #Get experiment values at this time
        physicalState = physicalStates[iTimeStep]
        action = actions[iTimeStep]
        beliefTuple = beliefList[iTimeStep]
        if rootNodeList is not None:
            rootNode = rootNodeList[iTimeStep]
        else:
            rootNode = None


        #Plot this time step
        dummyFig, ax = plt.subplots() #ax is used a lot with matplotlib so pylint: disable=invalid-name
        #Add obstacles, if any
        if safetyFunctionF is not None:
            renderPlanarVis.plotUnsafeRegions(ax,safetyFunctionF,plottingBounds,resolution=resolution)
        spacecraftDrawingF(physicalState,action,beliefTuple,possibleFailures,rootNode,ax,plottingBounds,
                                       legendFlag=False,rotationFlag=rotationFlag)
        ax.set_title(f"Physical Time: {time}/{maxTime}")

    saveRender(outputFilePath,showFig=showFig)

def saveRender(outputFilePath=None,makePngs=False,showFig=True):
    """
    Method that saves the produced figures

    Parameters
    ----------
    outputFilePath : string
        Where to save the output. If not specified saves locally
    makePngs : boolean (default=False)
        If true, also make a subfolder of pngs for each plot
    showFig : boolean (default=True)
        Shows the figures made when rendering finishes
    """
    #Default file name
    if outputFilePath is None:
        outputFilePath = os.path.join(os.getcwd(),"render.pdf")
    print(f"Making save file at {outputFilePath}")

    #Make png sub directory if needed,
    if makePngs:
        checkOrMakeDirectory(os.path.dirname(outputFilePath),"pngs")

    renderPlanarVis.saveFigs(outputFilePath,makePngs)
    if showFig:
        openFigs(outputFilePath)

def getSpacecraftPlottingParams(hardwareFlag,rotationFlag):
    """
    Sub method to get parameters based on experiment type
    """

    #Number of sensor or actuators is pre set
    if hardwareFlag:
        #numAct = 8
        #numSen = 6 #Add two rotation sensors
        dim =3
        #scale = .4
        #Much smaller field
        plottingBounds  = onp.array([[-3, 3],
                                    [-3, 3]])
    elif not rotationFlag:
        #numAct = 8
        #numSen = 4
        dim = 2
        #scale = 10
        plottingBounds  = onp.array([[-30, 30],
                                    [-30, 30],])
        #plottingBounds  = onp.array([[-45, 45],
        #                            [-45, 45],])

    else:
        #numAct = 10 #Add two wheels
        #numSen = 6 #Add two rotation sensors
        dim = 3
        #scale = 10 #10
        plottingBounds  = onp.array([[-30, 30],
                                    [-30, 30],])

    #return dim,scale,plottingBounds
    return dim,plottingBounds

def openFigs(fileName):
    """
    Ported from Ben's Plotter function.
    Opens the specified figure
    """

    pdfPath = os.path.join( os.getcwd(), fileName)
    if os.path.exists(pdfPath):
        #Changed this to be cross-platform compatible (as xdg-open is linux and open is mac)
        #Also has the bonus of opening in this in a detached mode
        subprocess.call(["python3", "-m", "webbrowser", pdfPath])


def interpolatePhysicalStates(currentPhysicalState,nextPhysicalState,interpolationAmount):
    """Interpolates between two physical states"""

    return (1-interpolationAmount) * onp.array(currentPhysicalState) + interpolationAmount * onp.array(nextPhysicalState)
