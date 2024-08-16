"""
Module to build solvers such as sFEAST and others
"""

import jax.numpy as jnp
import jax.random as jaxRandom

from failurePy.load.yamlLoaderUtilityMethods import loadOptionalParameter, loadRequiredParameter, raiseIncompatibleSpecifications, raiseSpecificationNotFoundError

#We do a lot of conditional importing to only load in the models, solvers, estimators as needed, and bind them to shared names to pass back to pipeline.
#We could change this to be done in sub modules, or import everything and conditionally bind the names, but we'd be importing a lot more than we need to. Maybe look into manifests?
# pylint: disable=import-outside-toplevel
def loadSolvers(inputDict,systemParametersTuple,dim,linear,legacyPaperCodeFlag,safetyFunctionF,silent):
    """
    Load the specified solver function(s) and return them, along with need parameters

    Parameters
    ----------
    inputDict : dict
        Contains the configuration parameters from the .yaml file
    systemParametersTuple : tuple
        Tuple of system parameters needed. See the model being used for details. (ie, linearModel)
    dim : int
        Number of dimensions
    linear : boolean
        Whether the system is linear or not
    legacyPaperCodeFlag : boolean
        Flag to generate available actions using the same logic as the old v1 code for the original paper
    safetyFunctionF : function
        Function used to evaluate safety of a physical state. Only used by the cbf solver, as that does not use the rewardF to determine safety
    silent : Boolean
        If true, will not print anything out while loading

    Returns
    -------
    solverFList : list
        List of solver functions to try
    solverParametersTuplesList : list
       List of solver parameters needed. Contents of each tuple are:
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
    solverNamesList : list
        List of names of solvers, for data logging
    """
    #Check for parameters first
    solverFStringList = loadRequiredParameter("solverFList",inputDict)
    #requiredParameterCheck("solverParametersListsList")
    #requiredParameterCheck("solverNames")

    #Get common parameters
    if linear:
        influenceMatrixIdx = 2
        influenceMatrix = systemParametersTuple[influenceMatrixIdx]
        numAct = len(influenceMatrix[0])
    elif dim == 3:
        positionInfluenceMatrixIdx = 0
        reactionWheelInfluenceMatrixIdx = 2
        influenceMatrix = systemParametersTuple[positionInfluenceMatrixIdx]
        numAct = len(influenceMatrix[0]) + len(systemParametersTuple[reactionWheelInfluenceMatrixIdx])
    else:
        raise NotImplementedError
    maxNumActions = loadOptionalParameter("maxNumActions",inputDict,defaultValue=min(20,numAct**2),silent=silent)
    specifiedActions = loadOptionalParameter("specifiedActions",inputDict,defaultValue=None,silent=silent)

    #Create action list (shared by solvers)
    actionRNGSeed = loadOptionalParameter("actionRNGSeed",inputDict,defaultValue=0,silent=silent)
    actionRNGKey = jaxRandom.PRNGKey(actionRNGSeed)
    if legacyPaperCodeFlag:
        from failurePy.utility.legacyPaperCode import makeAvailableActions as legacyMakeAvailableActions
        #Note this always returns actions of size 1, as the paper did, and ignores the maxNumActions flag
        availableActions = legacyMakeAvailableActions(numAct,influenceMatrix)
        print(f"legacyPaperCodeFlag set to True, overriding available actions configuration. ({len(availableActions)} total available actions)")
    else:
        availableActions = makeAvailableActions(maxNumActions,numAct,actionRNGKey,specifiedActions)

    #Loop through and add solvers to list
    solverFList,solverParametersTuplesList,solverNamesList = loadEachSolver(solverFStringList,inputDict,availableActions,silent,safetyFunctionF)



    return solverFList,solverParametersTuplesList,solverNamesList

def loadEachSolver(solverFStringList,inputDict,availableActions,silent,safetyFunctionF):
    """
    Helper method to load each solver one at a time.
    """

    solverFList = []
    solverParametersTuplesList = []
    solverNamesList = []


    for solverFString in solverFStringList:
        if solverFString.lower() in ("SFEAST".lower(), "s-FEAST".lower()):
            from failurePy.solvers.sFEAST import solveForNextAction as SFEASTSolverF
            solverFList.append(SFEASTSolverF)

            loadSFEASTParametersAndAppend(inputDict, solverParametersTuplesList, availableActions,silent=silent)

            #Add name to list
            solverNamesList.append("SFEAST")

        elif solverFString.lower() == "preSpecified".lower():
            from failurePy.solvers.preSpecifiedPolicy import PreSpecifiedPolicy

            #Instantiate object, then pass method as the "solver"
            preSpecifiedPolicyInstance = PreSpecifiedPolicy()
            solverFList.append(preSpecifiedPolicyInstance.takeNextAction)

            #No parameters needed, but availableActions needed for compatibility with pipeline, which expects access to the null action
            solverParametersTuplesList.append((availableActions,))

            #Add name to list
            solverNamesList.append("PreSpecified")
        elif solverFString.lower() == "realTimeSFEAST".lower():
            from failurePy.solvers.realTimeSFEAST import solveForNextAction as realTimeSFEASTSolverF
            solverFList.append(realTimeSFEASTSolverF)

            loadSFEASTParametersAndAppend(inputDict, solverParametersTuplesList, availableActions,silent=silent)

            #Add name to list
            solverNamesList.append("realTimeSFEAST")
        #cbf solver
        elif solverFString.lower() in ("cbf", "controlBarrierFunction".lower()):
            from failurePy.solvers.cbf import solveForNextAction as cbfSolverF
            solverFList.append(cbfSolverF)

            numActuators = len(availableActions[0])
            solverParametersTuplesList.append((numActuators,safetyFunctionF))
            #Add name to list
            solverNamesList.append("cbf")
        elif solverFString.lower() == "greedy":
            from failurePy.solvers.greedy import solveForNextAction as greedySolverF
            solverFList.append(greedySolverF)

            loadGreedyParametersAndAppend(inputDict, solverParametersTuplesList, availableActions,silent=silent)

            #Add name to list
            solverNamesList.append("greedy")
        elif solverFString.lower() == "scp":
            from failurePy.solvers.scp import solveForNextAction as scpSolverF
            solverFList.append(scpSolverF)

            loadScpParametersAndAppend(inputDict,solverParametersTuplesList,availableActions,safetyFunctionF,silent=silent)

            solverNamesList.append("scp")
        else:
            raiseSpecificationNotFoundError(solverFString,"solver")

    return solverFList,solverParametersTuplesList,solverNamesList

def loadSFEASTParametersAndAppend(inputDict, solverParametersTuplesList, availableActions, silent):
    """
    Sub method to load all the SFEAST parameters
    """
    discretization = loadOptionalParameter("discretization",inputDict,defaultValue=1,silent=silent)
    maxSimulationTime, explorationParameter,nMaxDepth,discountFactor = loadCommonParameters(inputDict,silent)

    #Add to list of parameters
    solverParametersTuplesList.append((availableActions,discretization,maxSimulationTime,explorationParameter,nMaxDepth,discountFactor))

def loadGreedyParametersAndAppend(inputDict, solverParametersTuplesList, availableActions,silent):
    """
    Sub method to load all the greedy parameters
    """
    discretization = loadOptionalParameter("discretization",inputDict,defaultValue=1,silent=silent)

    #Add to list of parameters
    solverParametersTuplesList.append((availableActions,discretization))

def loadScpParametersAndAppend(inputDict, solverParametersTuplesList, availableActions, safetyFunctionF, silent):
    """
    Sub method to load all the scp parameters
    """

    numAct = len(availableActions[0])
    horizon = loadOptionalParameter("nMaxDepth",inputDict,defaultValue=4,silent=silent)
    actuationLimit = jnp.max(availableActions)

    #Add to list of parameters
    solverParametersTuplesList.append((numAct,safetyFunctionF,horizon,actuationLimit))

def loadCommonParameters(inputDict,silent):
    "Sum method to load common solver parameters"

    maxSimulationTime = loadOptionalParameter("maxSimulationTime",inputDict,defaultValue=jnp.inf,silent=silent)
    explorationParameter = loadOptionalParameter("explorationParameter",inputDict,defaultValue=1.2,silent=silent)
    nMaxDepth = loadOptionalParameter("nMaxDepth",inputDict,defaultValue=4,silent=silent)
    discountFactor = loadOptionalParameter("discountFactor",inputDict,defaultValue=.9,silent=silent)

    return maxSimulationTime, explorationParameter,nMaxDepth,discountFactor

def makeAvailableActions(numActions,numAct,rngKey,specifiedActions=None):
    """
    Generate the actions the solvers will be allowed to consider.
    No longer scaled by testActuation size, now always 0 or 1, where 1 = full on for duration

    Parameters
    ----------
    numActions : int
        Number of actions that should be generated
    numAct : int
        The number of actuators there are
    rngKey : JAX PRNG key
        Key for the PRNG, to be used to create pseudorandomness
    specifiedActions : list or string (default=None)
        Optionally provided default actions from the yaml file

    Returns
    -------
    availableActions : array, shape(numActions,numAct)
        Array of actions that can be taken. First action is always null action
    """

    availableActions = jnp.zeros((numActions,numAct))

    #Check for pre-specified actions and set the last x actions to them, then generate less random ones
    numSpecifiedActions = 0
    if specifiedActions is not None:
        #Check if it was a string, as this is a specific type
        if isinstance(specifiedActions,str):
            specifiedActionString = ''.join(specifiedActions.lower().split())
            if  specifiedActionString in ("redundantXY".lower(),"redundantPosition".lower(),"redundantTranslation".lower()):
                specifiedActions = redundantXYActions(numAct)
            elif specifiedActionString in ("redundantXYT".lower(),"redundant".lower()):
                specifiedActions = redundantXYTActions(numAct)
            else:
                specifiedActionStringNotImplemented = f"The specified actions given by {specifiedActions} are not currently implemented."
                raise NotImplementedError(specifiedActionStringNotImplemented)
        #Treat as an array
        else:
            raise NotImplementedError

        numSpecifiedActions = len(specifiedActions)
        availableActions = availableActions.at[-numSpecifiedActions:].set(specifiedActions)

    #Check if all actions already specified (-1 because can specify 0 as first action by providing numActions-1)
    if numSpecifiedActions >= numActions - 1:
        return availableActions

    if numActions-numSpecifiedActions < 2**numAct:
        actionIdxes = jnp.array([0])
        #Force us to generate an array without 0 in it. Hacky, but should work
        while 0 in actionIdxes:
            rngKey, rngSubKey = jaxRandom.split(rngKey) # have to split in loop, or would infinite loop since key won't be updated
            actionIdxes = jaxRandom.choice(rngSubKey, 2**numAct, (numActions-1-numSpecifiedActions,),replace=False)
    else:
        actionIdxes = jnp.arange(1,2**numAct)



    #Generate actions (first action is null action)
    for iAction, actionIdx in enumerate(actionIdxes):
        action = []
        for jActuator in range(numAct):
            #Generate combinatorially (binary number encoding)
            action.append((int(actionIdx/(2**(jActuator)))%2)) #Order of operations matters here!
        availableActions = availableActions.at[iAction+1].set(jnp.array(action))

    return availableActions

def redundantXYActions(numAct):
    """
    Method that returns actions such that thrusters in each direction are both fired

    Parameters
    ----------
    numAct : int
        How many actuators we have.
        We assume actuators are redundant in pairs and there are 8, so this is just to check.

    Returns
    -------
    specifiedActions : array
        Array of the specified actions to take.

    """

    if numAct !=8:
        raiseIncompatibleSpecifications("Specified actions redundantXYActions","not exactly 8 actuators")

    return jnp.array([[1,1,0,0,0,0,0,0],
                      [0,0,1,1,0,0,0,0],
                      [0,0,0,0,1,1,0,0],
                      [0,0,0,0,0,0,1,1],])

def redundantXYTActions(numAct):
    """
    Method that returns actions such that thrusters in each direction are both fired
    and pure rotation actuations are fired

    Parameters
    ----------
    numAct : int
        How many actuators we have.
        We assume actuators are redundant in pairs and there are 8, so this is just to check.

    Returns
    -------
    specifiedActions : array
        Array of the specified actions to take.

    """

    if numAct !=8:
        raiseIncompatibleSpecifications("Specified actions redundantXYTActions","not exactly 8 actuators")

    return jnp.array([[1,1,0,0,0,0,0,0],
                      [0,0,1,1,0,0,0,0],
                      [0,0,0,0,1,1,0,0],
                      [0,0,0,0,0,0,1,1],
                      [1,0,1,0,0,0,0,0],
                      [0,1,0,1,0,0,0,0],
                      [0,0,0,0,1,0,1,0],
                      [0,0,0,0,0,1,0,1],])

def makeDistributedAvailableActions(maxNumActions,rngKey,specifiedActions,graphEdges): #Will fix when fully implemented pylint: disable=unused-argument
    """
    Generate the actions the solvers will be allowed to consider in the distributed setting.
    Here each action is an agent to point to and ping, including recursively. Need to multiplex recursive actions

    Parameters
    ----------
    numActions : int
        Number of actions that should be generated
    numAct : int
        The number of actuators there are
    rngKey : JAX PRNG key
        Key for the PRNG, to be used to create pseudorandomness
    specifiedActions : list or string (default=None)
        Optionally provided default actions from the yaml file

    Returns
    -------
    availableActions : array, shape(numActions,numAct)
        Array of actions that can be taken. First action is always null action
    """

    #Currently not implemented, returning None
    import warnings
    warnings.warn("makeDistributedAvailableActions isn't implemented, only random distributed actions will work.")

    #Get num agents by knowing every agent has at least one edge so will be listed in edges
    numAgents = len(jnp.unique(graphEdges)) #Flattens so is counting number of different entries = number of nodes!


    #Return possible pointings to make the agents move randomly
    #Will for now generate a random 100 actions from the edges

    availableActions = []
    for iAction in range(100):# pylint: disable=unused-variable
        #Generate a random new pointing
        action = jnp.zeros(numAgents)
        rngKey, rngSubKey = jaxRandom.split(rngKey)
        numReorientations = jaxRandom.randint(rngSubKey,shape=(),minval=1,maxval=numAgents)
        #Draw random edges as the reorientation
        for jReorientation in range(numReorientations): # pylint: disable=unused-variable
            rngKey, rngSubKey = jaxRandom.split(rngKey)
            edge = jaxRandom.choice(rngSubKey,graphEdges)

            rngKey, rngSubKey = jaxRandom.split(rngKey)
            #Pick which agent is turning and which is the target
            actingVertex = jaxRandom.choice(rngSubKey,jnp.array([0,1]))
            actingAgent = edge[actingVertex]
            targetAgent = edge[1-actingVertex]
            #Have the actor point at the target (yes this can overwrite, that's okay)
            action = action.at[actingAgent-1].set(targetAgent) #Need to zero index to select agent in action
        availableActions.append(action)


    return jnp.array(availableActions)
