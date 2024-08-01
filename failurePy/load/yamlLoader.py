"""
Collection of methods for loading the experiment parameters from a yaml file
Future TODO: could add more robust input checking/sanitizing, currently assuming user will do this correctly, but typos in the yamls could lead to weird behavior.
"""
import numbers
import multiprocessing as mp #For setting up multiprocessing

import jax.numpy as jnp
from failurePy.load.yamlLoaderUtilityMethods import getInputDict, loadOptionalParameter, loadRequiredParameter, raiseIncompatibleSpecifications
from failurePy.load.yamlLoaderUtilityMethods import checkExperimentParametersConsistency, checkParameter, raiseSpecificationNotFoundError
from failurePy.load.safetyLoader import loadSafetyConstrainedReward #, loadSafetyPenalizedReward #Not used by current proof
from failurePy.load.systemConstructors import loadAndBuildSingleAgentSystem
from failurePy.load.solverConstructors import loadSolvers
from failurePy.load.estimatorConstructors import loadEstimatorAndBelief, loadFaultParticleMethods


#We do a lot of conditional importing to only load in the models, solvers, estimators as needed, and bind them to shared names to pass back to pipeline.
#We could change this to be done in sub modules, or import everything and conditionally bind the names, but we'd be importing a lot more than we need to. Maybe look into manifests?
# pylint: disable=import-outside-toplevel
def loadExperimentParamsFromYaml(configFilePath,extraData=False):
    """
    Loads experiment parameters from a .yaml file. Wraps loadExperimentParams
    """
    inputDict = getInputDict(configFilePath)
    return loadExperimentParams(inputDict,extraData)

def loadExperimentParams(inputDict,extraData=False,silent=False):
    """
    Loads experiment parameters from an input dict.

    Parameters
    ----------
    configFilePath : String
        Absolute path to the config file for the experiment
    extraData : Boolean
        If true, return additional variables used to build the models and solvers
        but not needed by pipeline. Used for easier set up of ROS models
    silent : Boolean
        If true, will not print anything out while loading

    Returns
    -------
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
                    The number of repeated experiments per configuration.
                diagnosisThreshold : float
                    Level of the reward to consider high enough to return an answer.
                rngKeysOffset : int
                    Offset to the initial PRNG used in generating the initial failure states and randomness in the trials.
                    This is added to the trial number to allow for different experiments to be preformed
                initialState : array, shape(numState)
                    Initial state if any provided.
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
                safetyFunctionF : functions
                    None if no safetyMethod specified. Function that allows us to check the safety of a physicalState directly, useful for plotting or ground truth checking.
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
        String of the path to the saveDirectory for the results of the experiment. Auto determined from the experiment parameters, unless overwritten.
        Can now specify an absolute path, auto determined one is always relative
    extraDataDict : dict
        Only provided of extraData is True. Contains:
            linear : boolean
                If experiment system is linear or not
            dim : int
                Number of dimensions the system has
            sigmaW : float or array
                Standard deviation representing the process noise.
            sigmaV : float or array
                Standard deviation representing the sensor noise.
            (maybe more later)
    """

    #Top level configurations
    nSimulationsPerTreeList, multiprocessingFlag,saveTreeFlag, legacyPaperCodeFlag, clobber, mergeData, numWarmStart = loadTopLevelSpecifications(inputDict,silent)

    nExperimentSteps,nTrialsPerPoint,diagnosisThreshold,rngKeysOffset, initialState,filterDivergenceHandlingMethod = loadTrialParams(inputDict,legacyPaperCodeFlag,silent)

    nMaxComponentFailures,nMaxFailureParticles,providedFailure = loadFailureParams(inputDict, legacyPaperCodeFlag,silent)

    #Check if we are using general faults.
    if generalFaultFlag := loadOptionalParameter("generalFaultFlag",inputDict,False,alternativeString="generalFaults",silent=silent):
        failureParticleResampleF, failureParticleResampleCheckF, failureParticleInitializationF, trueFaultInInitialParticlesFlag = loadFaultParticleMethods(inputDict,silent)
        generalFaultDict = {"failureParticleResampleF": failureParticleResampleF, "failureParticleResampleCheckF":failureParticleResampleCheckF,
                            "failureParticleInitializationF":failureParticleInitializationF, "trueFaultInInitialParticlesFlag": trueFaultInInitialParticlesFlag}
        #Hybrid edge case uses the OLD fault method (no bias... could do this later?)
        if failureParticleResampleF is None:
            generalFaultFlag = False
    else:
        generalFaultDict = None

    #Check if multi agent or single agent (default)
    networkFlag = loadOptionalParameter("networkArchitecture",inputDict,defaultValue=False,silent=True)
    if networkFlag: #communication network
        futureCapability = "The distributed version of s-FEAST is intended future work, but is not currently implemented"
        raise NotImplementedError(futureCapability)

    #Single agent only for now
    #Load dimensions and linearity, from this construct system parameters. dt is pretty universal, so making an exception for it
    systemF, systemParametersTuple, physicalStateJacobianF, dim, linear, dt, sigmaW, sigmaV, numAct= loadAndBuildSingleAgentSystem(inputDict,providedFailure,generalFaultFlag,silent) # pylint: disable=invalid-name

    #Unneeded currently for any estimator, solver, or reward
    ##To get the number of states, this is length of covariance matrix, which is the -2 element of the systemParametersTuple
    #covarianceQ = systemParametersTuple[-2]
    #numState = len(covarianceQ)

    #Load estimator and belief initialization function
    estimatorF,physicalStateSubEstimatorF,physicalStateSubEstimatorSampleF, beliefInitializationF = loadEstimatorAndBelief(inputDict,linear,generalFaultFlag,silent=silent)

    #Load reward function
    rewardF, safetyFunctionF = loadRewardF(inputDict,physicalStateSubEstimatorSampleF,silent)

    #Load the solver(s) to run, and return parameters needed. Names are used for data logging
    solverFList, solverParametersTuplesList, solverNamesList = loadSolvers(inputDict,systemParametersTuple,dim,linear,legacyPaperCodeFlag,safetyFunctionF,silent)


    #Load plotting parameters (these are all optional and silent)
    plottingBounds, resolution = loadPlottingParameters(inputDict) #Already silent

    experimentParamsDict = {
        "nSimulationsPerTreeList": nSimulationsPerTreeList,
        "dt": dt,
        "nExperimentSteps": nExperimentSteps,
        "nTrialsPerPoint": nTrialsPerPoint,
        "diagnosisThreshold" : diagnosisThreshold,
        "rngKeysOffset": rngKeysOffset,
        "initialState": initialState,
        "nMaxComponentFailures": nMaxComponentFailures,
        "nMaxFailureParticles": nMaxFailureParticles,
        "providedFailure": providedFailure,
        "systemF": systemF,
        "systemParametersTuple": systemParametersTuple,
        "solverFList": solverFList,
        "solverParametersTuplesList": solverParametersTuplesList,
        "solverNamesList": solverNamesList,
        "estimatorF": estimatorF,
        "physicalStateSubEstimatorF": physicalStateSubEstimatorF,
        "physicalStateJacobianF" : physicalStateJacobianF,
        "physicalStateSubEstimatorSampleF": physicalStateSubEstimatorSampleF,
        "beliefInitializationF": beliefInitializationF,
        "rewardF": rewardF,
        "safetyFunctionF": safetyFunctionF,
        "multiprocessingFlag" : multiprocessingFlag,
        "saveTreeFlag" : saveTreeFlag,
        "numWarmStart" : numWarmStart,
        "clobber" : clobber,
        "mergeData" : mergeData,
        "plottingBounds": plottingBounds,
        "resolution": resolution,
        "networkFlag": networkFlag,
        "generalFaultDict": generalFaultDict,
        "filterDivergenceHandlingMethod": filterDivergenceHandlingMethod
    }



    #Check for consistency (this will be added as new conflicts are found)
    checkExperimentParametersConsistency(experimentParamsDict,dim,numAct)

    #Added extra field for processing
    if multiprocessingFlag:
        experimentParamsDict["virtualConfigDictList"] = getMultiprocessingVirtualConfigAndExperimentParams(inputDict, nTrialsPerPoint, rngKeysOffset)

    #Make the saved directory path. Checks if the folder already exists, and if so, if the experiments are compatible
    relativeSaveDirectoryPath = makeOrLoadSaveDirectoryPath(inputDict,experimentParamsDict,dim,linear,sigmaW, sigmaV,silent=silent)

    if extraData:
        if checkParameter("networkArchitecture",inputDict):
            raise NotImplementedError
        extraDataDict ={
            "linear" : linear,
            "dim" : dim,
            "sigmaW" : sigmaW,
            "sigmaV" : sigmaV,
        }
        return experimentParamsDict,relativeSaveDirectoryPath, extraDataDict

    return experimentParamsDict,relativeSaveDirectoryPath

def getMultiprocessingVirtualConfigAndExperimentParams(inputDict, nTrialsPerPoint, rngKeysOffset):
    """
    Packages up "virtual" config files and the needed experiment parameters to set up the data output

    Parameters
    ----------
    inputDict : dict
        Contains the configuration parameters from the .yaml file
    nTrialsPerPoint : int
        The number of repeated experiments per configuration.
    rngKeysOffset : int
        Offset to the initial PRNG used in generating the initial failure states and randomness in the trials. This is added to the trial number to allow for different experiments to be preformed

    Returns
    -------
    subExperimentsInputDictList : list
        List of "virtual" config file values to use for each sub experiment Next: pass in to yaml load (use wrapper for file vs dit input) wrapper in pipeline too for running, hook into that
    """
    numProcesses = int(mp.cpu_count() - inputDict["multiprocessingFlag"]) #When multiprocessingFlag != 0, should be num of left over cores (TODO: too overloaded?)
    #Stop further recursions!
    inputDict["multiprocessingFlag"] = False

    #We need to handle rounding errors
    numSeedsPerProcess = int(nTrialsPerPoint/numProcesses)
    numProcessesWithExtraTrial = jnp.mod(nTrialsPerPoint,numProcesses)
    #Create experiment params dict for each trial
    subExperimentsInputDictList =[]
    #Determine rngKeysOffsets per process
    #rngKeyOffsetList = []
    #numTrialsList = []
    rngKeyOffset = rngKeysOffset
    for iProcess in range(numProcesses):
        subExperimentParamsDict = inputDict.copy()
        #Add current offset
        subExperimentParamsDict["rngKeysOffset"] = rngKeyOffset
        #Adjust offset for next process
        rngKeyOffset += numSeedsPerProcess
        if iProcess < numProcessesWithExtraTrial:
            rngKeyOffset += 1
            subExperimentParamsDict["nTrialsPerPoint"] = numSeedsPerProcess+1
        else:
            subExperimentParamsDict["nTrialsPerPoint"] = numSeedsPerProcess
        subExperimentsInputDictList.append(subExperimentParamsDict)
    return subExperimentsInputDictList

def loadTopLevelSpecifications(inputDict,silent):
    """
    Load limits on simulations per tree and simulation time

    Parameters
    ----------
    inputDict : dict
        Contains the configuration parameters from the .yaml file

    Returns
    -------
    nSimulationsPerTreeList : list, len(numTrials)
        List of the number of simulations to try, if just one value, wrapped as list. This is now a list and not an array, as we want to access the bare integers
    multiprocessingFlag : int
        Wether to use multi-processing (if number is set other than 0) or not (if False/0)
    saveTreeFlag : boolean
        Whether to save the tree or not (it can be quite large, so if not visualizing, it is best to set this to false)
    legacyPaperCodeFlag : boolean
        Flag to set parameters using the same logic as the old v1 code for the original paper
    clobber : boolean
        Wether to overwrite existing data or not
    numWarmStart : int (default=0)
        Checks if we should run the solver a few times to compile first, and if so how many. Only does so on first trial. Currently only implemented for non-multiprocessing
    """
    nSimulationsPerTreeList = list(loadRequiredParameter("nSimulationsPerTree",inputDict))

    multiprocessingFlag = loadOptionalParameter("multiprocessingFlag",inputDict,False,silent=silent)

    saveTreeFlag = loadOptionalParameter("saveTreeFlag",inputDict,False,silent=silent)

    legacyPaperCodeFlag = loadOptionalParameter("legacyPaperCodeFlag",inputDict,False,silent=silent)

    clobber = loadOptionalParameter("clobber",inputDict,False,silent=True)

    mergeData = loadOptionalParameter("mergeData",inputDict,False,silent=True)

    #Not implemented yet
    if mergeData:
        #Merging data isn't implemented yet
        mergingDataNotSupported = "Merging existing data with new experiments is currently not supported"
        raise NotImplementedError(mergingDataNotSupported)

    numWarmStart = loadOptionalParameter("numWarmStart",inputDict,0,silent=True)

    return nSimulationsPerTreeList,multiprocessingFlag,saveTreeFlag,legacyPaperCodeFlag, clobber, mergeData, numWarmStart

def loadTrialParams(inputDict, legacyPaperCodeFlag,silent):
    """
    Load parameters to set up and run the various trials

    Parameters
    ----------
    inputDict : dict
        Contains the configuration parameters from the .yaml file
    legacyPaperCodeFlag : boolean
        Flag to set parameters using the same logic as the old v1 code for the original paper

    Returns
    -------
    nExperimentSteps : int
        How many time steps are in the experiment
    nTrialsPerPoint : int
        The number of repeated experiments per configuration.
    diagnosisThreshold : float
        Level of the reward to consider high enough to return an answer.
    rngKeysOffset : int
        Offset to the initial PRNG used in generating the initial failure states and randomness in the trials. This is added to the trial number to allow for different experiments to be preformed
    initialState : array, shape(numState)
        Initial state if any provided.
    """
    nExperimentSteps = loadOptionalParameter("nExperimentSteps",inputDict,20,silent=silent)

    nTrialsPerPoint = loadOptionalParameter("nTrialsPerPoint",inputDict,1, "1 trial per solver and computation limit",silent=silent)

    diagnosisThreshold = loadOptionalParameter("diagnosisThreshold",inputDict,.9,silent=silent)

    #Check if we should set using paper draft logic
    if legacyPaperCodeFlag:
        #Check if this is different than what the user specified
        if diagnosisThreshold != .81:
            print("legacyPaperCodeFlag set to True, overriding diagnosisThreshold setting to be .81")
            diagnosisThreshold = .81

    rngKeysOffset = loadOptionalParameter("rngKeysOffset",inputDict,0,silent=silent)

    #Need to check for None first, then convert to an array, otherwise we get an empty array
    initialState = loadOptionalParameter("initialState",inputDict,None,silent=silent)
    if initialState is not None:
        initialState = jnp.array(initialState)

    filterDivergenceHandlingMethod = loadOptionalParameter("filterDivergenceHandlingMethod",inputDict,None,silent=silent)

    return nExperimentSteps,nTrialsPerPoint,diagnosisThreshold,rngKeysOffset, initialState,filterDivergenceHandlingMethod

def loadFailureParams(inputDict, legacyPaperCodeFlag,silent):
    """
    Load parameters related to the set of failures we are considering

    Parameters
    ----------
    inputDict : dict
        Contains the configuration parameters from the .yaml file
    legacyPaperCodeFlag : boolean
        Flag to set max possible failures using the same logic as the old v1 code for the original paper

    Returns
    -------
    nMaxComponentFailures : int
        Maximum number of simultaneous failures of components that can be considered
    nMaxPossibleFailures : int
        Maximum number of possible failures to consider. If larger than the number of possible unique failures, all possibilities are considered
    providedFailure : array, shape(numAct+numSen) (default=None)
        Provided failure (if any) to have each trial use
    """
    nMaxComponentFailures = loadOptionalParameter("nMaxComponentFailures",inputDict,3,silent=silent)

    nMaxPossibleFailures = loadOptionalParameter("nMaxPossibleFailures",inputDict,40,silent=silent)

    #Check if we should set using paper draft logic
    if legacyPaperCodeFlag:
        #Check if this is different than what the user specified
        if nMaxPossibleFailures != 42:
            print("legacyPaperCodeFlag set to True, overriding nMaxPossibleFailures setting to be 42")
            nMaxPossibleFailures = 42
        if nMaxComponentFailures != 3:
            print("legacyPaperCodeFlag set to True, overriding nMaxComponentFailures setting to be 3")
            nMaxComponentFailures = 3

    providedFailure = loadOptionalParameter("providedFailure",inputDict,None,silent=silent)

    if providedFailure is not None:
        providedFailure = jnp.array(providedFailure)
         #Check all 0-1
        if jnp.max(providedFailure) > 1 or jnp.min(providedFailure) < 0:
            raiseIncompatibleSpecifications(f"provided failure {providedFailure}", "Fault/degradation/bias limits of [0,1]")

    return nMaxComponentFailures,nMaxPossibleFailures,providedFailure

def loadRewardF(inputDict,physicalStateSubEstimatorSampleF,silent):
    """
    Load the proper reward function and return it

    Parameters
    ----------
    inputDict : dict
        Contains the configuration parameters from the .yaml file
    physicalStateSubEstimatorSampleF : function
        Samples from the belief corresponding to this estimator

    Returns
    -------
    rewardF : function
        Reward function to evaluate the beliefs with
    safetyFunctionF : functions
        None if no safetyMethod specified. Function that allows us to check the safety of a physicalState directly, useful for plotting or ground truth checking.
    """
    if "reward" in inputDict:
        #Need to interpret reward function
        rewardFString = ''.join(inputDict["reward"].lower().split()) #String processing
        if rewardFString in ("squareSumFailureBeliefReward".lower(), "squaredFailureBeliefReward".lower(), "squareSumFailureBelief".lower(), "squaredFailureBelief".lower()):
            from failurePy.rewards.squareSumFailureBeliefReward import squareSumFailureBeliefReward
            #Check if there is a safety method specified
            if "safetyMethod" in inputDict:
                safetyMethodString = ''.join(inputDict["safetyMethod"].lower().split()) #String processing
                #HACK to vary M
                if "varyM" in inputDict:
                    rewardFs = []
                    nSimulationsPerTreeList = list(loadRequiredParameter("nSimulationsPerTree",inputDict))
                    for mSamples in nSimulationsPerTreeList:
                        nMaxDepth = loadOptionalParameter("nMaxDepth",inputDict,defaultValue=4,silent=silent)
                        rewardFs.append(loadSafetyConstrainedReward(inputDict,squareSumFailureBeliefReward,physicalStateSubEstimatorSampleF,nMaxDepth,mSamples=mSamples))
                    return rewardFs, None

                #Basic safety modulated reward, gives reward unless constraints violated, then gives 0
                if safetyMethodString in ("safetyConstrainedReward".lower(),"safetyReward".lower(),"safetyConstrained".lower(),"safetyConstraint".lower(),"safety"):
                    #Need solver depth to calculate r_0 (can only plan safety over a horizon)
                    nMaxDepth = loadOptionalParameter("nMaxDepth",inputDict,defaultValue=4,silent=silent)
                    return loadSafetyConstrainedReward(inputDict,squareSumFailureBeliefReward,physicalStateSubEstimatorSampleF,nMaxDepth)
                #Penalized reward, gives a stronger penalty if constraint violated, not just 0
                if safetyMethodString in ("safetyPenalizedReward".lower(),"safetyPenalty".lower(),"safetyPenalized".lower()):
                    #Not supported by proof
                    raiseIncompatibleSpecifications(safetyMethodString,"current safety proof")
                    #penalty = loadOptionalParameter("penalty",inputDict,1)
                    #return loadSafetyPenalizedReward(inputDict,squareSumFailureBeliefReward,physicalStateSubEstimatorSampleF,penalty)
                #Safety filter off, but computes safe region for plotting
                if safetyMethodString in ("safetyConstrainedRewardOff".lower(),"safetyRewardOff".lower(),"safetyConstrainedOff".lower(),"safetyConstraintOff".lower(),"safetyOff".lower()):
                    dummySafetyReward, safetyFunctionF = loadSafetyConstrainedReward(inputDict,squareSumFailureBeliefReward,physicalStateSubEstimatorSampleF)
                    return squareSumFailureBeliefReward, safetyFunctionF
                #This is invalid
                raiseSpecificationNotFoundError(safetyMethodString,"safetyMethod")

            return squareSumFailureBeliefReward, None
        #This is invalid
        raiseSpecificationNotFoundError(rewardFString,"reward")

    #Default behavior
    print("Defaulting to squared sum of failure belief reward function")
    from failurePy.rewards.squareSumFailureBeliefReward import squareSumFailureBeliefReward
    return squareSumFailureBeliefReward, None

def makeOrLoadSaveDirectoryPath(inputDict,experimentParamsDict,dim,linear, sigmaW, sigmaV, silent):
    """
    Function to auto generate save directory path (or load it if provided)

    Parameters
    ----------
    inputDict : dict
        Contains the configuration parameters from the .yaml file
    experimentParamsDict : dict
        Dictionary containing all the relevant experiment parameters.
    dim : int
        Number of dimensions
    linear : boolean
        Whether the system is linear or not
    sigmaW : float or array
        Standard deviation representing the process noise. If an array, only first value is used for default directory
    sigmaV : float or array
        Standard deviation representing the sensor noise. If an array, only first value is used for default directory

    Returns
    -------
    saveDirectoryPath : str
        String of the absolute path to the saveDirectory for the results of the experiment. Auto determined from the experiment parameters, unless overwritten.
    """
    saveDirectoryPath = ""

    #Autogenerate path
    #Linearity
    if linear:
        saveDirectoryPath+="Linear"
    elif linear is None: #Set to None if networked
        saveDirectoryPath+="Network"
    else:
        saveDirectoryPath+="Non-Linear"

    #Dimensionality
    saveDirectoryPath+=f"_{dim}DOF"

    #Append solver list
    for solverName in experimentParamsDict["solverNamesList"]:
        saveDirectoryPath+="_"
        saveDirectoryPath+=solverName
    #Append noise
    #First need to see if we need to get the first value out. If not a float, assume it is an array
    if not sigmaW is None and not isinstance(sigmaW, numbers.Number):
        sigmaW = sigmaW[0]
    if not sigmaV is None and not isinstance(sigmaV, numbers.Number):
        sigmaV = sigmaV[0]
    if not (sigmaW is None or sigmaV is None):
        saveDirectoryPath += f"_sigmaW_{sigmaW:.1f}_sigmaV_{sigmaV:.1f}"
    #See if a path was provided
    saveDirectoryPath = loadOptionalParameter("saveDirectoryPath",inputDict,saveDirectoryPath,silent=silent)
    #Check if absolute
    if saveDirectoryPath[0] == "/":
        return saveDirectoryPath

    #Add relative start to the front of the path
    saveDirectoryPath = "SavedData/" + saveDirectoryPath #Relative path!
    #Return relative path (as we don't want to rely on necessarily python running in the directory above SavedData)
    return saveDirectoryPath
    #return os.path.join(os.getcwd(), saveDirectoryPath) #Join path to make abs path

def loadPlottingParameters(inputDict):
    """
    Function to load in optional plotting parameters. Loads silently

    Parameters
    ----------
    inputDict : dict
        Contains the configuration parameters from the .yaml file

    Returns
    -------
    plottingBounds : array, shape(2,2) (default=None)
        Bounds of the axis
    resolution : int (default=200)
        How high of a resolution the safe zone should be drawn in.
    """
    plottingExtent =  loadOptionalParameter("plottingExtent",inputDict,None,silent=True)
    #Always a square plot
    if plottingExtent is not None:
        plottingBounds = jnp.array([[-plottingExtent,plottingExtent],[-plottingExtent,plottingExtent]])
    else:
        plottingBounds = None
    resolution =  loadOptionalParameter("resolution",inputDict,200,silent=True)
    return plottingBounds,resolution
