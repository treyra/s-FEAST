"""
Additional methods used in yaml loader that don't load any other files
"""

from pathlib import Path #Starting to use path capabilities
import yaml

#Import functions for comparison
from failurePy.models.threeDOFModel import simulateSystemWrapper as threeDOFSystemF
from failurePy.models.threeDOFGeneralFaultModel import simulateSystemWrapper as threeDOFGeneralFaultSystemF
from failurePy.solvers.preSpecifiedPolicy import PreSpecifiedPolicy

def getInputDict(configFilePath):
    """
    Helper method that gets the input dict from the configuration file path.
    Checks for duplicate key errors and converts "None" to None.

    Parameters
    ----------
    configFilePath : string
        Relative path to the config file to be used

    Returns
    -------
    inputDict : dict
        The contents of the yaml file parsed to a dictionary.
    """
    #Open file (use Path to make this compatible with our package imports)
    return loadUniqueKeyYaml(Path(configFilePath),"configuration file")

def loadUniqueKeyYaml(yamlFile,fileDescriptionString):
    """
    Helper method that loads a specified yaml file

    Parameters
    ----------
    yamlFile : File object
        File to open yaml from
    fileDescriptionString : string
        Description of file for more useful failure message

    Returns
    -------
    loadedDict : dict
        The contents of the yaml file parsed to a dictionary.
    """

    #Open file
    with yamlFile.open('r',encoding="UTF-8") as inputFile:
        try:
            inputDict = yaml.load(inputFile,UniqueKeyLoader)
            #Set "None" to None
            checkDictForNone(inputDict)

        except AssertionError as yamlError: #Check if any key provided twice
            duplicateKey = f"Error duplicate keys occur in the {fileDescriptionString}. To avoid undefined behavior, remove duplicate keys"
            raise KeyError(duplicateKey) from yamlError

    return inputDict

def checkDictForNone(inputDict):
    """
    Method that modifies a dictionary in place to change any "None" to None type.

    Parameters
    ----------
    inputDict : dict
        Dictionary to iterate through
    """

    for key in inputDict:
        if isinstance(inputDict[key],str):
            if inputDict[key].lower() == "none":
                inputDict[key] = None

def checkParameter(parameterString,inputDict):
    """
    Returns if the parameterString is a valid dict key
    """
    return parameterString in inputDict

def loadOptionalParameter(parameterString,inputDict,defaultValue, defaultMessage=None,silent=False,alternativeString=None):
    """
    Loads specified parameter, replacing it with the default value if it is not specified.
    Will inform user (can be configured) unless silent is set to True

    Alternative string allows for a different key name (used by stealthAttackPy for clearer variable names)
    """
    if not parameterString in inputDict:
        if alternativeString in inputDict:
            return inputDict[alternativeString]

        if defaultMessage is None:
            defaultMessage = str(defaultValue)
        if not silent:
            print(f"No specification for optional parameter {parameterString} provided. Defaulting to {defaultMessage}.")
        return defaultValue
    return inputDict[parameterString]

def loadRequiredParameter(parameterString,inputDict,alternativeString=None):
    """
    Tries to load a required parameter, will raise an exception if it is not found
    """
    if not parameterString in inputDict:
        #Allow for alternative name
        if not alternativeString in inputDict:
            requiredParameterNotFound(parameterString)
            #This will always raise, so no return needed
        return inputDict[alternativeString]
    return inputDict[parameterString]

def requiredParameterNotFound(parameterString):
    """
    Raises an exception informing the user if a required parameter is missing.
    """
    specificationNotProvided = f"No specification for required parameter {parameterString} provided."
    raise ValueError(specificationNotProvided)

def raiseIncompatibleSpecifications(incompatibleSpecification1,incompatibleSpecification2,extraText=None):
    """
    Raises an exception informing the user if two specifications are incompatible.
    """
    if extraText is not None:
        incompatibleSpecifications = f"{incompatibleSpecification1} is incompatible with {incompatibleSpecification2}. {extraText}"
    else:
        incompatibleSpecifications = f"{incompatibleSpecification1} is incompatible with {incompatibleSpecification2}."
    raise ValueError(incompatibleSpecifications)

def checkExperimentParametersConsistency(experimentParamsDict, dim, numAct):
    """
    Method that checks for known inconsistencies in the experimentParameters.
    This method will be expanded as more inconsistencies are identified.
    Raises error with inconsistent parameters when identified

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
                multiprocessingFlag : boolean
                    Wether to use multi-processing or not
                saveTreeFlag : boolean
                    Whether to save the tree or not (it can be quite large, so if not visualizing, it is best to set this to false)
                clobber : boolean
                    Wether to overwrite existing data or not
                plottingBounds : array, shape(2,2) (default=None)
                    Bounds of the plotting axis
                resolution : int (default=200)
                    How high of a resolution the safe zone should be drawn in when showing the safety function.
    dim : int
        Number of dimensions
    """
    #Validate solver names list length and solverFList length match
    if len(experimentParamsDict["solverFList"]) != len(experimentParamsDict["solverNamesList"]):
        raiseIncompatibleSpecifications(f"{len(experimentParamsDict['solverFList'])} solver functions", f"{len(experimentParamsDict['solverNamesList'])} solver names")

    #Initial State validation TODO: Add network validation
    if experimentParamsDict["initialState"] is not None and dim is not None: #No dim for network system
        initialState = experimentParamsDict["initialState"]
        #Check it matches system dimensions
        if len(initialState) != 2*dim:
            #Only raise if not 3DOF, as that has extra states from reaction wheels
            if experimentParamsDict["systemF"] is not threeDOFSystemF and experimentParamsDict["systemF"] is not threeDOFGeneralFaultSystemF:
                raiseIncompatibleSpecifications(f"dim: {dim}", f"initialState length of {len(initialState)}")
        #Check it matches covarianceQ (if not, probably not accounting for numWheels). covarianceQ is always second from back of systemParametersTuple
        covarianceQIdx = -2
        covarianceQ = experimentParamsDict["systemParametersTuple"][covarianceQIdx]
        if len(initialState) != len(covarianceQ):
            #Raise error, just check which one, 3DOF or otherwise
            if experimentParamsDict["systemF"] is threeDOFSystemF:
                reactionWheelInfluenceMatrixIdx = 2
                raiseIncompatibleSpecifications(f"initialState length of: {len(initialState)}", f"initial covarianceQ length of: {len(covarianceQ)}",
                    f"Check that initial states were provided for each of the {len(experimentParamsDict['systemParametersTuple'][reactionWheelInfluenceMatrixIdx])} reaction wheels.")
            raiseIncompatibleSpecifications(f"initialState length of: {len(initialState)}", f"initial covarianceQ length of: {len(covarianceQ)}",
                                                "Check that initial states were provided for each state.")

    #Check saveTreeFlag on when > 1 trials per point (lots of data)
    if experimentParamsDict["saveTreeFlag"]:
        if experimentParamsDict["nTrialsPerPoint"] > 1:
            raiseIncompatibleSpecifications("Saving each tree search",
                                            f">1 nTrialsPerPoint per point (currently {experimentParamsDict['nTrialsPerPoint']}). Too much data is generated.")

    solverNamesList = experimentParamsDict["solverNamesList"]

    #Check pre specified policy errors (a little hard to debug)
    if "PreSpecified" in solverNamesList:
        #Make a new one and check the action length! Since it is hard coded!
        preSpecifiedSolver = PreSpecifiedPolicy()
        numActionsPreSpecified = len(preSpecifiedSolver.actionList[0])
        if numAct != numActionsPreSpecified:
            raiseIncompatibleSpecifications(f"Model {experimentParamsDict['systemF']} with {numAct} actions",
                                    f"Pre specified action policy with {numActionsPreSpecified} actions specified per time step")

    #Check for baseline policies and nSimulationsPerTreeList that isn't [1]
    if "cbf" in solverNamesList or "greedy" in solverNamesList or "scp" in solverNamesList:
        if experimentParamsDict["nSimulationsPerTreeList"] != [1]:
            raiseIncompatibleSpecifications("Using any baseline solvers", f"nSimulationsPerTreeList that is not [1] (received {experimentParamsDict['nSimulationsPerTreeList']})")


    #Potential noise inconsistency to check
    #sigmaW = jnp.array(sigmaW)
    #if len(sigmaW) != dim*2:
    #    raiseIncompatibleSpecifications("sigmaW with length {}".format(len(sigmaW)),"a system with {} states".format(len(dynamicsMatrix)))
    #else:
    #    diagCovarianceQ = jnp.square(sigmaW)

#Can't change number of ancestors, as this is a 3rd party package
class UniqueKeyLoader(yaml.SafeLoader): # pylint: disable=too-many-ancestors,too-few-public-methods
    """
    Yaml loader specification that checks if there are duplicate keys and raises and assertion error if this fails.
    Inherits from SafeLoader, which protects against data injection by limiting evaluation of code.
    """

    def construct_mapping(self, node, deep=False): #Override, so can't enforce naming convention. pylint: disable=invalid-name
        """
        Overrides yaml.constructor.SafeConstructor

        Creates the mapping for loading the yaml file in.
        This is an overwrite of the base package behavior, so be vary careful making changes
        """
        mapping = []
        #Not removing unused argument or changing name to make mapping to the base method clearer.
        for key_node, value_node in node.value: # pylint: disable=unused-variable,invalid-name
            key = self.construct_object(key_node, deep=deep)
            assert key not in mapping
            mapping.append(key)
        return super().construct_mapping(node, deep)


def raiseSpecificationNotFoundError(specification,parameter="parameter"):
    """
    Method that raises error when the given specification not found in range of possible models/solvers/estimators/etc.
    """
    specificationNotFound= f"Specified {parameter}, {specification}, does not exist or is not currently implemented."
    raise ValueError(specificationNotFound)
