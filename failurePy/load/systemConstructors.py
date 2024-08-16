"""
Module for loading and constructing each system
"""

import numbers

import jax.numpy as jnp
from jax.scipy.linalg import block_diag as blockDiag

from failurePy.load.yamlLoaderUtilityMethods import checkParameter, loadOptionalParameter, loadRequiredParameter, raiseIncompatibleSpecifications

#We do a lot of conditional importing to only load in the models, solvers, estimators as needed, and bind them to shared names to pass back to pipeline.
#We could change this to be done in sub modules, or import everything and conditionally bind the names, but we'd be importing a lot more than we need to. Maybe look into manifests?
# pylint: disable=import-outside-toplevel
def loadAndBuildSingleAgentSystem(inputDict,providedFailure,generalFaultFlag,silent):
    """
    Load which type of system we're running and initializes appropriately

    Parameters
    ----------
    inputDict : dict
        Contains the configuration parameters from the .yaml file
    providedFailure : array, shape(numAct+numSen) (default=None)
        Provided failure (if any) to have each trial use. Here for consistency checking
    generalFaultFlag : Boolean
        If we are using general fault model or not
    silent : Boolean
        If true, will not print anything out while loading

    Returns
    -------
    systemF : function
        Function reference of the system to call to run experiment
    systemParametersTuple : tuple
        Tuple of system parameters needed. See the model being used for details. (ie, linearModel)
    physicalStateJacobianF : function
        Jacobian of the model for use in estimating.
    dim : int
        Number of dimensions
    linear : boolean
        Whether the system is linear or not
    dt : float
        The time between time steps of the experiment
    sigmaW : float or array
        Standard deviation representing the process noise.
    sigmaV : float or array
        Standard deviation representing the sensor noise.
    numAct : int
        Number of actuators this system has
    """

    #Get the dimensions (2 or less = linear system)
    dim = loadRequiredParameter("dim", inputDict)

    #Check for linear flag
    #Determine default
    if dim<=3:
        default = True
        defaultMessage = "linear system as dim <= 3."
    else:
        default = False
        defaultMessage = "non-linear system as dim > 3."

    linear = loadOptionalParameter("linear",inputDict,default,defaultMessage,silent=silent)
    #Check for consistency (only inconsistent if user specified values don't match)
    if not linear and dim <=2:
        raiseIncompatibleSpecifications("dim <=2","a non-linear system")
    if linear and dim > 3:
        raiseIncompatibleSpecifications("dim > 3","a linear system")

    #Start list, will cast to a tuple (to be immutable)
    systemParametersList = []

    #Get shared parameters
    numAct = loadOptionalParameter("numAct",inputDict,None,"model default of 2 per direction",silent=silent)
    numSen = loadOptionalParameter("numSen",inputDict,None,"model default of 2 per axis",silent=silent)
    #dt is pretty universal, so making an exception for it
    dt = loadOptionalParameter("dt",inputDict,1,silent=silent) # pylint: disable=invalid-name

    #Make linear or non linear system
    if linear:
        systemParametersList, systemF, physicalStateJacobianF, sensingMatrix, numAct, numSen = importLinearSystem(inputDict,systemParametersList,dt,dim,numAct,numSen,generalFaultFlag,silent=silent)
        numWheels = 0 #Just to avoid possible referencing this when it isn't set
    elif dim == 3:
        systemParametersList, systemF, physicalStateJacobianF, sensingMatrix, numWheels, numAct, numSen = importThreeDofSystem(inputDict,systemParametersList,dt,numAct,
                                                                                                                                numSen,generalFaultFlag,silent=silent)
    else:
        raise NotImplementedError

    #Consistency check for the provided failure (if all provided)
    if providedFailure is not None:
        if generalFaultFlag and len(providedFailure) != 2*(numAct+numSen):
            raiseIncompatibleSpecifications(f"A provided general failure of length {len(providedFailure)}",
                                            f"{numAct} actuators and {numSen} sensors (expected length {2*(numAct+numSen)})")
        elif not generalFaultFlag and len(providedFailure) != numAct+numSen:
            raiseIncompatibleSpecifications(f"A provided failure of length {len(providedFailure)}",
                                            f"{numAct} actuators and {numSen} sensors (expected length {(numAct+numSen)}")

    #Currently all noise handled the same time
    covarianceQ, diagCovarianceR, sigmaW, sigmaV = loadNoise(inputDict,linear,dim,dt,sensingMatrix,numWheels)

    #Keep only the covariance matrices in systemParameters
    #Trying to enable hardware emulation
    if "hardwareEmulationFlag" in inputDict and inputDict["hardwareEmulationFlag"]:
        raise NotImplementedError #THIS DOESN'T WORK! The estimator gets the same Q!!!!! (so converge too fast)
        #systemParametersList.append(.02*covarianceQ)
        #systemParametersList.append(jnp.diag(.1*diagCovarianceR))

    systemParametersList.append(covarianceQ)
    systemParametersList.append(jnp.diag(diagCovarianceR))

    return systemF,tuple(systemParametersList), physicalStateJacobianF, dim, linear, dt, sigmaW, sigmaV, numAct

def importLinearSystem(inputDict,systemParametersList,dt,dim,numAct,numSen,generalFaultFlag,silent): #dt is pretty universal, so making an exception for it, pylint: disable=invalid-name
    """
    Sub method to import the linear system
    """
    #General or original fault model
    if generalFaultFlag:
        from failurePy.models.linearModelGeneralFaults import simulateSystemWrapper as systemF
    else:
        from failurePy.models.linearModel import simulateSystemWrapper as systemF
    from failurePy.models.linearModel import makeDynamicsMatrix, makeInfluenceMatrix, makeSensingMatrix, makeStateTransitionMatrices

    physicalStateJacobianF = None

    #Make systemParametersList
    ##########################
    #Make the base matrices

    #Get s/c mass for scaling influence matrices
    spacecraftMass = loadOptionalParameter("spacecraftMass",inputDict,defaultValue=1,silent=silent)

    #Make dynamics and influence matrices
    dynamicsMatrix = makeDynamicsMatrix(dim)
    influenceMatrix = makeInfluenceMatrix(dim,spacecraftMass,numAct)
    numAct = len(influenceMatrix[0])
    #Make sensing matrices
    sensingMatrix = makeSensingMatrix(dim,numSen)
    numSen = len(sensingMatrix)

    stateTransitionMatrix,controlTransitionMatrix = makeStateTransitionMatrices(dynamicsMatrix,dt,influenceMatrix)
    systemParametersList.append(stateTransitionMatrix)
    systemParametersList.append(controlTransitionMatrix)
    systemParametersList.append(sensingMatrix)

    return systemParametersList, systemF, physicalStateJacobianF, sensingMatrix, numAct, numSen

def importThreeDofSystem(inputDict,systemParametersList,dt,numAct,numSen,generalFaultFlag,silent): #dt is pretty universal, so making an exception for it, pylint: disable=invalid-name
    """
    Sub method to import the 3DOF system
    """
    #General or original fault model
    if generalFaultFlag:
        from failurePy.models.threeDOFGeneralFaultModel import simulateSystemWrapper as systemF
        from failurePy.models.threeDOFGeneralFaultModel import dynamicsJacobianWrapper as physicalStateJacobianF
    else:
        from failurePy.models.threeDOFModel import simulateSystemWrapper as systemF
        from failurePy.models.threeDOFModel import dynamicsJacobianWrapper as physicalStateJacobianF
    from failurePy.models.threeDOFModel import makeInfluenceMatrices, makeSensingMatrix, makeCalibratedInfluenceMatricesNoWheels

    #Need to check for calibrated parameters or use model
    if checkParameter("positionInfluenceMatrix",inputDict) and checkParameter("rotationInfluenceMatrix",inputDict) and  checkParameter("reactionWheelInfluenceMatrix",inputDict):
        positionInfluenceMatrix = loadRequiredParameter("positionInfluenceMatrix",inputDict)
        rotationInfluenceMatrix = loadRequiredParameter("rotationInfluenceMatrix",inputDict)
        reactionWheelInfluenceMatrix = loadRequiredParameter("reactionWheelInfluenceMatrix",inputDict)
        numWheels = len(reactionWheelInfluenceMatrix)

    elif checkParameter("positionInfluenceMatrix",inputDict) or checkParameter("rotationInfluenceMatrix",inputDict) or  checkParameter("reactionWheelInfluenceMatrix",inputDict):
        incompatibleSpecifications = "If any calibrated influence matrices are provided, all arrays must be provided. Mixing modeled and calibrated behavior currently unsupported"
        raise ValueError(incompatibleSpecifications)

    elif checkParameter("positionInfluence",inputDict) and checkParameter("rotationInfluence",inputDict):
        positionInfluence = loadRequiredParameter("positionInfluence",inputDict)
        rotationInfluence = loadRequiredParameter("rotationInfluence",inputDict)
        positionInfluenceMatrix, rotationInfluenceMatrix = makeCalibratedInfluenceMatricesNoWheels(positionInfluence,rotationInfluence)
        #OVERRIDES numWheels
        numWheels = 0
        reactionWheelInfluenceMatrix = jnp.ones(numWheels)
    elif checkParameter("positionInfluence",inputDict) and checkParameter("rotationInfluence",inputDict) and checkParameter("reactionWheelInfluence",inputDict):
        raise NotImplementedError

    elif checkParameter("positionInfluence",inputDict) or checkParameter("rotationInfluence",inputDict) or checkParameter("reactionWheelInfluence",inputDict):
        incompatibleSpecifications = "positionInfluence and rotationInfluence must be supplied together and with reactionWheelInfluence. " +\
                                    "Mixing modeled and calibrated behavior currently unsupported"
        raise ValueError(incompatibleSpecifications)

    else:
        #Get s/c mass for scaling influence matrices
        spacecraftMass = loadOptionalParameter("spacecraftMass",inputDict,defaultValue=1,silent=silent)

        #Get s/c inertia for scaling influence matrices
        spacecraftMoment = loadOptionalParameter("spacecraftMoment",inputDict,defaultValue=1,silent=silent)

        #Get num wheels
        numWheels = loadOptionalParameter("numWheels",inputDict,2,silent=silent)

        #Get lever arms
        leverArm = loadOptionalParameter("thrusterLeverArm",inputDict,.4,silent=silent)

        #Get wheel inertia for influence
        reactionWheelMoment = loadOptionalParameter("reactionWheelMoment",inputDict,defaultValue=None, defaultMessage=".05 kg m^2 per wheel",silent=silent)
        if isinstance(reactionWheelMoment, numbers.Number):
            reactionWheelMoment = reactionWheelMoment*jnp.ones(numWheels)

        #Make the actuation and sensing matrices
        positionInfluenceMatrix, rotationInfluenceMatrix, reactionWheelInfluenceMatrix = makeInfluenceMatrices(spacecraftMass,spacecraftMoment,leverArm,numWheels,numAct,reactionWheelMoment)

    numAct = len(positionInfluenceMatrix[0]) + len(reactionWheelInfluenceMatrix)

    sensingMatrix = makeSensingMatrix(numWheels,numSen)
    numSen = len(sensingMatrix)

    #Check for errors (note if calibrated matrices given, we won't enter here by construction)
    if len(reactionWheelInfluenceMatrix) != numWheels:
        raiseIncompatibleSpecifications(f"{numWheels} reaction wheels",f"reactionWheelMoment array of length {len(reactionWheelMoment)}")

    positionCoefficientOfFriction = loadOptionalParameter("positionCoefficientOfFriction",inputDict,0,silent=silent)
    rotationalCoefficientOfFriction = loadOptionalParameter("rotationalCoefficientOfFriction",inputDict,0,silent=silent)

    systemParametersList.append(positionInfluenceMatrix)
    systemParametersList.append(rotationInfluenceMatrix)
    systemParametersList.append(reactionWheelInfluenceMatrix)
    systemParametersList.append(positionCoefficientOfFriction)
    systemParametersList.append(rotationalCoefficientOfFriction)
    systemParametersList.append(sensingMatrix)
    systemParametersList.append(dt)

    return systemParametersList, systemF, physicalStateJacobianF, sensingMatrix, numWheels, numAct, numSen

def loadNoise(inputDict,linear,dim,dt,sensingMatrix,numWheels): #dt is pretty universal, so making an exception for it, pylint: disable=invalid-name
    """
    Sub method to load the noise parameters

    Checks if there is a single sigma value
    Produce block diagonal process noise to match model from Bar-Shalom. "Estimation with Applications To Tracking and Navigation".  John Wiley & Sons, 2001. Page 270.
    Notice power spectral density = variance here
    """
    if "sigma" in inputDict:
        sigma = inputDict["sigma"]
        #We think of our noise in terms of sigmaW and sigmaV, so set them to be the same explicitly
        sigmaW = sigma
        sigmaV = sigma
        covarianceQBlock = sigmaW**2*jnp.array([[(dt**3)/3, (dt**2)/2],
                                               [(dt**2)/2, dt       ]]) #We'll assume same block for each dimension
        covarianceQ = jnp.kron(jnp.eye(dim,dtype=int),covarianceQBlock)
        #Measurement noise assumed to be independent
        diagCovarianceR = sigmaV**2*jnp.ones(len(sensingMatrix))
    elif "sigmaW" in inputDict and "sigmaV" in inputDict:
        sigmaW = inputDict["sigmaW"]
        sigmaV = inputDict["sigmaV"]
        #Check if given array or scalar
        if isinstance(sigmaW, numbers.Number):
            covarianceQBlock = sigmaW**2*jnp.array([[(dt**3)/3, (dt**2)/2],
                                               [(dt**2)/2, dt       ]]) #We'll assume same block for each dimension
            #Stack covarianceQBlock into block matrix
            covarianceQ = jnp.kron(jnp.eye(dim,dtype=int),covarianceQBlock)
        #Otherwise assume it's an array
        elif len(sigmaW) == dim:
            covarianceQBlocks = []
            for iDim in range(dim):
                covarianceQBlocks.append(sigmaW[iDim]**2*jnp.array([[(dt**3)/3, (dt**2)/2],
                                               [(dt**2)/2, dt       ]]))
            #Make covariance out of blocks (unpack first)
            covarianceQ = blockDiag(*covarianceQBlocks)
        else:
            #Currently not checking
            notImplemented = "Setting more complicated process noises than per dimensions is currently not supported."
            raise NotImplementedError(notImplemented)

        if isinstance(sigmaV, numbers.Number):
            diagCovarianceR = sigmaV**2*jnp.ones(len(sensingMatrix))
        #Otherwise assume it's an array, return only first value for save directories
        else:
            sigmaV = jnp.array(sigmaV)
            if len(sigmaV) != len(sensingMatrix):
                raiseIncompatibleSpecifications(f"sigmaV with length {len(sigmaV)}",f"a system with {len(sensingMatrix)} sensors")
            else:
                diagCovarianceR = jnp.square(sigmaV)
    else:
        specificationNotProvided = "No specification for required parameter sigma (or sigmaW AND sigmaV) provided."
        raise ValueError(specificationNotProvided)

    #Need to add noise for wheels (will make infinitesimal, needs to be non zero though for estimation)
    if not linear:
        wheelCovarianceQ = jnp.diag(.001* jnp.ones(numWheels))
        covarianceQ = blockDiag(covarianceQ,wheelCovarianceQ)
    return covarianceQ, diagCovarianceR, sigmaW, sigmaV #Will load or raise errors. pylint: disable=possibly-used-before-assignment
