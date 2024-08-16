"""
Module for loading and constructing each estimator
"""

from failurePy.load.yamlLoaderUtilityMethods import loadOptionalParameter, loadRequiredParameter, raiseIncompatibleSpecifications, raiseSpecificationNotFoundError

#We do a lot of conditional importing to only load in the models, solvers, estimators as needed, and bind them to shared names to pass back to pipeline.
#We could change this to be done in sub modules, or import everything and conditionally bind the names, but we'd be importing a lot more than we need to. Maybe look into manifests?
# pylint: disable=import-outside-toplevel
def loadEstimatorAndBelief(inputDict,linear,generalFaultFlag,silent):
    """
    Load the proper estimator function and return it

    Parameters
    ----------
    inputDict : dict
        Contains the configuration parameters from the .yaml file
    linear : boolean
        Whether the system is linear or not
    silent : Boolean
        If true, will not print anything out while loading

    Returns
    -------
    estimatorF : function
        Estimator function to update the beliefs with. Takes batch of filters
    physicalStateSubEstimatorF : function
        Physical state estimator for use with the marginal filter, if any
    physicalStateSubEstimatorSampleF : function
        Function that can sample a physical state from the sub estimator
    beliefInitializationF : function
        Function that creates the initial belief
    """

    estimatorFString = loadRequiredParameter("estimator",inputDict)
    estimatorFString = ''.join(estimatorFString.lower().split()) #String processing, caps and spaces don't matter
    if estimatorFString in ("marginalizedFilter".lower(), "marginalFilter".lower(), "marginalized", "marginal"):
        #Import the marginalized filter
        from failurePy.estimators.marginalizedFilter import updateMarginalizedFilterEstimate as estimatorF
    elif  estimatorFString in ("conservativeMarginalizedFilter".lower(), "conservativeMarginalFilter".lower(),
                                "conservativeMarginalized".lower(), "conservativeMarginal".lower()):
        #Import the conservative marginalized filter
        #Check for a different updateRate
        updateRate = loadOptionalParameter("updateRate",inputDict,.25,silent=silent)
        from failurePy.estimators.conservativeMarginalizedFilter import updateMarginalizedFilterEstimate as estimatorFUnWrapped
        #Wrap with specified updateRate
        def estimatorF(action,observation,previousBeliefTuple,possibleFailures,systemF,systemParametersTuple,physicalStateSubEstimatorF,physicalStateJacobianF):
            return estimatorFUnWrapped(action,observation,previousBeliefTuple,possibleFailures,systemF,systemParametersTuple,physicalStateSubEstimatorF,
                                        physicalStateJacobianF,updateRate=updateRate)
    else:
        raiseSpecificationNotFoundError(estimatorFString,"estimator")

    #Now always look for physicalStateSubEstimator, although only marginalized filters will use it, as it won't hurt anything
    if "physicalStateSubEstimator" in inputDict:
        physicalStateSubEstimatorF,physicalStateSubEstimatorSampleF,beliefInitializationF = loadPhysicalStateSubEstimator(inputDict,linear,generalFaultFlag)
    else:
        specificationNotProvided = \
            "No specification for required parameter physicalStateSubEstimator provided. (When using the marginalized filter, the sub filter is required)"
        raise ValueError(specificationNotProvided)

    return (estimatorF,physicalStateSubEstimatorF,physicalStateSubEstimatorSampleF,beliefInitializationF)

def loadPhysicalStateSubEstimator(inputDict,linear,generalFaultFlag): #Lots of estimators. pylint: disable=too-many-branches
    """
    Load the proper sub estimator function and return it

    Parameters
    ----------
    inputDict : dict
        Contains the configuration parameters from the .yaml file
    linear : boolean
        Whether the system is linear or not

    Returns
    -------
    physicalStateSubEstimatorF : function
        Physical state estimator for use with the marginal filter, if any
    physicalStateSubEstimatorSampleF : function
        Function that can sample a physical state from the sub estimator
    beliefInitializationF : function
        Function that creates the initial belief
    """
    subEstimatorFString = ''.join(inputDict["physicalStateSubEstimator"].lower().split()) #String processing, caps and spaces don't matter
    #Load in the filters.
    #NOTE many shared components, but some are different, so need to check the type of filter and import the appropriate methods
    #KF
    if subEstimatorFString in ("kalmanFilter".lower(), "kalman", "kf"):
        #Check that the system is linear
        if not linear:
            raiseIncompatibleSpecifications("A Kalman filter","a nonlinear system")
        #Import estimator
        if generalFaultFlag:
            from failurePy.estimators.kalmanFilter import makeGeneralFaultKalmanFilter
            physicalStateSubEstimatorF = makeGeneralFaultKalmanFilter()
        else:
            from failurePy.estimators.kalmanFilter import predictAndUpdateAll as physicalStateSubEstimatorF
        from failurePy.estimators.kalmanFilterCommon import sampleFromFilter as physicalStateSubEstimatorSampleF
        #initialize to uniform belief for now
        from failurePy.estimators.beliefInitialization import uniformFailureBeliefMarginalizedKalman as beliefInitializationF
    #EKF
    elif subEstimatorFString in ("extendedKalmanFilter".lower(), "extendedKalman".lower(), "ekf"):
        #Check that the system is nonlinear
        if linear:
            raiseIncompatibleSpecifications("An extended Kalman filter","a linear system")

        #Import estimator
        if generalFaultFlag:
            from failurePy.estimators.extendedKalmanFilterLinearSensing import makeGeneralFaultExtendedKalmanFilter
            physicalStateSubEstimatorF = makeGeneralFaultExtendedKalmanFilter()
        else:
            from failurePy.estimators.extendedKalmanFilterLinearSensing import predictAndUpdateAll as physicalStateSubEstimatorF
        from failurePy.estimators.kalmanFilterCommon import sampleFromFilter as physicalStateSubEstimatorSampleF
        #initialize to uniform belief for now
        from failurePy.estimators.beliefInitialization import uniformFailureBeliefMarginalizedKalman as beliefInitializationF
    #FEJ-EKF
    elif subEstimatorFString in {"firstEstimatesJacobianExtendedKalmanFilter".lower(), "firstEstimatesJacobianExtendedKalman".lower(), "firstEstimatesExtendedKalman".lower(),
        "fejExtendedKalmanFilter".lower(),"fejkalman","fejekf"}:
        #Check that the system is nonlinear
        if linear:
            raiseIncompatibleSpecifications("An extended Kalman filter","a nonlinear system")

        #Import estimator. Note it uses a different sampleFromFilter, as needs to modify for the first estimates
        if generalFaultFlag:
            from failurePy.estimators.firstEstimatesJacobianExtendedKalmanFilterLinearSensing import makeGeneralFaultFEJExtendedKalmanFilter
            physicalStateSubEstimatorF = makeGeneralFaultFEJExtendedKalmanFilter()
        else:
            from failurePy.estimators.firstEstimatesJacobianExtendedKalmanFilterLinearSensing import predictAndUpdateAll as physicalStateSubEstimatorF
        from failurePy.estimators.firstEstimatesJacobianExtendedKalmanFilterLinearSensing import sampleFromFilter as physicalStateSubEstimatorSampleF
        #initialize to uniform belief for now
        from failurePy.estimators.beliefInitialization import uniformFailureBeliefMarginalizedFEJKalman as beliefInitializationF
    #UKF
    elif subEstimatorFString in {"unscentedKalmanFilter".lower(), "unscentedKalman".lower(), "unscented","ukf"}:
        #Not a development priority at this pint
        futureCapability = "An unscentedKalmanFilter compatible with our marginalized filter is possible future work, but is not currently implemented"
        raise NotImplementedError(futureCapability)

    else:
        raiseSpecificationNotFoundError(subEstimatorFString,"physicalStateSubEstimator")

    return physicalStateSubEstimatorF,physicalStateSubEstimatorSampleF,beliefInitializationF #Will load or raise errors. pylint: disable=possibly-used-before-assignment


def loadFaultParticleMethods(inputDict,silent):
    """
    Load the functions for updating sampling particles from fault space

    Parameters
    ----------
    inputDict : dict
        Contains the configuration parameters from the .yaml file
    silent : Boolean
        If true, will not print anything out while loading

    Returns
    -------
    failureParticleResampleF : function
        Function that resamples the particles when needed
    failureParticleResampleCheckF : function
        Function that determines if resampling is needed
    failureParticleInitializationF : function
        Function that creates the initial failure particles
    """

    #Will check if we are in the general true fault in initial set or general true fault not in initial set case. Defaulting to in set to match experiment history
    trueFaultInInitialParticlesFlag = loadOptionalParameter("trueFaultInInitialParticleSetFlag",inputDict,True,alternativeString="trueFaultKnownFlag", silent=silent)

    #Assumed that if we called this, we are using the general fault model, so some parameters are required.
    #Exception to this is if we want the hybrid binary/general fault, in which case initialization is all that matters
    particleInitializationMethodString = loadRequiredParameter("particleInitializationFunction",inputDict,alternativeString="particleInitializationMethod")
    particleInitializationMethodString = ''.join(particleInitializationMethodString.lower().split()) #String processing, caps and spaces don't matter

    if particleInitializationMethodString in ("random", "randomInitialParticles".lower(),"randomInitialFailureParticles".lower()):
        from failurePy.estimators.generalFaultSampling import randomInitialFailureParticles as failureParticleInitializationF
    elif particleInitializationMethodString in ("biased","randomBiased".lower(), "randomBiasedInitialParticles".lower(),"randomBiasedInitialFailureParticles".lower(),
                                               "biasedRandom".lower(), "biasedRandomInitialParticles".lower(),"biasedRandomInitialFailureParticles".lower()):
        from failurePy.estimators.generalFaultSampling import biasedRandomInitialFailureParticles as failureParticleInitializationF
    elif particleInitializationMethodString in ("biasedRedundant","randomBiasedRedundant".lower(), "randomBiasedRedundantInitialParticles".lower(),
                                                "randomBiasedRedundantInitialFailureParticles".lower(),"biasedRandomRedundant".lower(),
                                                "biasedRandomRedundantInitialParticles".lower(),"biasedRandomRedundantInitialFailureParticles".lower()):

        from failurePy.estimators.generalFaultSampling import biasedRandomInitialFailureParticlesRedundantBiases as failureParticleInitializationF
    elif particleInitializationMethodString in ("hybrid", "binaryToDegradationFaults".lower(),"binaryDegradation".lower(),"binary"):
        from failurePy.estimators.generalFaultSampling import binaryToDegradationFaults as failureParticleInitializationF
        #Don't need the other parameters in this case!
        return None, None, failureParticleInitializationF, trueFaultInInitialParticlesFlag
    else:
        raiseSpecificationNotFoundError(particleInitializationMethodString,"particle resample method")


    particleResampleTypeString = loadRequiredParameter("particleResampleType",inputDict,alternativeString="particleResampleMethod")
    particleResampleTypeString = ''.join(particleResampleTypeString.lower().split()) #String processing, caps and spaces don't matter

    if particleResampleTypeString in ("gaussian", "gaussianDiffusion".lower()):
        from failurePy.estimators.generalFaultSampling import gaussianDiffusion as singleParticleResampleMethodF
        faultParticleSampleSigma = loadOptionalParameter("faultParticleSampleSigma",inputDict,defaultValue=.1,silent=silent)
        from failurePy.estimators.generalFaultSampling import makeFailureParticleResampleF
        failureParticleResampleF = makeFailureParticleResampleF(singleParticleResampleMethodF,faultParticleSampleSigma)

    else:
        raiseSpecificationNotFoundError(particleResampleTypeString,"particle resample method")

    particleResampleCheckString = loadRequiredParameter("particleResampleCheck",inputDict,alternativeString="particleResampleCondition")
    particleResampleCheckString = ''.join(particleResampleCheckString.lower().split()) #String processing, caps and spaces don't matter


    if particleResampleCheckString  in ("ratio", "maxRatio".lower(),"maxRatioCheck".lower(),"maxRatioResampleCheck".lower()):
        from failurePy.estimators.generalFaultSampling import  makeMaxRatioResampleCheck as makeFailureParticleResampleCheckF
        threshold = loadOptionalParameter("failureParticleResampleThreshold",inputDict,10)
        failureParticleResampleCheckF = makeFailureParticleResampleCheckF(threshold)

    elif particleResampleCheckString in ("never", "neverSample".lower(), "noSample".lower(), "none"):
        from failurePy.estimators.generalFaultSampling import neverResampleCheck as failureParticleResampleCheckF

    else:
        raiseSpecificationNotFoundError(particleResampleCheckString,"particle resample condition")

    return failureParticleResampleF, failureParticleResampleCheckF, failureParticleInitializationF, trueFaultInInitialParticlesFlag #Will load or raise errors. pylint: disable=possibly-used-before-assignment
