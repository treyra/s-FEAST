"""
Module to handle loading safety as part of the reward, as there a lot of configurable options.
"""

#We do a lot of conditional importing to only load in the models, solvers, estimators as needed, and bind them to shared names to pass back to pipeline.
#We could change this to be done in sub modules, or import everything and conditionally bind the names, but we'd be importing a lot more than we need to. Maybe look into manifests?
# pylint: disable=import-outside-toplevel
import jax.numpy as jnp

#Import all of the possible constraints, as it'll be messy to import them while looping through
from failurePy.rewards.safetyConstraint import makeCircularObstacleConstraintF,makeCircularSafeZoneConstraintF,makeLinearObstacleConstraintF,makeLinearSafeZoneConstraintF
from failurePy.load.yamlLoaderUtilityMethods import loadOptionalParameter, raiseSpecificationNotFoundError

def loadSafetyConstrainedReward(inputDict,rewardF,physicalStateSubEstimatorSampleF,nMaxDepth=4,mSamples=100):
    """
    Top-level method for loading the safety constrained rewards.
    Specificity done in sub-methods

    Parameters
    ----------
    inputDict : dict
        Contains the configuration parameters from the .yaml file
    rewardF : function
        Reward function that should accept a beliefTuple and an rngKey as arguments.
        This reward should give positive rewards, as safety violations will return as 0 (no reward)
    physicalStateSubEstimatorSampleF : function
        Samples from the belief corresponding to this estimator
    nMaxDepth : int (default=4)
        Maximum depth of the tree. This is the horizon we need to be safe over (can't be safe over a longer horizon, as we don't search over it)
    mSamples : int (default=100)
        The number of samples to take to approximately evaluate the safety condition at each belief

    Returns
    -------
    safetyConstrainedReward : function
        Reward function to evaluate the beliefs with, with safety constraints enforced
    safetyFunctionF : functions
       Function that allows us to check the safety of a physicalState directly, useful for plotting or ground truth checking.
    """

    from failurePy.rewards.safetyConstraint import makeSafetyConstrainedReward as safetyConstrainedRewardFactoryF

    safetyFunctionEvaluationF, safetyFunctionF = loadSafetyModulatedRewardComponents(inputDict,physicalStateSubEstimatorSampleF,mSamples)

    return safetyConstrainedRewardFactoryF(rewardF,safetyFunctionEvaluationF,nMaxDepth), safetyFunctionF

def loadSafetyPenalizedReward(inputDict,rewardF,physicalStateSubEstimatorSampleF, penalty=1):
    """
    Top-level method for loading the safety penalized rewards.
    Specificity done in sub-methods

    Parameters
    ----------
    inputDict : dict
        Contains the configuration parameters from the .yaml file
    rewardF : function
        Reward function that should accept a beliefTuple and an rngKey as arguments.
        This reward should give positive rewards, as safety violations will return as 0 (no reward)
    physicalStateSubEstimatorSampleF : function
        Samples from the belief corresponding to this estimator
    penalty : float (default=1)
        This is how big the penalty to the reward is.

    Returns
    -------
    safetyConstrainedReward : function
        Reward function to evaluate the beliefs with, with safety constraints enforced
    safetyFunctionF : functions
       Function that allows us to check the safety of a physicalState directly, useful for plotting or ground truth checking.
    """

    from failurePy.rewards.safetyConstraint import makeSafetyPenalizedReward as makeSafetyPenalizedRewardFactoryF

    safetyFunctionEvaluationF, safetyFunctionF = loadSafetyModulatedRewardComponents(inputDict,physicalStateSubEstimatorSampleF)

    return makeSafetyPenalizedRewardFactoryF(rewardF,safetyFunctionEvaluationF,penalty), safetyFunctionF


def loadSafetyModulatedRewardComponents(inputDict,physicalStateSubEstimatorSampleF,numSamples=100):
    """
    Top-level method for loading the rewards that depend on the safety criteria.
    Specificity done in sub-methods

    Parameters
    ----------
    inputDict : dict
        Contains the configuration parameters from the .yaml file
    physicalStateSubEstimatorSampleF : function
        Samples from the belief corresponding to this estimator
    numSamples : int (default=100)
        The number of samples to take to approximately evaluate the safety condition at each belief

    Returns
    -------
    safetyFunctionEvaluationF : functions
       Method of checking the safety constraints
    safetyFunctionF : functions
       Function that allows us to check the safety of a physicalState directly, useful for plotting or ground truth checking.
    """

    #Load method for evaluating safety constraints
    safetyFunctionEvaluationFactoryF = loadSafetyFunctionEvaluationF(inputDict)

    #Load type of safety constraints (inequality, etc.)
    safetyFunctionFactoryF = loadSafetyFunctionF(inputDict)

    #Load the constraint functions
    constraintFTuple = loadConstraintFunctionTuple(inputDict)

    #Now build up the constrained reward function
    safetyFunctionF = safetyFunctionFactoryF(constraintFTuple)

    #Load allowable failure chance
    allowableFailureChance = loadOptionalParameter("allowableFailureChance",inputDict,defaultValue=.05)

    safetyFunctionEvaluationF = safetyFunctionEvaluationFactoryF(safetyFunctionF,physicalStateSubEstimatorSampleF,numSamples,allowableFailureChance)

    return safetyFunctionEvaluationF,safetyFunctionF


def loadSafetyFunctionEvaluationF(inputDict):
    """
    Sub method to load safetyFunctionEvaluationF

    Parameters
    ----------
    inputDict : dict
        Contains the configuration parameters from the .yaml file

    Returns
    -------
    safetyFunctionEvaluationF : function
        Function that implements a method to evaluate the safety constraints
    """
    #Load how to check the safety constraints
    if "safetyFunctionEvaluation" in inputDict:
        #Need to interpret reward function
        safetyConstraintFString = ''.join(inputDict["safetyFunctionEvaluation"].lower().split()) #String processing
        if safetyConstraintFString in ("probabilisticSafetyConstraint".lower(), "probSafetyConstraint".lower(), "probabilistic", "probabilisticSafety".lower(),
                                       "probabilisticSafetyFunction".lower(), "probSafetyFunction".lower(), "probabilisticSafetyFunctionEvaluation".lower()):
            from failurePy.rewards.safetyConstraint import makeProbabilisticSafetyFunctionEvaluation as safetyFunctionEvaluationFactoryF
        elif safetyConstraintFString in ("chebyshevInequalitySafetyFunction".lower(), "chebyshevIneqSafetyFunction".lower(),
                                        "chebyshevSafetyFunction".lower(),"chebyshevInequalitySafetyConstraint".lower(),
                                        "chebyshevIneqSafetyConstraint".lower(),"chebyshevSafetyConstraint".lower(),"chebyshevInequality".lower(),
                                        "chebyshevIneq".lower(),"chebyshev","chebyshevInequalitySafetyFunctionEvaluation".lower()):
            from failurePy.rewards.safetyConstraint import makeChebyshevIneqSafetyFunctionEvaluation as safetyFunctionEvaluationFactoryF
        elif safetyConstraintFString in ("probabilisticAlphaSafetyFunctionEvaluation".lower(),"probabilisticAlpha".lower()):
            from failurePy.rewards.safetyConstraint import makeProbabilisticAlphaSafetyFunctionEvaluation as safetyFunctionEvaluationFactoryF
        elif safetyConstraintFString in ("filterMeansSafetyConstraint".lower(), "meansSafetyConstraint".lower(),"meanSafetyConstraint".lower(),
                                        "filterMeansSafety".lower(), "meansSafety".lower(),"meanSafety".lower()):
            raise NotImplementedError
        else: #This is invalid
            raiseSpecificationNotFoundError(safetyConstraintFString,"safetyFunctionEvaluation")

    else:
        #Default behavior
        print("Defaulting to probabilistic safety constraint")
        from failurePy.rewards.safetyConstraint import makeProbabilisticSafetyFunctionEvaluation as safetyFunctionEvaluationFactoryF

    return safetyFunctionEvaluationFactoryF #Will load or raise errors. pylint: disable=possibly-used-before-assignment

def loadSafetyFunctionF(inputDict):
    """
    Sub method to load safetyFunctionF

    Parameters
    ----------
    inputDict : dict
        Contains the configuration parameters from the .yaml file

    Returns
    -------
    safetyFunctionF : function
        A function that implements the specified type of safety constraints.
        Evaluates the provided safety constraints against the type of condition (inequality, etc.)
    """

    #Load the constraints condition type (may remove later if we don't implement other types)
    if "safetyFunction" in inputDict:
        safetyFunctionFString = ''.join(inputDict["safetyFunction"].lower().split()) #String processing
        if safetyFunctionFString in ("booleanInequalitySafetyFunction".lower(),"booleanSafetyFunction".lower(),"booleanInequalitySafety".lower(),
                                     "inequalitySafetyFunction".lower(), "inequalitySafetyConstraint".lower(), "inequalitySafety".lower(),
                                     "booleanSafety".lower(),"boolean","inequality","inequalitySafetyConstraintCondition".lower(),):
            from failurePy.rewards.safetyConstraint import makeBooleanInequalitySafetyFunctionF as safetyFunctionFactoryF
        elif safetyFunctionFString in ("worstCaseInequalitySafetyFunction".lower(), "worstInequalitySafetyFunction".lower(),
                                        "worstCaseInequalitySafety".lower(),"worstInequalitySafety".lower(),"worstCaseInequality".lower(),
                                        "worstInequality".lower(),"worstCase".lower(),"worst"):
            from failurePy.rewards.safetyConstraint import makeWorstInequalitySafetyFunctionF as safetyFunctionFactoryF
        else:
            #This is invalid
            raiseSpecificationNotFoundError(safetyFunctionFString,"safetyFunction")
    else:
        #Default behavior
        print("Defaulting to inequality safety constraint conditions")
        from failurePy.rewards.safetyConstraint import makeBooleanInequalitySafetyFunctionF as safetyFunctionFactoryF

    return safetyFunctionFactoryF #Will load or raise errors. pylint: disable=possibly-used-before-assignment

def loadConstraintFunctionTuple(inputDict):
    """
    Sub method to load constraintFTuple

    Parameters
    ----------
    inputDict : dict
        Contains the configuration parameters from the .yaml file

    Returns
    -------
    constraintFTuple : tuple
        A function that implements the specified type of safety constraints.
        Evaluates the provided safety constraints against the type of condition (inequality, etc.)
    """

    #Load the constraints condition type (may remove later if we don't implement other types)
    if "safetyConstraints" in inputDict:
        #Get list of specified constraints
        safetyConstraintsList = inputDict["safetyConstraints"]
        #Loop through and interpret each
        constraintFList = []
        #Label indexes
        constraintNameIdx = 0
        constraintParamsIdx = 1
        for safetyConstraint in safetyConstraintsList:

            constraintName = ''.join(safetyConstraint[constraintNameIdx].lower().split()) #String processing
            #Obstacle Constraint
            if constraintName in ("circularObstacleConstraint".lower(),"circularObstacle".lower()):
                radius = safetyConstraint[constraintParamsIdx][0]
                center = jnp.array(safetyConstraint[constraintParamsIdx][1])
                #Get constraint function from function factory.
                constraintFList.append(makeCircularObstacleConstraintF(radius,center))
            #Safe zone constraint
            elif constraintName in ("circularSafeZoneConstraint".lower(),"circularSafeZone".lower(),"circularSafe".lower()):
                radius = safetyConstraint[constraintParamsIdx][0]
                center = jnp.array(safetyConstraint[constraintParamsIdx][1])
                #Get constraint function from function factory.
                constraintFList.append(makeCircularSafeZoneConstraintF(radius,center))
            #Linear obstacle constraint
            elif constraintName in ("linearObstacleConstraint".lower(),"linearObstacle".lower()):
                normalMatrix = jnp.array(safetyConstraint[constraintParamsIdx][0])
                offsetVector = jnp.array(safetyConstraint[constraintParamsIdx][1])
                #Get constraint function from function factory.
                constraintFList.append(makeLinearObstacleConstraintF(normalMatrix,offsetVector))
            #Linear safe zone constraint
            elif constraintName in ("linearSafeZoneConstraint".lower(),"linearSafeZone".lower(),"linearSafe".lower()):
                normalMatrix = jnp.array(safetyConstraint[constraintParamsIdx][0])
                offsetVector = jnp.array(safetyConstraint[constraintParamsIdx][1])
                #Get constraint function from function factory.
                constraintFList.append(makeLinearSafeZoneConstraintF(normalMatrix,offsetVector))
            #No match
            else:
                constraintNotImplemented=f"Specified constraint {safetyConstraint[constraintNameIdx]} is currently not implemented"
                raise NotImplementedError(constraintNotImplemented)

    else:
        #Default behavior
        print("Defaulting to radius 10 safe zone centered at the origin")

        #Get constraint function from function factory.
        constraintFList = [makeCircularSafeZoneConstraintF(10,jnp.array([0,0]))]

    #Cast to a hashable form then return (for jitting). Lists aren't hashable because they are mutable.
    return tuple(constraintFList)
