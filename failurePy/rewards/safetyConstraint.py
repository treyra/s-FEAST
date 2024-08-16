"""
Module that implements safety constraints as modifications for the tree reward function
"""
from functools import partial

import jax
import jax.numpy as jnp
from jax import random as jaxRandom

@partial(jax.jit, static_argnames=['rewardF','safetyFunctionEvaluationF'])
def safetyConstrainedReward(beliefTuple,rngKey,rewardF,safetyFunctionEvaluationF,safetyRewardR0):
    """
    Takes in a belief and gives a reward unless it violates the safety constraint.

    Method used by s-FEAST in our paper

    Parameters
    ----------
    beliefTuple : tuple
        The belief is represented by an array of failureWeights and the filter array corresponding to each possibleFailure. Contains:
            failureWeights : array, shape(maxPossibleFailures)
                Normalized weighting of the relative likelihood of each failure
            filters : array
                Array of conditional filters on the physical state
    rngKey : JAX PRNG key
        Key for the PRNG, to be used to create pseudorandomness (for some safety constraints)
    rewardF : function
        Reward function that should accept a beliefTuple and an rngKey as arguments.
        This reward should give positive rewards, as safety violations will return as 0 (no reward)
    safetyFunctionEvaluationF : function
        Safety constraint function that should accept a beliefTuple and an rngKey as arguments.
        Should return a boolean value of True (1) if we are safe or False (0) if we violated the constraint
    safetyRewardR0 : float
        r0. Constant that guarantees any safe trajectory has a higher reward than any unsafe trajectory

    Returns
    -------
    reward : float
        Reward for how certain we are. Note this will always be 0-1 because
        the belief distribution is a probability distribution
    """

    #Returns 0 if we violate the constraint, as safetyFunctionEvaluationF returns a boolean value
    return safetyFunctionEvaluationF(beliefTuple,rngKey) * (safetyRewardR0 + (1-safetyRewardR0) * rewardF(beliefTuple)) #the rewardF under this will not be probabilistic

def makeSafetyConstrainedReward(rewardF,safetyFunctionEvaluationF,nMaxDepth):
    """
    Function factory to make probabilisticSafetyFunctionEvaluation functions.
    We shouldn't run into late binding problems here, but I think this is easier to read

    Parameters
    ----------
    rewardF : function
        Reward function that should accept a beliefTuple and an rngKey as arguments.
        This reward should give positive rewards, as safety violations will return as 0 (no reward)
    safetyFunctionEvaluationF : function
        Safety constraint function that should accept a beliefTuple and an rngKey as arguments.
        Should return a boolean value of True (1) if we are safe or False (0) if we violated the constraint
    nMaxDepth : int
        Maximum depth of the tree. This is the horizon we need to be safe over (can't be safe over a longer horizon, as we don't search over it)

    Returns
    -------
    safetyConstrainedRewardF : function
        Wrapper around makeSafetyConstrainedReward with the specified rewardF,safetyConstraintF
    """
    def safetyConstrainedRewardF(beliefTuple,rngKey):
        safetyRewardR0 = nMaxDepth/(nMaxDepth+1)
        return safetyConstrainedReward(beliefTuple,rngKey,rewardF,safetyFunctionEvaluationF,safetyRewardR0)
    return safetyConstrainedRewardF

@partial(jax.jit, static_argnames=['rewardF','safetyFunctionEvaluationF'])
def safetyPenalizedReward(beliefTuple,rngKey,rewardF,safetyFunctionEvaluationF,penalty=1):
    """
    Takes in a belief and gives a reward unless it violates the safety constraint.
    If the constraint is violated it also assigns a penalty

    Parameters
    ----------
    beliefTuple : tuple
        The belief is represented by an array of failureWeights and the filter array corresponding to each possibleFailure. Contains:
            failureWeights : array, shape(maxPossibleFailures)
                Normalized weighting of the relative likelihood of each failure
            filters : array
                Array of conditional filters on the physical state
    rngKey : JAX PRNG key
        Key for the PRNG, to be used to create pseudorandomness (for some safety constraints)
    rewardF : function
        Reward function that should accept a beliefTuple and an rngKey as arguments.
        This reward should give positive rewards, as safety violations will return as 0 (no reward)
    safetyFunctionEvaluationF : function
        Safety constraint function that should accept a beliefTuple and an rngKey as arguments.
        Should return a boolean value of True (1) if we are safe or False (0) if we violated the constraint
    penalty : float (default=1)
        This is how big the penalty to the reward is.

    Returns
    -------
    reward : float
        Reward for how certain we are. Note this will always be 0-1 because
        the belief distribution is a probability distribution
    """

    safetyFlag = safetyFunctionEvaluationF(beliefTuple,rngKey)

    #Compute penalty (note it is zeroed out if the safeFlag return from safetyFunctionEvaluationF is 1) and return it instead of safetyFlag=0
    return -penalty * (1-safetyFlag) + safetyFlag * rewardF(beliefTuple) #the rewardF under this will not be probabilistic

def makeSafetyPenalizedReward(rewardF,safetyFunctionEvaluationF,penalty=1):
    """
    Function factory to make probabilisticSafetyFunctionEvaluation functions.
    We shouldn't run into late binding problems here, but I think this is easier to read

    Parameters
    ----------
    rewardF : function
        Reward function that should accept a beliefTuple and an rngKey as arguments.
        This reward should give positive rewards, as safety violations will return as 0 (no reward)
    safetyFunctionEvaluationF : function
        Safety constraint function that should accept a beliefTuple and an rngKey as arguments.
        Should return a boolean value of True (1) if we are safe or False (0) if we violated the constraint
    penalty : float (default=1)
        This is how big the penalty to the reward is.

    Returns
    -------
    safetyPenalizedRewardF : function
        Wrapper around makeSafetyPenalizedReward with the specified rewardF,safetyConstraintF
    """
    def safetyPenalizedRewardF(beliefTuple,rngKey):
        return safetyPenalizedReward(beliefTuple,rngKey,rewardF,safetyFunctionEvaluationF,penalty)
    return safetyPenalizedRewardF

def filterMeansSafetyFunctionEvaluation(beliefTuple,rngKey,safetyFunctionF):
    """
    Function that defines a safety constraint based on the means of the belief tuple filters.

    Parameters
    ----------
    beliefTuple : tuple
        The belief is represented by an array of failureWeights and the filter array corresponding to each possibleFailure. Contains:
            failureWeights : array, shape(maxPossibleFailures)
                Normalized weighting of the relative likelihood of each failure
            filters : array
                Array of conditional filters on the physical state
    rngKey : JAX PRNG key
        Key for the PRNG, to be used to create pseudorandomness (for some safety constraints)
    safetyFunctionF : function
        Function that represents the conditions the physical state must satisfy for safety.
        Must accept a physical state as an input.
    """
    raise NotImplementedError

#Jitting: numSamples needs to be a static argument (prettySure)
@partial(jax.jit, static_argnames=['safetyFunctionF','physicalStateSubEstimatorSampleF','numSamples'])
def probabilisticAlphaSafetyFunctionEvaluation(beliefTuple,rngKey,safetyFunctionF,physicalStateSubEstimatorSampleF,numSamples=100,alpha=.95):
    """
    Function that evaluates a safety constraint based using samples from the belief to determine if the belief is alpha safe.
    At high number of samples, acts as "ground truth" of alpha-safety of a belief

    Parameters
    ----------
    beliefTuple : tuple
        The belief is represented by an array of failureWeights and the filter array corresponding to each possibleFailure. Contains:
            failureWeights : array, shape(maxPossibleFailures)
                Normalized weighting of the relative likelihood of each failure
            filters : array
                Array of conditional filters on the physical state
    rngKey : JAX PRNG key
        Key for the PRNG, to be used to create pseudorandomness (for some safety constraints)
    safetyFunctionF : function
        Function that represents the conditions the physical state must satisfy for safety.
        Must accept a physical state as an input.
    physicalStateSubEstimatorSampleF : function
        Samples the physical state from the belief
    numSamples : int (default=100)
        Number of samples to be drawn
    alpha : float (default)
        Threshold for safety. ie samples must have % safety > alpha

    Returns
    -------
    safeFlag : boolean
        1 if safety constraints satisfied, 0 if not
    """

    batchSafetyReturn = batchSampleSafetyHelperMethod(beliefTuple,rngKey,safetyFunctionF,physicalStateSubEstimatorSampleF,numSamples)
    #Dealing with non-boolean returns, because Chebyshev uses non-boolean returns and we want to be compatible
    #Negative return is safe! #Not used in our paper, we instead set to 0 in a different function
    batchSafetyReturn = - jnp.sign(batchSafetyReturn)

    #Now determine if we are above the threshold (alpha) percentage of safe samples
    safetyPercent = jnp.sum(batchSafetyReturn)/numSamples
    safetyFlag = jnp.sign(safetyPercent-alpha) #Check if safetyPercent > alpha
    #Deal with possibility of equality (okay) and sign being negative
    safetyFlag = jnp.sign(.5*safetyFlag +.5)

    return safetyFlag

def makeProbabilisticAlphaSafetyFunctionEvaluation(safetyFunctionF,physicalStateSubEstimatorSampleF,numSamples=100,alpha=.95):
    """
    Function factory to make probabilisticAlphaSafetyFunctionEvaluation functions.
    We shouldn't run into late binding problems here, but I think this is easier to read

    Parameters
    ----------
    safetyFunctionF : function
        Function that represents the conditions the physical state must satisfy for safety.
        Must accept a physical state as an input.
    physicalStateSubEstimatorSampleF : function
        Samples the physical state from the belief
    numSamples : int (default=100)
        Number of samples to be drawn
    alpha : float (default)
        Threshold for safety. ie samples must have % safety > alpha

    Returns
    -------
    probabilisticSafetyFunctionEvaluationF : function
        Wrapper around probabilisticSafetyFunctionEvaluation with the specified safetyFunctionF,physicalStateSubEstimatorSampleF,numSamples
    """
    def probabilisticAlphaSafetyFunctionEvaluationF(beliefTuple,rngKey):
        return probabilisticAlphaSafetyFunctionEvaluation(beliefTuple,rngKey,safetyFunctionF,physicalStateSubEstimatorSampleF,numSamples,alpha)
    return probabilisticAlphaSafetyFunctionEvaluationF

#Jitting: numSamples needs to be a static argument (prettySure, actually not)
@partial(jax.jit, static_argnames=['safetyFunctionF','physicalStateSubEstimatorSampleF','numSamples'])
def probabilisticSafetyFunctionEvaluation(beliefTuple,rngKey,safetyFunctionF,physicalStateSubEstimatorSampleF,numSamples=100):
    """
    Function that evaluates a safety constraint based using samples from the belief.

    Parameters
    ----------
    beliefTuple : tuple
        The belief is represented by an array of failureWeights and the filter array corresponding to each possibleFailure. Contains:
            failureWeights : array, shape(maxPossibleFailures)
                Normalized weighting of the relative likelihood of each failure
            filters : array
                Array of conditional filters on the physical state
    rngKey : JAX PRNG key
        Key for the PRNG, to be used to create pseudorandomness (for some safety constraints)
    safetyFunctionF : function
        Function that represents the conditions the physical state must satisfy for safety.
        Must accept a physical state as an input.
    physicalStateSubEstimatorSampleF : function
        Samples the physical state from the belief
    numSamples : int (default=100)
        Number of samples that must successfully be drawn without violation

    Returns
    -------
    safeFlag : boolean
        1 if safety constraints satisfied, 0 if not
    """

    batchSafetyReturn = batchSampleSafetyHelperMethod(beliefTuple,rngKey,safetyFunctionF,physicalStateSubEstimatorSampleF,numSamples)

    #Implicitly assuming that we are using the boolean safety function here
    #If all pass, each safeFlag will still be 1. Any failure sets this to zero
    return jnp.prod(batchSafetyReturn)

def batchSampleSafetyHelperMethod(beliefTuple,rngKey,safetyFunctionF,physicalStateSubEstimatorSampleF,numSamples):
    """
    Helper method that does the batch sampling for the sample based safety methods
    """

    #Check specified number of samples #vmapping this for speed ups (especially on compile side) >10x faster!!
    #First create the keys for each sample
    batchRngKeys = jaxRandom.split(rngKey,num=numSamples)
    #vmap the helper function. Only rng keys change
    batchSampleSafety = jax.vmap(probabilisticSafetyFunctionEvaluationVMapHelper, in_axes=[None,0,None,None])
    return batchSampleSafety(beliefTuple,batchRngKeys,safetyFunctionF,physicalStateSubEstimatorSampleF)



def probabilisticSafetyFunctionEvaluationVMapHelper(beliefTuple,rngKey,safetyFunctionF,physicalStateSubEstimatorSampleF):
    """
    Helper function for parallelizing the loop in probabilisticSafetyFunctionEvaluation for faster compilation
    """
    #Sample a state from the belief
    failureWeightsIdx = 0
    filtersIdx = 1

    #Pick failure from possible, weighted by current belief
    rngKey,rngSubKey = jaxRandom.split(rngKey) #Make rngSubKey, as consumed on use
    failureIdx = jaxRandom.choice(rngSubKey,len(beliefTuple[failureWeightsIdx]),p=beliefTuple[failureWeightsIdx])

    #Now use the filter to sample x. Use rngKey here, as don't need to split it again
    physicalStateSample = physicalStateSubEstimatorSampleF(beliefTuple[filtersIdx][failureIdx],rngKey)

    return safetyFunctionF(physicalStateSample)

def makeProbabilisticSafetyFunctionEvaluation(safetyFunctionF,physicalStateSubEstimatorSampleF,numSamples=100):
    """
    Function factory to make probabilisticSafetyFunctionEvaluation functions.
    We shouldn't run into late binding problems here, but I think this is easier to read

    Parameters
    ----------
    safetyFunctionF : function
        Function that represents the conditions the physical state must satisfy for safety.
        Must accept a physical state as an input.
    physicalStateSubEstimatorSampleF : function
        Samples the physical state from the belief
    numSamples : int (default=100)
        Number of samples that must successfully be drawn without violation

    Returns
    -------
    probabilisticSafetyFunctionEvaluationF : function
        Wrapper around probabilisticSafetyFunctionEvaluation with the specified safetyFunctionF,physicalStateSubEstimatorSampleF,numSamples
    """
    def probabilisticSafetyFunctionEvaluationF(beliefTuple,rngKey):
        return probabilisticSafetyFunctionEvaluation(beliefTuple,rngKey,safetyFunctionF,physicalStateSubEstimatorSampleF,numSamples)
    return probabilisticSafetyFunctionEvaluationF

@partial(jax.jit, static_argnames=['safetyFunctionF','physicalStateSubEstimatorSampleF','numSamples'])
def chebyshevIneqSafetyFunctionEvaluation(beliefTuple,rngKey,safetyFunctionF,physicalStateSubEstimatorSampleF,numSamples=100,allowableFailureChance=.05):
    """
    Function that evaluates a safety constraint based using samples from the belief. Uses sample based Chebyshev inequality bound from Kaban 2012
    to place an upper bound on the chance of safety inequalities being violated

    Parameters
    ----------
    beliefTuple : tuple
        The belief is represented by an array of failureWeights and the filter array corresponding to each possibleFailure. Contains:
            failureWeights : array, shape(maxPossibleFailures)
                Normalized weighting of the relative likelihood of each failure
            filters : array
                Array of conditional filters on the physical state
    rngKey : JAX PRNG key
        Key for the PRNG, to be used to create pseudorandomness (for some safety constraints)
    safetyFunctionF : function
        Function that represents the conditions the physical state must satisfy for safety.
        Must accept a physical state as an input.
        For this method, using the worstCaseSafetyFunction will work best (enforced in function factory)
        In the future other methods may be added, but they must all return strictly negative values when the function is safe, and positive otherwise
    physicalStateSubEstimatorSampleF : function
        Samples the physical state from the belief
    numSamples : int (default=100)
        Number of samples that must successfully be drawn without violation
    allowableFailureChance : float (default=.05)
        Maximum allowable failure chance (safety chance = 1-allowableFailureChance = alpha)

    Returns
    -------
    safeFlag : boolean
        1 if safety condition satisfied, 0 if not
    """

    batchSafetyReturn = batchSampleSafetyHelperMethod(beliefTuple,rngKey,safetyFunctionF,physicalStateSubEstimatorSampleF,numSamples)

    #Now we need to compute the statistics to get the bound

    safetySampleMean = jnp.mean(batchSafetyReturn) #Note if this is not negative, we're already unsafe.
    safetySampleVariance = jnp.var(batchSafetyReturn,ddof=1) #ddof =1 gives us 1/(N-1), which we needed for an unbiased estimate

    #We want to bound the probability of the safety function giving a positive (unsafe) return.
    # To use Chebyshev, we need to represent this in terms of standard deviations from the mean
    #Use definition of sample std from Kaban 2012
    safetySampleStd = jnp.sqrt((numSamples+1)/numSamples * safetySampleVariance) #We re-square later, but need this to check sampleStdsUntilUnsafe>=1 condition
    #How many standard deviations until we're at 0? Note we're assuming safetySampleMean < 0. This we'll check for explicitly later when returning
    sampleStdsUntilUnsafe = -safetySampleMean/safetySampleStd

    #ASSUMPTIONS: numSamples>=2 (enforced in function factory), sampleStdsUntilUnsafe>=1
    unsafeProbabilityBound = jnp.floor((numSamples+1)/numSamples * ((numSamples-1)/sampleStdsUntilUnsafe**2 +1)) / (numSamples+1)

    #Now we need to determine if unsafeProbabilityBound > allowableFailureChance or if sampleStdsUntilUnsafe < 1, as in either case our bound has failed
    # unsafeProbabilityBound > allowableFailureChance
    boundPassed = jnp.sign(allowableFailureChance - unsafeProbabilityBound) #This needs to be 1, not 0 or -1
    # sampleStdsUntilUnsafe < 1
    assumptionsPassed = jnp.sign(sampleStdsUntilUnsafe - 1) #This is okay to be zero
    #Check all three conditions passed. boundPassed != -1, boundPassed != 0, assumptionsPassed != -1
    safeFlag = (.5 + .5*boundPassed) * jnp.abs(boundPassed) * (.5 + .5*assumptionsPassed)

    return safeFlag

def makeChebyshevIneqSafetyFunctionEvaluation(safetyFunctionF,physicalStateSubEstimatorSampleF,numSamples=100,allowableFailureChance=.05):
    """
    Function factory to make chebyshevIneqSafetyFunctionEvaluation functions.
    We shouldn't run into late binding problems here, but I think this is easier to read

    Parameters
    ----------
    safetyFunctionF : function
        Function that represents the conditions the physical state must satisfy for safety.
        Must accept a physical state as an input.
    physicalStateSubEstimatorSampleF : function
        Samples the physical state from the belief
    numSamples : int (default=100)
        Number of samples that must successfully be drawn without violation
    allowableFailureChance : float (default=.05)
        Maximum allowable failure chance (safety chance = 1-allowableFailureChance)

    Returns
    -------
    chebyshevIneqSafetyFunctionEvaluationF : function
        Wrapper around probabilisticSafetyFunctionEvaluation with the specified safetyFunctionF,physicalStateSubEstimatorSampleF,numSamples
    """

    #Check to ensure valid construction so our assumptions hold/make sense
    if allowableFailureChance <= 0:
        invalidSpecification = "The allowableFailureChance must be greater than 0 for the Chebyshev bound to have meaning"
        raise ValueError(invalidSpecification)
    if numSamples < 2:
        invalidSpecification = "The numSamples must be greater than or equal to 2 for the Chebyshev bound to be valid"
        raise ValueError(invalidSpecification)
    #Check the safety function is compatible (booleanInequalitySafetyFunction for example doesn't work well here )
    #This feels a little ugly, thinking if there is a better way to do it.
    if not safetyFunctionF.__name__ in ("worstCaseSafetyFunctionF",):
        invalidSpecification = "Using the safety function {safetyFunctionF.__name__} is invalid for the Chebyshev bound evaluation method."
        raise ValueError(invalidSpecification)

    def chebyshevIneqSafetyFunctionEvaluationF(beliefTuple,rngKey):
        return chebyshevIneqSafetyFunctionEvaluation(beliefTuple,rngKey,safetyFunctionF,physicalStateSubEstimatorSampleF,numSamples,allowableFailureChance)
    return chebyshevIneqSafetyFunctionEvaluationF

#IDEA: The spacecraft can clip through the obstacle with this formulation, b/ we only check the position at discrete time steps. Would it make sense
#to add a variation that is aware of the time step and velocity and checks for collision this way? Additional complexity=slower, but would be more accurate.
@partial(jax.jit, static_argnames=['inequalityConstraintFTuple'])
def booleanInequalitySafetyFunction(physicalState,inequalityConstraintFTuple):
    """
    Function that defines a safety constraint based on specified inequality constraints.
    These don't need to be convex as we aren't doing optimization, just checking.
    May need to have convex constraints if trying to check safety of a gaussian exactly or something like this.
    Equality constraints omitted because these aren't usually found for safety. Also, achieving these exactly may be infeasible

    Parameters
    ----------
    physicalState : array, shape(numState)
        Physical state of the system to evaluate constraints against
    inequalityConstraintFTuple : tuple
        Tuple of inequality constraints to evaluate. These should all be < 0 when safety is satisfied.
        Strict inequality assumed because: 1) equality can be considered collision. 2) ease of implementation

    Returns
    -------
    safeFlag : boolean
        1 if safety constraints satisfied, 0 if not
    """

    #Will set to 0 if we ever aren't safe
    safeFlag = 1

    #We can't vmap this, because vmap only works over arrays, not tuples (and not of functions)
    for inequalityConstraintF in inequalityConstraintFTuple:
        #If constraint < 0, we are safe.
        inequalityConstraintReturnValSign = jnp.sign(inequalityConstraintF(physicalState))
        #Check for > 0 (positive) AND check for = 0, as both are bad
        inequalityConstraintSatisfied = (.5 - .5 *inequalityConstraintReturnValSign) * jnp.abs(inequalityConstraintReturnValSign)
        safeFlag = safeFlag*inequalityConstraintSatisfied

    #If all pass, each safeFlag will still be 1. Any failure sets this to zero
    return safeFlag #jnp.prod(batchSafetyReturn)

#Can't vmap over tuples or functions
#def inequalitySafetyFunctionVMapHelper(physicalState,inequalityConstraintF):
#    """
#    Helper function for parallelizing the loop in probabilisticSafetyFunctionEvaluation for faster compilation
#    """
#    #If constraint < 0, we are safe.
#    inequalityConstraintReturnValSign = jnp.sign(inequalityConstraintF(physicalState))
#    #Check for > 0 (positive) AND check for = 0, as both are bad
#    inequalityConstraintSatisfied = (.5 - .5 *inequalityConstraintReturnValSign) * jnp.abs(inequalityConstraintReturnValSign)
#    return inequalityConstraintSatisfied



def makeBooleanInequalitySafetyFunctionF(inequalityConstraintFTuple):
    """
    Function factory to make inequalitySafetyFunction functions.
    We shouldn't run into late binding problems here, but I think this is easier to read

    Parameters
    ----------
    inequalityConstraintFTuple : tuple
        Tuple of inequality constraints to evaluate. These should all be < 0 when safety is satisfied.
        Strict inequality assumed because: 1) equality can be considered collision. 2) ease of implementation

    Returns
    -------
    booleanInequalitySafetyFunctionF : function
        Wrapper around booleanInequalitySafetyFunction with the specified constraint tuple
    """
    def inequalitySafetyFunctionF(physicalState):
        return booleanInequalitySafetyFunction(physicalState,inequalityConstraintFTuple)
    return inequalitySafetyFunctionF

@partial(jax.jit, static_argnames=['inequalityConstraintFTuple'])
def worstCaseSafetyFunction(physicalState,inequalityConstraintFTuple):
    """
    Function that returns the worst case (largest) safety constraint value.

    Parameters
    ----------
    physicalState : array, shape(numState)
        Physical state of the system to evaluate constraints against
    inequalityConstraintFTuple : tuple
        Tuple of inequality constraints to evaluate. These should all be < 0 when safety is satisfied.
        Strict inequality assumed because: 1) equality can be considered collision. 2) ease of implementation

    Returns
    -------
    worstCaseSafetyValue : float
        Largest safety constraint value. We assume < 0 is safe.
    """

    #Set to best case (negative infinity)
    worstCaseSafetyValue = - jnp.inf

    #We can't vmap this, because vmap only works over arrays, not tuples (and not of functions)
    for inequalityConstraintF in inequalityConstraintFTuple:
        #If constraint < 0, we are safe. We can accept multiple returns in an array
        inequalityConstraintReturn = inequalityConstraintF(physicalState)

        #Keep largest value
        inequalityConstraintValues = jnp.append(inequalityConstraintReturn,worstCaseSafetyValue)
        worstCaseSafetyValue = jnp.max(inequalityConstraintValues)

    return worstCaseSafetyValue

def makeWorstInequalitySafetyFunctionF(inequalityConstraintFTuple):
    """
    Function factory to make worstCaseSafetyFunction functions.
    We shouldn't run into late binding problems here, but I think this is easier to read

    Parameters
    ----------
    inequalityConstraintFTuple : tuple
        Tuple of inequality constraints to evaluate. These should all be < 0 when safety is satisfied.
        Strict inequality assumed because: 1) equality can be considered collision. 2) ease of implementation

    Returns
    -------
    worstCaseSafetyFunctionF : function
        Wrapper around worstCaseSafetyFunction with the specified constraintTuple
    """
    def worstCaseSafetyFunctionF(physicalState):
        return worstCaseSafetyFunction(physicalState,inequalityConstraintFTuple)
    return worstCaseSafetyFunctionF

@jax.jit
def circularObstacleConstraint(physicalState,radiusObstaclePlusRadiusSpacecraft,center):
    """
    Function that enforces a circular obstacle constraint.

    Parameters
    ----------
    physicalState : array, shape(numState)
        Physical state of the system to evaluate constraints against.
        NOTE: Assumed to be a double integrator state, so for example, we have x, vx, y, vy, ...
        Only the position for each dimension (x,y,...) will be used to determine collision.
        The first len(center) dimensions will be checked for collision, in the same order
    radiusObstaclePlusRadiusSpacecraft : float
        Radius of obstacle AND the spacecraft. This is because this is the closest the centers can come to each other.
    center : array, shape(numDimensionsObstacle)
        The center of the obstacle. Length of this will be used to determine how many dimensions to check

    Returns
    -------
    constraintReturn : int
        If 0 or greater, constraint is violated.
    """
    #Get the positions out of the physical state
    positionSpaceCraft = getImplicitPositionSpaceCraftToCompare(physicalState,center)

    #If within radius, positive, so violated
    return radiusObstaclePlusRadiusSpacecraft - jnp.linalg.norm(center-positionSpaceCraft)

def makeCircularObstacleConstraintF(radiusObstaclePlusRadiusSpacecraft,center):
    """
    Function factory to make circularObstacleConstraint functions.
    This is needed because of python's late binding. If multiple constraints are defined in a loop, only the last set of variables will be used.

    Parameters
    ----------
    radiusObstaclePlusRadiusSpacecraft : float
        Radius of obstacle AND the spacecraft. This is because this is the closest the centers can come to each other.
    center : array, shape(numDimensionsObstacle)
        The center of the obstacle. Length of this will be used to determine how many dimensions to check

    Returns
    -------
    circularObstacleConstraintF : function
        Wrapper around circularObstacleConstraint with the specified radiusObstaclePlusRadiusSpacecraft,center
    """
    def circularObstacleConstraintF(physicalState):
        return circularObstacleConstraint(physicalState,radiusObstaclePlusRadiusSpacecraft,center)
    return circularObstacleConstraintF

@jax.jit
def circularSafeZoneConstraint(physicalState,radiusSafeZoneMinusRadiusSpacecraft,center):
    """
    Function that enforces a circular safe zone constraint.

    Parameters
    ----------
    physicalState : array, shape(numState)
        Physical state of the system to evaluate constraints against.
        NOTE: Assumed to be a double integrator state, so for example, we have x, vx, y, vy, ...
        Only the position for each dimension (x,y,...) will be used to determine collision.
        The first len(center) dimensions will be checked for collision, in the same order
    radiusSafeZoneMinusRadiusSpacecraft : float
        Radius of safe zone MINUS the spacecraft. This is because this is the closest the spacecraft center can come the edge of the safe zone.
    center : array, shape(numDimensionsObstacle)
        The center of the safe zone. Length of this will be used to determine how many dimensions to check

    Returns
    -------
    constraintReturn : int
        If 0 or greater, constraint is violated.
    """
    #Get the positions out of the physical state
    positionSpaceCraft = getImplicitPositionSpaceCraftToCompare(physicalState,center)

    #If outside radius, positive, so violated
    return jnp.linalg.norm(center-positionSpaceCraft) - radiusSafeZoneMinusRadiusSpacecraft

def makeCircularSafeZoneConstraintF(radiusSafeZoneMinusRadiusSpacecraft,center):
    """
    Function factory to make circularSafeZoneConstraint functions.
    This is needed because of python's late binding. If multiple constraints are defined in a loop, only the last set of variables will be used.

    Parameters
    ----------
    radiusSafeZoneMinusRadiusSpacecraft : float
        Radius of obstacle minus the spacecraft. This is because this is the farthest the centers can be from each other.
    center : array, shape(numDimensionsObstacle)
        The center of the obstacle. Length of this will be used to determine how many dimensions to check

    Returns
    -------
    circularSafeZoneConstraintF : function
        Wrapper around circularSafeZoneConstraint with the specified radiusSafeZoneMinusRadiusSpacecraft,center
    """
    def circularSafeZoneConstraintF(physicalState):
        return circularSafeZoneConstraint(physicalState,radiusSafeZoneMinusRadiusSpacecraft,center)
    return circularSafeZoneConstraintF

#Too much avoiding repeated code? #Don't need to jit as never called outside jitted methods
def getImplicitPositionSpaceCraftToCompare(physicalState,constraintPositionState):
    """
    Gets the dimensions of the spacecraft's position that are relevant to a constraint we are evaluating

    Parameters
    ----------
    physicalState : array, shape(numState)
        Physical state of the system to evaluate constraints against.
        NOTE: Assumed to be a double integrator state, so for example, we have x, vx, y, vy, ...
        Only the position for each dimension (x,y,...) will be used to determine collision.
        The first len(center) dimensions will be checked for collision, in the same order
    constraintPositionState : array, shape(numState)
        Physical state of the constraint to compare with the physical state.

    Returns
    -------
    positionSpaceCraft : array, shape(numDimensionsObstacle)
        The relevant position of the SpaceCraft to evaluate the constraint against
    """

    numDimensionsObstacle = len(constraintPositionState)
    #Get the positions out of the physical state
    return physicalState[0:2*numDimensionsObstacle:2]

@jax.jit
def linearObstacleConstraint(physicalState,normalMatrix,offsetVector):
    """
    Function that enforces a (convex) linear obstacle constraints on the position.
    Defined as Ax - b < 0 for each constraint. To be outside the obstacle, one constraint must be satisfied.
    so return the best case (most negative or min) from each line


    Parameters
    ----------
    physicalState : array, shape(numState)
        Physical state of the system to evaluate constraints against.
        NOTE: Assumed to be a double integrator state, so for example, we have x, vx, y, vy, ...
        Only the position for each dimension (x,y,...) will be used to determine collision.
        The first len(center) dimensions will be checked for collision, in the same order
    normalMatrix : array, shape(numConstraints,numDimensionsObstacle)
        A Matrix defining the normal vector of each constraint
    offsetVector : array, shape(numConstraints)
        b Vector defining the offset from the origin of each constraint

    Returns
    -------
    worstConstraint : float
        If 0 or greater, at least one constraint is violated.
    """
    #Get the positions out of the physical state
    positionSpaceCraft = getImplicitPositionSpaceCraftToCompare(physicalState,normalMatrix[0])

    constraintEvaluation = jnp.matmul(normalMatrix, positionSpaceCraft) - offsetVector

    #If outside of obstacle, at least one of these will be negative. Only when all are positive are we in violation
    return jnp.min(constraintEvaluation)

def makeLinearObstacleConstraintF(normalMatrix,offsetVector):
    """
    Function factory to make circularSafeZoneConstraint functions.
    This is needed because of python's late binding. If multiple constraints are defined in a loop, only the last set of variables will be used.

    Parameters
    ----------
    normalMatrix : array, shape(numConstraints,numDimensionsObstacle)
        A Matrix defining the normal vector of each constraint
    offsetVector : array, shape(numConstraints)
        b Vector defining the offset from the origin of each constraint

    Returns
    -------
    circularSafeZoneConstraintF : function
        Wrapper around circularSafeZoneConstraint with the specified radiusObstaclePlusRadiusSpacecraft,center
    """
    def linearObstacleConstraintF(physicalState):
        return linearObstacleConstraint(physicalState,normalMatrix,offsetVector)
    return linearObstacleConstraintF

@jax.jit
def linearSafeZoneConstraint(physicalState,normalMatrix,offsetVector):
    """
    Function that enforces a (convex) composition of linear safe zone constraints on the position.
    Defined as Ax - b < 0 for each constraint. We require every constraint is satisfied for safety,
    so return the worst case (most positive or max) from each line


    Parameters
    ----------
    physicalState : array, shape(numState)
        Physical state of the system to evaluate constraints against.
        NOTE: Assumed to be a double integrator state, so for example, we have x, vx, y, vy, ...
        Only the position for each dimension (x,y,...) will be used to determine collision.
        The first len(center) dimensions will be checked for collision, in the same order
    normalMatrix : array, shape(numConstraints,numDimensionsObstacle)
        A Matrix defining the normal vector of each constraint
    offsetVector : array, shape(numConstraints)
        b Vector defining the offset from the origin of each constraint

    Returns
    -------
    worstConstraint : float
        If 0 or greater, at least one constraint is violated.
    """
    #Get the positions out of the physical state
    positionSpaceCraft = getImplicitPositionSpaceCraftToCompare(physicalState,normalMatrix[0])

    constraintEvaluation = jnp.matmul(normalMatrix, positionSpaceCraft) - offsetVector

    #If outside on any constraint, we are in violation, so require all negative
    return jnp.max(constraintEvaluation)

def makeLinearSafeZoneConstraintF(normalMatrix,offsetVector):
    """
    Function factory to make circularSafeZoneConstraint functions.
    This is needed because of python's late binding. If multiple constraints are defined in a loop, only the last set of variables will be used.

    Parameters
    ----------
    normalMatrix : array, shape(numConstraints,numDimensionsObstacle)
        A Matrix defining the normal vector of each constraint
    offsetVector : array, shape(numConstraints)
        b Vector defining the offset from the origin of each constraint

    Returns
    -------
    circularSafeZoneConstraintF : function
        Wrapper around circularSafeZoneConstraint with the specified radiusObstaclePlusRadiusSpacecraft,center
    """
    def linearSafeZoneConstraintF(physicalState):
        return linearSafeZoneConstraint(physicalState,normalMatrix,offsetVector)
    return linearSafeZoneConstraintF



#Scratch ideas:
#Obstacle defined by radius:
#g(x) = radius**2 - 2norm(xCenter-xPos)**2. If within radius, 2norm**2 < radius**, so positive, so violated


#Add and Multiplication check on -1/1: .5 - .5*testVal (0 if testVal=1, 1 if testVal=-1)
