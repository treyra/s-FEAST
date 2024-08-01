"""
Default random policy baseline
"""

from jax import random as jaxRandom

#Made to be compatible with distributed solvers, so many unused arguments here
def distributedSolveForNextAction(beliefTuple,solverParametersTuple,possibleFailures,systemF,systemParametersTuple,rewardF,estimatorF, # pylint: disable=unused-argument
                       physicalStateSubEstimatorF,physicalStateSubEstimatorSampleF,nSimulationsPerTree,rngKey): # pylint: disable=unused-argument
    """
    Compatible with d-SFEAST, but random result
    """

    return solveForNextAction(beliefTuple,solverParametersTuple,possibleFailures,systemF,systemParametersTuple,rewardF,estimatorF,
                       physicalStateSubEstimatorF,None,physicalStateSubEstimatorSampleF,nSimulationsPerTree,rngKey)


#Made to be compatible with SFEAST, so many unused arguments here
def solveForNextAction(beliefTuple,solverParametersTuple,possibleFailures,systemF,systemParametersTuple,rewardF,estimatorF, # pylint: disable=unused-argument
                       physicalStateSubEstimatorF,physicalStateJacobianF,physicalStateSubEstimatorSampleF,nSimulationsPerTree,rngKey): # pylint: disable=unused-argument
    """
    Compatible with SFEAST, but random result

    Parameters
    ----------
    beliefTuple : tuple
        The belief is represented by an array of failureWeights and the filter array corresponding to each possibleFailure. Contains:
            failureWeights : array, shape(maxPossibleFailures)
                Normalized weighting of the relative likelihood of each failure
            filters : array
                Array of conditional filters on the physical state
    solverParametersTuple : tuple
        List of solver parameters needed. Contents are:
            availableActions : array, shape(maxNumActions,numAct)
                Array of actions that can be taken. First action is always null action.
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
    possibleFailures : array, shape(maxPossibleFailures,numAct+numSen)
        Array of possible failures to be considered by the solver. If not exhaustive, these are a random subset. Belief failure weights are are over these possibilities
    systemF : function
        Function reference of the system to call to run experiment. Not used here, but provided to make compatible with marginal filter
    systemParametersTuple : tuple
        Tuple of system parameters needed
        Abstracted for the system function
    rewardF : function
        Reward function to evaluate the beliefs with
    estimatorF : function
        Estimator function to update the beliefs with. Takes batch of filters
    physicalStateJacobianF : function
        Jacobian of the model for use in estimating.
    physicalStateSubEstimatorSampleF : function
        Samples from the belief corresponding to this estimator
    nSimulationsPerTree : int
        Number of max simulations per tree for the solver to search
    rngKey : JAX PRNG key
        Key for the PRNG, to be used to create pseudorandomness

    Returns
    -------
    action : array, shape(numAct)
        Action to take next
    rootNode : None
        No belief node for random policy
    """
    #errRandom  NEED TO ADD VALIDATION ON NUM TRIALS PER POINT FOR BASELINES
    availableActions = solverParametersTuple[0]

    #Store action so we can print while debugging
    action = jaxRandom.choice(rngKey,availableActions)
    #print(action)

    return action,None
