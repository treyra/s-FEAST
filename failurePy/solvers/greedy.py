"""
one-step greedy algorithm that finds the next best action after trying them all to a depth of 1.
"""

import jax.numpy as jnp
from jax import random as jaxRandom
#from failurePy.solvers.sFEAST import BeliefNode
from failurePy.solvers.sFEAST import simulateHelperFunction


#Could jit, but probably no need to for speed.
def solveForNextAction(beliefTuple,solverParametersTuple,possibleFailures,systemF,systemParametersTuple,rewardF,estimatorF,
                       physicalStateSubEstimatorF,physicalStateJacobianF,physicalStateSubEstimatorSampleF,nSimulationsPerTree,rngKey): # pylint: disable=unused-argument
    """
    Function that takes in the current belief tuple, parameters, possible failures and system to determine the next best action to take.
    Uses the SFEAST algorithm

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
    possibleFailures : array, shape(maxPossibleFailures,numAct+numSen)
        Array of possible failures to be considered by the solver. If not exhaustive, these are a random subset. Belief failure weights are are over these possibilities
    systemF : function
        Function reference of the system to call to run experiment. Not used here, but provided to make compatible with marginal filter
    systemParametersTuple : tuple
        Tuple of system parameters needed. See the model being used for details. (ie, linearModel)
        Abstracted for the system function
    rewardF : function
        Reward function to evaluate the beliefs with
    estimatorF : function
        Estimator function to update the beliefs with. Takes batch of filters
    physicalStateSubEstimatorF : function
        Function to update all of the conditional position filters
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
    rootNode : BeliefNode
        Root node of the tree that is now expanded to amount requested (N trajectories)
    """

    #Unpack
    availableActions,discretization = unpackSolverParameters(solverParametersTuple)

    #We perform a greedy search over possible futures. Create the root node (no observation used to get here)
    #We now allow for probabilistic rewards (usually b/ of safety constraint)
    #Unused here in greedy
    #rngKey, rngSubKey = jaxRandom.split(rngKey)
    #rootNode = BeliefNode(None,rewardF(beliefTuple,rngSubKey),beliefTuple,availableActions)

    #Rewards
    rewards = jnp.zeros(len(availableActions))

        #Try every action and save reward
    for iAction, action in enumerate(availableActions):
        #Since already chose action, will use the simulate sub methods directly
        rngKey,rngSubKey = jaxRandom.split(rngKey) #Make rngSubKey, as consumed on use
        nextObservation = simulateHelperFunction(action,possibleFailures,beliefTuple,systemF,systemParametersTuple,physicalStateSubEstimatorSampleF,discretization,rngSubKey)

        #Now see if this observation is any of the existing children of the decision node, or make new node. Get reward
        #We now allow for probabilistic rewards (usually b/ of safety constraint)
        rngKey, rngSubKey = jaxRandom.split(rngKey)
        #Update belief tuple
        nextBeliefTuple = estimatorF(action,nextObservation,beliefTuple,possibleFailures,systemF,systemParametersTuple,physicalStateSubEstimatorF,physicalStateJacobianF)
        #We now allow for probabilistic rewards (usually b/ of safety constraint)
        rewards= rewards.at[iAction].set(rewardF(nextBeliefTuple,rngKey))

        #And done, don't look any further!

    #Need to return best reward! (No tree to return)
    #Rarely the EKF diverges, leading to nans. For now just going to side step this by selecting randomly
    if jnp.isnan(rewards[0]):
        bestRewardIdx = jaxRandom.randint(rngKey,shape=(1,),minval=0,maxval=len(rewards))[0]
    else:


        bestRewardsIdxes = jnp.argwhere(rewards == jnp.amax(rewards))
        #print(len(bestRewardsIdxes),bestRewardsIdxes)
        #Dealing with possible ties, split randomly (probably not going to have any here..,)
        #print(bestRewardsIdxes)
        #print(rewards)
        if len(bestRewardsIdxes) > 1:
            #Pick random most likely failure and assume this to be the dynamics
            bestRewardIdx = jaxRandom.choice(rngKey,bestRewardsIdxes)[0]
        else:
            bestRewardIdx = bestRewardsIdxes[0,0]

    #print(bestRewardIdx)

    return availableActions[bestRewardIdx], None

def unpackSolverParameters(solverParametersTuple):
    """
    Helper method to unpack parameters for readability
    """

    #Get solver parameters
    availableActions = solverParametersTuple[0]
    discretization = solverParametersTuple[1]

    return availableActions,discretization
