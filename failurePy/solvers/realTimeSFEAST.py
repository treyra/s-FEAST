"""
POMCP Algorithm adapted to carry the updated belief state forward instead of the approximation,
as this is what our reward is conditioned on. Uses a marginalized filter to do so.

Goal of this implementation is to be purely functional, to allow for jitting of the algorithm.
Currently the tree has to be represented as objects, but there are ideas to do this with arrays instead.

This version is designed for real-time use, meaning that the action active on the system while
the tree search is running is incorporated in the tree search
"""

import time
from tqdm import tqdm
import jax
import jax.numpy as jnp
from jax import random as jaxRandom
from failurePy.solvers.sFEAST import BeliefNode, simulateHelperFunction, simulate, unpackSolverParameters,getNextBeliefNodeAndReward #Re-used methods


def solveForNextAction(beliefTuple, solverParametersTuple, possibleFailures, systemF, systemParametersTuple, rewardF, estimatorF,
                       physicalStateSubEstimatorF,physicalStateJacobianF,physicalStateSubEstimatorSampleF,nSimulationsPerTree,rngKey,currentAction=None):
    """
    Function that takes in the current belief tuple, parameters, possible failures and system to determine the next best action to take.
    This action is dependant on the observation seen after this tree search terminates.
    Therefore, only the rootNode (beliefNode) is returned from this function, and the best action is extracted once the solver knows the real observation
    Uses the SFEAST algorithm in real-time

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
    currentAction : array, shape(numAct) (default=None)
        Action that is active while the tree search is running. The first action that is automatically taken in the tree search.
        If unspecified, assumed to be the null action

    Returns
    -------
    action : array, shape(numAct)
        Action to take next
    rootNode : BeliefNode
        Root node of the tree that is now expanded to amount requested (N trajectories)

    """

    #Unpack
    startTime,availableActions,discretization,maxSimulationTime,explorationParameter,nMaxDepth,discountFactor = unpackSolverParameters(solverParametersTuple)

    if currentAction is None:
        currentAction = availableActions[0]
    #We now allow for probabilistic rewards (usually b/ of safety constraint)
    rngKey, rngSubKey = jaxRandom.split(rngKey)
    #Need to check if a default action that was NOT part of the normal action set was used (ie, Tom's idea)
    if jnp.any(jnp.all(currentAction != availableActions,axis=1)):
        augmentedAvailableActions = jnp.zeros((len(availableActions)+1,len(availableActions[0])))
        augmentedAvailableActions = augmentedAvailableActions.at[:len(availableActions),:].set(availableActions)
        augmentedAvailableActions = augmentedAvailableActions.at[len(availableActions),:].set(currentAction)
        # root node, no observation to get here, just the belief
        rootNode = BeliefNode(None, rewardF(beliefTuple,rngSubKey), beliefTuple, augmentedAvailableActions)
         #will return array, takes first (should be only) value of that array. Need first 0 to get the device array, second to pop the integer out
        actionIdx = jnp.where((augmentedAvailableActions == currentAction).all(axis=1))[0][0]
    else:
        # root node, no observation to get here, just the belief
        rootNode = BeliefNode(None, rewardF(beliefTuple,rngSubKey), beliefTuple, availableActions)

        #What's different here from standard POMCP is that the first action is prescribed (BUT, we DON'T know what the observation is going to be still)
        #Search for actionIdx that matches the prescribed action
        #Will return array, takes first (should be only) value of that array. Need first 0 to get the device array, second to pop the integer out
        actionIdx = jnp.where((availableActions == currentAction).all(axis=1))[0][0]
    decisionNode = rootNode.children[actionIdx]


    # search until time-out or use all simulations
    for iSimulation in tqdm(range(nSimulationsPerTree)): # pylint: disable=unused-variable
    #for iSimulation in range(nSimulationsPerTree): # pylint: disable=unused-variable

        #Time out
        if (time.time() - startTime ) > maxSimulationTime:
            print("Timeout")
            break
        #make rngSubKey, as consumed on use
        rngKey, rngSubKey = jaxRandom.split(rngKey)

        #Need to sim first observation before working down the tree. Don't really care about the reward here
        nextObservation = simulateHelperFunction(currentAction, possibleFailures, beliefTuple, systemF, systemParametersTuple,physicalStateSubEstimatorSampleF,discretization,rngSubKey)

        #Now see if this observation is any of the existing children of the decision node, or make new node.
        #We get reward, but we don't need it here at the outer loop
        rngKey, rngSubKey = jaxRandom.split(rngKey)
        dummyReward, nextBeliefNode = getNextBeliefNodeAndReward(availableActions, possibleFailures, systemF, systemParametersTuple, rewardF, estimatorF, physicalStateSubEstimatorF,
                                                                 physicalStateJacobianF, beliefTuple, currentAction, decisionNode, nextObservation,rngSubKey)

        #make rngSubKey, as consumed on use (as we loop and will use rngKey a again)
        rngKey, rngSubKey = jaxRandom.split(rngKey)

        #Simulate forward (don't use the reward)
        dummyDiscountedReward = simulate(nextBeliefNode,nMaxDepth,discretization,explorationParameter,
                                availableActions,possibleFailures,systemF,systemParametersTuple,discountFactor,
                                rewardF,estimatorF,physicalStateSubEstimatorF,physicalStateJacobianF,physicalStateSubEstimatorSampleF,
                                depth=0,rngKey=rngSubKey) # Choose depth to be zero, since the previous cycle was just applying previously selected action
        #Simulate returns the discounted reward, currently we don't use if for anything

    #Now need to get the next action to take. We will get the best actions for each possible next observation considered, then weight by the likelihood of each observation
    action = availableActions[getWeightedBestActionIdx(decisionNode.children,rngKey)]

    return action, rootNode


#This uses best action instead of most visited, departure from POMCP, but I think this is better (avoids degeneracies when not every node is visited, likely here because of observation sampling)
#When the number of visits is high, they converge to the same value, so this only matters in the case of limited simulations (like when running in real-time)
def getWeightedBestActionIdx(beliefNodesList,rngKey):
    """
    Helper function to sample from the best actions of each belief node, weighted by visits to the belief nodes

    Parameters
    ----------
    beliefNodesList : list
        List of belief nodes to chose best actions from
    rngKey : JAX PRNG key
        Key for the PRNG, to be used to create pseudorandomness

    Returns
    -------
    bestActionIdx : int
        Index of the action to take next, sampled from the belief nodes' best actions weighted by visits
    """

    bestActionIdxes = jnp.zeros(len(beliefNodesList))
    nVisitsArray = jnp.zeros(len(beliefNodesList))

    for iBeliefNode, beliefNode in enumerate(beliefNodesList):
        #rngKey is needed incase we tie on zero visits
        rngKey, rngSubKey = jaxRandom.split(rngKey)
        bestActionIdxes = bestActionIdxes.at[iBeliefNode].set(beliefNode.getBestActionIdx(rngKey=rngSubKey))
        nVisitsArray = nVisitsArray.at[iBeliefNode].set(beliefNode.nVisits)

    return int(getWeightedBestActionIdxHelperFunction(rngKey,bestActionIdxes,nVisitsArray))

@jax.jit
def getWeightedBestActionIdxHelperFunction(rngKey,bestActionIdxes,nVisitsArray):
    """
    jittable helper function
    """
    return jaxRandom.choice(rngKey,bestActionIdxes,p=nVisitsArray/jnp.sum(nVisitsArray))
