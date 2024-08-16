"""
Simple implementation of a control barrier function based solver for safety.
"""

import jax.numpy as jnp
from jax import random as jaxRandom
import numpy as onp #Following Jax conventions to be explicit about which implementation of numpy we use
from scipy.optimize import minimize, Bounds



#Usually doesn't do anything, going to just take a brute force approach and see how well that works
def solveForNextAction(beliefTuple,solverParametersTuple,possibleFailures,systemF,systemParametersTuple,rewardF,estimatorF, # pylint: disable=unused-argument
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
    numActuators,safetyFunctionF = unpackSolverParameters(solverParametersTuple)


    #We sample over possible actions that might satisfy cbf

    #Outline of function.
    # Compute ML failure. In event of a tie (such as at initial state), select randomly
    # Sample within actuation limits (thinking setting any actuator to +/- 5 m/s^2 influence, since empirically, velocity usually below 10m/s^2
    # Compute nominal next state given the most likely failure
    # see if satisfies hcbf(next state) > hcbf(current state), hcbf = h * (10-||v||) -> safe + velocity bounded
    # If fails bound, keep if violation is less than previous best. Resample up to 100 times
    # Use smallest violation if failed to find an action that satisfies the bound.
    # Actually, just set hcbf to h, and progressively widen a until we find one that works or until we have widened 100 times.


    #Get ML failure
    failureWeights = beliefTuple[0]

    #When cbf fails to solve, large actions can have the EKF diverge, leading to nans. For now just going to side step this by selecting randomly
    if jnp.isnan(failureWeights[0]):
        mostLikelyFailuresIdx = jaxRandom.randint(rngKey,shape=(1,),minval=0,maxval=len(failureWeights))[0]
    else:
        mostLikelyFailuresIdxes = jnp.argwhere(failureWeights == jnp.amax(failureWeights))
        #Dealing with possible ties/lack there of
        if len(mostLikelyFailuresIdxes) > 1:
            #Pick random most likely failure and assume this to be the dynamics
            mostLikelyFailuresIdx = jaxRandom.choice(rngKey,mostLikelyFailuresIdxes)[0]
        else:
            mostLikelyFailuresIdx = mostLikelyFailuresIdxes[0,0]

    mostLikelyFailure = possibleFailures[mostLikelyFailuresIdx]
    #Need to assume a state. Idea, propagate posterior of this failure and require up to 2 sigma (95%!) to be safe?
    # Downside: why not just use full failure and propagate action? Maybe we should? Could re-use our chebyshev bound? With random observations/noise?

    #Best plan so far, select action, simulate 100 times on state/failure sampled from initial belief, evaluate average hcbf as approximate hcbf, check if

    #Now get ML state (first row of associated filter)
    assumedState = beliefTuple[1][mostLikelyFailuresIdx,0]

    cbfConstraintWrapperF = makeCbfConstraintWrapper(assumedState, mostLikelyFailure, systemF, systemParametersTuple, safetyFunctionF)

    initialActionGuess = onp.zeros(numActuators) #When this is feasible, it's always the solution
    actuationBounds = Bounds(lb=-10,ub=10)

    #print(mostLikelyFailuresIdx)
    #print(mostLikelyFailure,"mostLikelyFailure")
    #print(assumedState,"assumedState")
    #print(initialActionGuess,"initialActionGuess")

    #Solve as non-linear program (note that this is probably not optimized, hoping it can solve well enough numerically without needing to optimize)
    # x is the optimal solution
    solution = minimize(minControlObjective, initialActionGuess, constraints={'type': 'ineq', 'fun': cbfConstraintWrapperF},bounds=actuationBounds)
    action = solution['x']
    #print(solution['success'],solution['message'])
    #Map negative controls to opposite thruster
    for iThruster in range(8):
        if action[iThruster] < 0:
            #Influence matrices are: -1,-1,1,1 per axis, 1,-1,1,-1,... for rotation
            #Map to new thruster
            thrusterOppositeMap = (3,2,1,0,7,6,5,4)
            idxNewThruster = thrusterOppositeMap[iThruster]
            action[idxNewThruster] -= action[iThruster] #Note sign, since action is negative at this idx
            action[iThruster] = 0
    #for iActuator in range(10):
    # #Zero out controls close to 0 (just for vis)
    #    if action[iActuator] < 1e-3:
    #        action[iActuator] = 0
    #print(-safetyFunctionF(assumedState),action,cbfConstraintWrapperF(solution['x']))
    return action, None #No tree to return


def makeCbfConstraintWrapper(assumedState, mostLikelyFailure, systemF, systemParametersTuple, safetyFunctionF):
    """
    Constructor that wraps up the constraints for the cbf so that we can pass to optimizer.
    """
    def cbfConstraintWrapper(action):
        return cbfConstraint(action,assumedState, mostLikelyFailure, systemF, systemParametersTuple, safetyFunctionF)

    return cbfConstraintWrapper

def minControlObjective(action):
    """
    Minimizes ||u||^2
    """
    return onp.linalg.norm(action,ord=2)

def cbfConstraint(action, physicalState, failureState, systemF, systemParametersTuple, safetyFunctionF):
    """
    Determines the cbf value at this action, given the current state, system dynamics, and safety function
    """

    #rngKey not used, because noisyPropagationBooleanInt = 0, so just hand it basic one
    rngKey = jaxRandom.PRNGKey(0)
    nominalNextPhysicalState,dummyNextFailureState,dummyNextObservation = systemF(physicalState,failureState,action,rngKey,systemParametersTuple,noisyPropagationBooleanInt=0)

    #Evaluate the value of the safety function (this is h), Note minus sign convention, as cbf literature has h=>0 safe, but optimizers want constraints <=0, but our safety already has <0 safe

    sigmaW = .4 #Hard coded for now

    #CBF with alpha = 0 (gamma = 1), as this is the least restrictive
    #Adding distance, so that it needs to be in 90% confidence (relative to linear noise sigmaW)
    return -safetyFunctionF(nominalNextPhysicalState) - 1.28155*sigmaW#If h>0, this should be -safetyFunctionF

def unpackSolverParameters(solverParametersTuple):
    """
    Helper method to unpack parameters for readability
    """

    #Get solver parameters
    numActuators = solverParametersTuple[0]
    safetyFunctionF = solverParametersTuple[1]

    return numActuators,safetyFunctionF
