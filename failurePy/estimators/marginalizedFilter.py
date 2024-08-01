"""
Implements the marginalized filter, agnostic to the physical state estimator
"""
from functools import partial
import jax
import jax.numpy as jnp


#jit (we can use from functools import partial to make functions static args! @partial(jax.jit, static_argnames=['n']))
@partial(jax.jit, static_argnames=['systemF','physicalStateSubEstimatorF','physicalStateJacobianF']) #Big slow down to compile, esp with EKF (2 min!)
def updateMarginalizedFilterEstimate(action,observation,previousBeliefTuple,possibleFailures,systemF,systemParametersTuple,physicalStateSubEstimatorF,physicalStateJacobianF):
    """
    Function that propagates and updates the marginalized filter, given physical state estimator

    Parameters
    ----------
    action : array, shape(numAct)
        Action taken
    observation : array, shape(numSenors)
        Observation received
    previousBeliefTuple : tuple
        The belief is represented by an array of failureWeights and the filter array corresponding to each possibleFailure. Contains:
            failureWeights : array, shape(maxPossibleFailures)
                Normalized weighting of the relative likelihood of each failure
            filters : array
                Array of conditional filters on the physical state
    possibleFailures : array, shape(maxPossibleFailures,numAct+numSen)
        Array of possible failures to be considered by the solver. If not exhaustive, these are a random subset. Belief failure weights are are over these possibilities
    systemF : function
        Function reference of the system to call to run experiment
    systemParametersTuple : tuple
        Tuple of system parameters needed. See the model being used for details. (ie, linearModel)
        Contents are in order:
            stateTransitionMatrix : array, shape(numState,numState)
                State transition matrix from previous state to the next state. This depends on dt, and should be recreated if dt changes
                Future optimization:
            controlTransitionMatrix : array, shape(numState,numAct)
                State Transition matrix resulting from constant control input
            sensingMatrix : array, shape(numSen, numState)
                C matrix
            covarianceQ : array, shape(numState,numState)
                Covariance of the dynamics noise.
            covarianceR : array, shape(numSen,numSen)
                Covariance of the sensing noise.
    physicalStateSubEstimatorF : function
        Function to update all of the conditional position filters
    physicalStateJacobianF : function
        Jacobian of the model for use in estimating. Can be none if physicalStateSubEstimatorF doesn't need it.

    Returns
    -------
    updatedBeliefTuple : tuple
        The tuple consists of an updated array of weights over possibleFailures and corresponding conditional filters
    """


    #Get previous failure weights, filters
    perviousFailureWeights = previousBeliefTuple[0]
    previousFilters = previousBeliefTuple[1]

    #Get new filters
    updatedFilters,relativeLikelihoods = physicalStateSubEstimatorF(action,observation,previousFilters,possibleFailures,systemF,systemParametersTuple,physicalStateJacobianF)

    #print("relativeLikelihoods")
    #print(relativeLikelihoods)

    #Update failure weights
    updatedFailureWeights = updateFailureWeights(relativeLikelihoods,perviousFailureWeights)
    #print("updatedFailureWeights")
    #print(updatedFailureWeights)
    #error
    return (updatedFailureWeights,updatedFilters)

def updateFailureWeights(relativeLikelihoods,perviousFailureWeights):
    """
    Function that updates the failure weights of the marginalized filter via a Bayesian update

    Parameters
    ----------
    relativeLikelihoods : array, shape(maxPossibleFailures)
        Relative likelihood of each observation given the action taken and the previous (conditional) physical state filters
    previousFailureWeights : array, shape(maxPossibleFailures)
        Normalized weighting of the relative likelihood of each failure before this update. Sorted to match possibleFailures

    Returns
    -------
    updatedFailureWeights : array, shape(maxPossibleFailures)
        Normalized weighting of the relative likelihood of each failure after this update
    """

    #Create new failure weights. Adding stabilizing noise
    updatedFailureWeights = jnp.multiply(relativeLikelihoods,perviousFailureWeights)

    #Normalize
    return updatedFailureWeights/jnp.sum(updatedFailureWeights)
