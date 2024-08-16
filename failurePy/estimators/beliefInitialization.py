"""
File of belief initialization functions. Currently tailored to each type of belief that needs to be initialized
"""

import jax.numpy as jnp

def uniformFailureBeliefMarginalizedKalman(initialPhysicalState,possibleFailures,initialUncertainty=.001):
    """
    Creates the initial beliefTuple as narrowly distributed in physical belief about the origin,
    and uniform in failure belief. Only valid for marginalized filter with kalman filter

    Parameters
    ----------
    initialPhysicalState : array, shape(numState)
        Initial state the belief is centered around
    possibleFailures : array, shape(nMaxPossibleFailures,numAct+numSen)
        List of possible failures to be considered by the solver. If not exhaustive, these are a random subset. Belief failure weights are are over these possibilities
    initialUncertainty : float (default=.001)
        Initial state uncertainty magnitude, if any


    Returns
    -------
    beliefTuple : tuple
        The belief is represented by an array of failureWeights and the filter array corresponding to each possibleFailure. Contains:
            failureWeights : array, shape(maxPossibleFailures)
                Normalized weighting of the relative likelihood of each failure
            filters : array
                Array of conditional filters on the physical state
    """

    #Make failure weights array
    failureWeights = jnp.ones(len(possibleFailures))/len(possibleFailures)
    filters = []
    for iFailure in range(len(possibleFailures)): # pylint: disable=unused-variable
        #Initialize to small (non-zero) diagonal covariance. Needs to be PD!
        #We are using a convention where the filter is a n+1xn array, where the filter[0] = mean, filter[1:] = covariance!
        filters.append(jnp.vstack((initialPhysicalState,initialUncertainty* jnp.eye(len(initialPhysicalState)))))

    return (failureWeights,jnp.array(filters))


def uniformFailureBeliefMarginalizedFEJKalman(initialPhysicalState,possibleFailures):
    """
    Creates the initial beliefTuple as narrowly distributed in physical belief about the origin,
    and uniform in failure belief. Only valid for marginalized filter with FEJ kalman filter, as adds extra row to the end

    Parameters
    ----------
    possibleFailures : array, shape(nMaxPossibleFailures,numAct+numSen)
        List of possible failures to be considered by the solver. If not exhaustive, these are a random subset. Belief failure weights are are over these possibilities

    Returns
    -------
    beliefTuple : tuple
        The belief is represented by an array of failureWeights and the filter array corresponding to each possibleFailure. Contains:
            failureWeights : array, shape(maxPossibleFailures)
                Normalized weighting of the relative likelihood of each failure
            filters : array
                Array of conditional filters on the physical state
    """

    #Make failure weights array
    failureWeights = jnp.ones(len(possibleFailures))/len(possibleFailures)
    filters = []
    for iFailure in range(len(possibleFailures)): # pylint: disable=unused-variable
        #Initialize to small (non-zero) diagonal covariance. Needs to be PD!
        #We are using a convention where the filter is a n+1xn array, where the filter[0] = mean, filter[1:-1] = covariance, filter[-1] = previousPredictedMean (only for this filter)
        filters.append(jnp.vstack((initialPhysicalState,.001* jnp.eye(len(initialPhysicalState)),initialPhysicalState)))

    return (failureWeights,jnp.array(filters))
