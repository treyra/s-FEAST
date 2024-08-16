"""
Simple reward function that is the square sum over the failure beliefs
"""

import jax.numpy as jnp
import jax

@jax.jit
def squareSumFailureBeliefReward(beliefTuple,rngKey=None): #pylint: disable=unused-argument
    """
    Reward on the certainty of the failure belief, or the L2 norm of the failure belief

    Parameters
    ----------
    beliefTuple : tuple
        The belief is represented by an array of failureWeights and the filter array corresponding to each possibleFailure. Contains:
            failureWeights : array, shape(maxPossibleFailures)
                Normalized weighting of the relative likelihood of each failure
            filters : array
                Array of conditional filters on the physical state
    rngKey : JAX PRNG key (default=None)
        Unused here, included for compatibility with the safety filters

    Returns
    -------
    reward : float
        Reward for how certain we are. Note this will always be 0-1 because
        the belief distribution is a probability distribution
    """

    failureWeights = beliefTuple[0]
    return jnp.sum(jnp.square(failureWeights))
