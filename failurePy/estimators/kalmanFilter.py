"""
Adapts the  FilterPy library.
http://github.com/rlabbe/filterpy

Documentation at:
https://filterpy.readthedocs.org

FilterPy library is licensed under an MIT license. See the readme.MD file
for more information.

Copyright 2014-2018 Roger R Labbe Jr.
"""

import jax
#from functools import partial Unused unless this is ever used outside of marginal filter
import jax.numpy as jnp
from failurePy.estimators.kalmanFilterCommon import updateMultiDimensional, computeRelativeLikelihood
#0/1 realization
from failurePy.models.linearModel import getRealizedAction as linearGetRealizedActionF
from failurePy.models.linearModel import getRealizedNominalMeasurement as linearGetRealizedNominalMeasurementF
#General realization
from failurePy.models.linearModelGeneralFaults import getRealizedAction as linearGeneralFaultsGetRealizedActionF
from failurePy.models.linearModelGeneralFaults import getRealizedNominalMeasurement as linearGeneralFaultsGetRealizedNominalMeasurementF


#Using default control realization and sensing realizations, makes more backwards compatible. BUT! We must have this function after those methods now, or otherwise
#we would need to import ourselves, which is bad form.
#Don't need to jit unless using it separately from marginal filter! currently we are not
#@partial(jax.jit, static_argnames=['systemF']) #Only a major speed up for many trials! Otherwise compiling takes too long
def predictAndUpdateAll(nominalAction,observation,previousFilters,possibleFailures,systemF,systemParametersTuple,physicalStateJacobianF,# pylint: disable=unused-argument
                        getRealizedActionF=linearGetRealizedActionF,getRealizedNominalMeasurementF=linearGetRealizedNominalMeasurementF):
    """
    Function that propagates and updates the Kalman filter given a LINEAR system and relevant parameters.
    Only works for MULTIVARIATE (ie, state and observation 2 or higher dimensions) cases

    Parameters
    ----------
    nominalAction : array, shape(numAct)
        Action taken before failures are considered (this is handled in by the controlTransitionMatrices)
    observation : array, shape(numSenors)
        Observation received
    previousFilters : array
        Array of previous filters, each element is an array representing mean and covariance
    possibleFailures : array, shape(maxPossibleFailures,numAct+numSen)
        Array of possible failures to be considered by the solver. If not exhaustive, these are a random subset. Belief failure weights are are over these possibilities
    systemF : function
        Function reference of the system to call to run experiment. Not used here, but provided to make compatible with marginal filter
    systemParametersTuple : tuple
        Tuple of system parameters needed. See the model being used for details. (ie, linearModel). Model must be linear
        Contents are in order:
            stateTransitionMatrix : array, shape(numState,numState)
                State transition matrix from previous state to the next state. This depends on dt, and should be recreated if dt changes

            controlTransitionMatrix : array, shape(numState,numAct)
                State Transition matrix resulting from constant control input
            sensingMatrix : array, shape(numSen, numState)
                C matrix
            covarianceQ : array, shape(numState,numState)
                Covariance of the dynamics noise.
            covarianceR : array, shape(numSen,numSen)
                Covariance of the sensing noise.
    physicalStateJacobianF : function
        Jacobian of the model for use in estimating. Not used here, but provided to make compatible with marginal filter
    getRealizedActionF : function
        Function used to realize the action given a failure particle
    getRealizedNominalMeasurementF : function
        Function used to realize the nominal measurement given a failure particle

    Returns
    -------
    updatedFilters : array
        Array of updated Kalman filters
    relativeLikelihoods : array, shape(maxPossibleFailures)
        The relative likelihood of the observation (for failure weight updating)
    """
    #Unpack systemParametersTuple here
    stateTransitionMatrix = systemParametersTuple[0]
    controlTransitionMatrix = systemParametersTuple[1]
    sensingMatrix = systemParametersTuple[2]
    #Get noise covariance matrices
    processNoiseMatrixQ = systemParametersTuple[3]
    measurementNoiseMatrixR = systemParametersTuple[4]

    #We will use vmap to vectorize our updates. This provides another 30% speed up over non-vectorized performance (using filter arrays)

    #Get the realized actions for each filter (no longer looking at transition matrices, cause can get this directly)
    batchRealizeActions = jax.vmap(getRealizedActionF, in_axes=[0, None])
    realizedActions = batchRealizeActions(possibleFailures,nominalAction)

    batchPredictFilter = jax.vmap(predict,in_axes=[0, None, None, 0, None])
    predictedMeans, predictedCovariancePs = batchPredictFilter(previousFilters, stateTransitionMatrix, processNoiseMatrixQ, realizedActions, controlTransitionMatrix)

    #Create the sensing matrices for all failures
    batchPredictedMeasurements = jax.vmap(getRealizedNominalMeasurementF, in_axes=[0,0, None])
    predictedObservations = batchPredictedMeasurements(predictedMeans,possibleFailures,sensingMatrix)

    #Now get the updated filters and relative likelihoods
    batchPredictAndUpdateFilter = jax.vmap(updateFilter, in_axes=[0, 0, 0, None, None, None])
    updatedFilters,relativeLikelihoods = batchPredictAndUpdateFilter(predictedMeans, predictedCovariancePs,predictedObservations,observation,sensingMatrix,measurementNoiseMatrixR)

    return (updatedFilters,relativeLikelihoods)

def makeGeneralFaultKalmanFilter():
    """
    Constructor that makes the general fault version using the alternate control transition matrix
    and sensing matrix functions
    """
    def predictAndUpdateAllGeneral(nominalAction,observation,previousFilters,possibleFailures,systemF,systemParametersTuple,physicalStateJacobianF):
        return predictAndUpdateAll(nominalAction,observation,previousFilters,possibleFailures,systemF,systemParametersTuple,physicalStateJacobianF,
                        getRealizedActionF=linearGeneralFaultsGetRealizedActionF,getRealizedNominalMeasurementF=linearGeneralFaultsGetRealizedNominalMeasurementF)
    return predictAndUpdateAllGeneral


def updateFilter(predictedMean, predictedCovarianceP,predictedObservation,observation,sensingMatrix,measurementNoiseMatrixR):
    """
    Function that propagates and updates the Kalman filter given a LINEAR system and relevant parameters

    Parameters
    ----------
    realizedAction : array, shape(numAct)
        Action taken (after fault applied)
    observation : array, shape(numSenors)
        Observation received
    previousFilter : array
        The tuple consists of the pervious mean and covariance of the filter
    stateTransitionMatrix : array, shape(numState,numState)
        State transition matrix from previous state to the next state. This depends on dt, and should be recreated if dt changes
    controlTransitionMatrix : array, shape(numState,numAct)
        Maps constant control input to change in state
    sensingMatrix : array, shape(numSen,numState)
        Measurement function (conditioned on failure)
    covarianceQ : array, shape(numState,numState)
        Covariance of the dynamics noise.

    Returns
    -------
    updatedFilter : array
        Array of the updated Kalman filter
    relativeLikelihood : float
        The relative likelihood of the observation (for failure weight updating)
    """

    #Relative likelihood of observation (we need this for failure weighting)
    relativeLikelihood = computeRelativeLikelihood(observation,predictedObservation, predictedCovarianceP,sensingMatrix,measurementNoiseMatrixR)

    #Innovate with new measurement
    updatedMean,updatedCovarianceP = updateMultiDimensional(predictedMean,predictedCovarianceP,predictedObservation,observation,measurementNoiseMatrixR,sensingMatrix)

    #Stack into a single object and return (so we don't need to re-assign later. This might already be optimized when jitted, doesn't make much of a change)
    return jnp.vstack((updatedMean,updatedCovarianceP)),relativeLikelihood

#Original FilterPy method, re-written to match our conventions alpha discarded as we don't use memory here
def predict(previousFilter, stateTransitionMatrix, processNoiseMatrixQ, realizedAction, controlTransitionMatrix): #alpha=1.):  # pylint: disable=invalid-name
    """
    Predict next state (prior) using the Kalman filter state propagation
    equations.

    Parameters
    ----------
    previousFilter : array, shape(numState+1,numState)
        The rows consists of the pervious mean and covariance of the filter
            previousMean : array, shape(numState)
                Previous mean estimate
            perviousCovarianceP : array, shape(numState,numState)
                Previous covariance estimate
    stateTransitionMatrix : array, shape(numState,numState)
        State transition matrix from previous state to the next state. This depends on dt, and should be recreated if dt changes
    covarianceQ : array, shape(numState,numState)
        Covariance of the dynamics noise.
    realizedAction : array, shape(numAct)
        Action taken (after fault applied)
    controlTransitionMatrix : array, shape(numState,numAct)
        Maps constant control input to change in state

    Returns
    -------
    predictedMean : array, shape(numState)
        Prior state estimate vector
    predictedCovarianceP : array, shape(numState,numState)
        Prior covariance matrix
    """

    #processNoiseMatrixQ *= 1.5

    #Get old values
    #We are using a convention where the filter is a n+1xn array, where the filter[0] = mean, filter[1:] = covariance!
    previousMean = previousFilter[0]
    perviousCovarianceP = previousFilter[1:]

    predictedMean = jnp.matmul(stateTransitionMatrix, previousMean) + jnp.matmul(controlTransitionMatrix, realizedAction)
    predictedCovarianceP = jnp.matmul(jnp.matmul(stateTransitionMatrix, perviousCovarianceP), jnp.transpose(stateTransitionMatrix)) + processNoiseMatrixQ

    return predictedMean, predictedCovarianceP
