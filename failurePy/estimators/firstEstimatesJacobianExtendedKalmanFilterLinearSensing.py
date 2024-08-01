"""
Extended kalman filter on non-linear dynamics and linear sensing.
Uses first estimates Jacobian method to try to reduce estimator inconsistency
https://link.springer.com/chapter/10.1007/978-3-642-00196-3_43

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

from jax.random import multivariate_normal as multivariateNormal
from failurePy.estimators.kalmanFilter import updateFilter
#0/1 realization
from failurePy.models.linearModel import getRealizedNominalMeasurement as linearGetRealizedNominalMeasurementF
#General realization
from failurePy.models.linearModelGeneralFaults import getRealizedNominalMeasurement as linearGeneralFaultsGetRealizedNominalMeasurementF



#Don't need to jit or vmap unless using it separately from marginal filter! currently we are not.
#If we need to, use from functools import partial to make functions static args: @partial(jax.jit, static_argnames=['n']))
#@partial(jax.jit, static_argnames=['systemF']) #yse with vmap from calling function, otherwise compiling takes too long (2 mins!). Only need if called outside marginal filter.
def predictAndUpdateAll(nominalAction,observation,previousFilters,possibleFailures,systemF,systemParametersTuple,physicalStateJacobianF,
                        getRealizedNominalMeasurementF=linearGetRealizedNominalMeasurementF):
    """
    Function that propagates and updates the Kalman filter given a nonlinear system and relevant parameters.
    Only works for MULTIVARIATE (ie, state and observation 2 or higher dimensions) cases

    Parameters
    ----------
    nominalAction : array, shape(numAct)
        Action taken before failures are considered (this is handled in systemF)
    observation : array, shape(numSenors)
        Observation received
    previousFilters : array
        Array of previous filters, each element is an array representing mean and covariance
    possibleFailures : array, shape(maxPossibleFailures,numAct+numSen)
        Array of possible failures to be considered by the solver. If not exhaustive, these are a random subset. Belief failure weights are are over these possibilities
    systemF : function
        Function reference of the system to call to run experiment.
    systemParametersTuple : tuple
        Needed parameters for systemF to simulate forward
        Contents are in order (for s/c rotations):
            positionInfluenceMatrix : array, shape(2, numThrusters)
                Bp matrix. Influence of the thrusters on the position scaled by mass. We choose to break these out so we can deal with rotation of body frame
            rotationInfluenceMatrix : array, shape(numThrusters)
                Br matrix. Influence of the thrusters on the rotation scaled by moment. We choose to break these out so we can deal with rotation of body frame
            reactionWheelInfluenceMatrix : array, shape(numWheels)
                G matrix. Influence of reaction wheels on the rotation (they don't affect position). We choose to break these out so we can deal with rotation of body frame and stored momentum.
                For 3DOF this is just the ones array
            positionCoefficientOfFriction : float
                Coefficient of friction that would be in the Ap matrix. Each axis is assumed to be the same coefficient of friction
            rotationalCoefficientOfFriction : float
                Coefficient of friction that would be in the Ar matrix. Since this can vary independent of the position coefficient (for example, based on pad spacing) as it yields a torque,
                we report it separately. We could instead deal with the lever arms, but this is easier and more robust
            sensingMatrix : array, shape(numSen, numState)
                C matrix
            dt : float
                The time between time steps of the experiment
            covarianceQ : array, shape(numState,numState)
                Covariance of the dynamics noise.
            covarianceR : array, shape(numSen,numSen)
                Covariance of the sensing noise.
    physicalStateJacobianF : function
        Jacobian of the model for use in estimating.

    Returns
    -------
    updatedFilters : array
        Array of updated Kalman filters
    relativeLikelihoods : array, shape(maxPossibleFailures)
        The relative likelihood of the observation (for failure weight updating)
    """

    #Unpack systemParametersTuple here
    sensingMatrix = systemParametersTuple[-4]
    measurementNoiseMatrixR = systemParametersTuple[-1]

    #We will use vmap to vectorize our updates. This provides another 30% speed up over non-vectorized performance (using filter arrays)

    #Don't need to get realized actions, as this will happen in systemF
    #Slightly different prediction using previous mean
    batchPredictFilter = jax.vmap(predictNonLinearFEJ,in_axes=[0, None, 0, None, None,None])
    predictedMeans, predictedCovariancePs = batchPredictFilter(previousFilters, nominalAction, possibleFailures, systemF,systemParametersTuple,physicalStateJacobianF )

    #Create the sensing matrices for all failures
    batchPredictedMeasurements = jax.vmap(getRealizedNominalMeasurementF, in_axes=[0,0, None])
    predictedObservations = batchPredictedMeasurements(predictedMeans,possibleFailures,sensingMatrix)

    #Now get the updated filters and relative likelihoods
    #Note this is re-using the linear update, as we assume linear measurements
    batchPredictAndUpdateFilter = jax.vmap(updateFilter, in_axes=[0, 0, 0, None, None, None])
    updatedFilters,relativeLikelihoods = batchPredictAndUpdateFilter(predictedMeans, predictedCovariancePs,predictedObservations,observation,sensingMatrix,measurementNoiseMatrixR)

    return (updatedFilters,relativeLikelihoods)

def makeGeneralFaultFEJExtendedKalmanFilter():
    """
    Constructor that makes the general fault version using the alternate control transition matrix
    and sensing matrix functions
    """
    def predictAndUpdateAllGeneral(nominalAction,observation,previousFilters,possibleFailures,systemF,systemParametersTuple,physicalStateJacobianF):
        return predictAndUpdateAll(nominalAction,observation,previousFilters,possibleFailures,systemF,systemParametersTuple,physicalStateJacobianF,
                        getRealizedNominalMeasurementF=linearGeneralFaultsGetRealizedNominalMeasurementF)
    return predictAndUpdateAllGeneral

def predictNonLinearFEJ(previousFilter,nominalAction,failureState,systemF,systemParametersTuple,physicalStateJacobianF):
    """
    Function that propagates and updates the Kalman filter

    Parameters
    ----------
    previousFilter : array
        The array rows consists of:
        previousMean : array, shape(numState)
            The previous estimate of the physical state
        previousPredictedMean : array, shape(numState)
            The previous estimate of the physical state, BEFORE the measurement update. This follows the FEJ-EKF introduced in: https://doi.org/10.1109%2FROBOT.2008.4543252
        perviousCovarianceP : array, shape(numState,numState)
            The previous covariance estimate of the physical state
    nominalAction : array, shape(numAct)
        Action taken before failures are considered (this is handled in systemF)
    failureState : array, shape(numAct+numSen)
        Failure afflicting the s/c
    systemF : function
        Function reference of the system to call to run experiment. Not used here, but provided to make compatible with marginal filter
    systemParametersTuple : tuple
        Needed parameters for systemF to simulate forward
        Contents are in order (for s/c rotations):
            positionInfluenceMatrix : array, shape(2, numThrusters)
                Bp matrix. Influence of the thrusters on the position scaled by mass. We choose to break these out so we can deal with rotation of body frame
            rotationInfluenceMatrix : array, shape(numThrusters)
                Br matrix. Influence of the thrusters on the rotation scaled by moment. We choose to break these out so we can deal with rotation of body frame
            reactionWheelInfluenceMatrix : array, shape(numWheels)
                G matrix. Influence of reaction wheels on the rotation (they don't affect position). We choose to break these out so we can deal with rotation of body frame and stored momentum.
                For 3DOF this is just the ones array
            reactionWheelMoment : array, shape(numWheels,numWheels)
                Jw, The moment of inertia for each reaction wheel (default is the same for all wheels)
            sensingMatrix : array, shape(numSen, numState)
                C matrix
            dt : float
                The time between time steps of the experiment
            covarianceQ : array, shape(numState,numState)
                Covariance of the dynamics noise.
            covarianceR : array, shape(numSen,numSen)
                Covariance of the sensing noise.
    physicalStateJacobianF : function
        Jacobian of the model for use in estimating.

    Returns
    -------
    predictedMean : array, shape(numState)
        The predicted estimate of the physical state
    predictedCovarianceP : array, shape(numState,numState)
        The predicted covariance estimate of the physical state
    """

    #Get old values
    #We are using a convention where the filter is a n+1xn array, where the filter[0] = mean, filter[1:-1] = covariance, filter[-1] = previousPredictedMean (only this filter)
    previousMean = previousFilter[0]
    perviousCovarianceP = previousFilter[1:-1]
    previousPredictedMean = previousFilter[-1]  #(only this filter)

    covarianceQ = systemParametersTuple[-2]

    #Noise free propagation, so set rng to default since it doesn't matter
    rngKey = jax.random.PRNGKey(0)

    #Propagate system forward to estimate mean (Don't use the other return values of systemF
    predictedMean, dummy, dummy = systemF(previousMean,failureState,nominalAction,rngKey,systemParametersTuple,noisyPropagationBooleanInt=0)
    #Get jacobian for current state.
    jacobian = physicalStateJacobianF(physicalState=previousPredictedMean,failureState=failureState,nominalAction=nominalAction,systemParametersTuple=systemParametersTuple)

    #Propagate covariance with jacobian
    predictedCovarianceP = jnp.matmul(jnp.matmul(jacobian, perviousCovarianceP), jnp.transpose(jacobian)) + covarianceQ

    return predictedMean, predictedCovarianceP

#Used in POMCP to sample a physical state (and Monte Carlo belief propagation)
def sampleFromFilter(filterToSample,rngKey):
    """
    Returns random sample of the state using the current estimate and covariance estimate. DOES NOT return estimate (in general)

    Parameters
    ----------
    filterToSample : tuple
        mean : array, shape(numState)
            The estimate of the physical state
        covarianceP : array, shape(numState,numState)
            The covariance estimate of the physical state
    rngKey : JAX PRNG key
        Key for the PRNG, to be used to create pseudorandomness

    Returns
    -------
    x : array, shape(len(xHat))
        Sample of state
    """

    #We are using a convention where the filter is a n+1xn array, where the filter[0] = mean, filter[1:-1] = covariance, filter[-1] = previousPredictedMean (only this filter)
    return multivariateNormal(rngKey, filterToSample[0], filterToSample[1:-1]) #Different for this filter b/ of the first estimates
