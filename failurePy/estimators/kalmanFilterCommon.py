"""
Common functions used in the kalman filter classes, to avoid repeated code

Adapts the  FilterPy library.
http://github.com/rlabbe/filterpy

Documentation at:
https://filterpy.readthedocs.org

FilterPy library is licensed under an MIT license. See the readme.MD file
for more information.

Copyright 2014-2018 Roger R Labbe Jr.
"""

from jax.random import multivariate_normal as multivariateNormal
from  jax.scipy.stats.multivariate_normal import pdf as multivariateNormalPDF
import jax.numpy as jnp



#Future idea: square root form of filter to force P to remain PD?

def updateMultiDimensional(predictedMean, predictedCovarianceP, predictedObservation, observation, covarianceR, sensingMatrix):
    """
    Updates the kalman filter given an new observation. Uses the more numerically stable Joseph form

    This now ONLY WORKS FOR the multidimensional case.

    Parameters
    ----------

    predictedMean : array, shape(numState)
        The predicted estimate of the physical state
    predictedCovarianceP : array, shape(numState,numState)
        The previous covariance estimate of the physical state
    observation : array, shape(numSenors)
        Observation received
    covarianceR : array, shape(numSen,numSen)
        Covariance of the sensing noise.
    sensingMatrix : array, shape(numSen, numState)
        C matrix

    Returns
    -------

    x : numpy.array
        Posterior state estimate vector

    P : numpy.array
        Posterior covariance matrix
    """

    #Increasing R as stabilizing noise, from of regularization effectively
    #covarianceR *= 1.2

    #predictedObservation = jnp.matmul(sensingMatrix, predictedMean)

    # error (residual) between measurement and prediction
    residual = observation - predictedObservation

    #Compute this only once
    covariancePSensingMatrixTrans = jnp.matmul(predictedCovarianceP,sensingMatrix.T)

    # project system uncertainty into measurement space
    intermediateGainS = jnp.matmul(sensingMatrix, covariancePSensingMatrixTrans) + covarianceR


    # map system uncertainty into kalman gain ONLY WORKS FOR MULTIDIMENSIONAL CASE

    kalmanGainK = jnp.matmul(covariancePSensingMatrixTrans, jnp.linalg.inv(intermediateGainS))


    # predict new x with residual scaled by the kalman gain
    updatedMean = predictedMean + jnp.matmul(kalmanGainK, residual)

    #Use joseph form for covariance propagation to ensure numerical stability
    # P = (I-KH)P(I-KH)' + KRK' instead of shorter but unstable P = (I-KH)P
    identityMinusKH = jnp.eye(len(predictedMean)) - jnp.matmul(kalmanGainK, sensingMatrix)

    updatedCovarianceP = jnp.matmul(jnp.matmul(identityMinusKH, predictedCovarianceP),identityMinusKH.T) + jnp.matmul(jnp.matmul(kalmanGainK,covarianceR),kalmanGainK.T)

    #Stabilizing noise,
    #updatedCovarianceP *= 1.5

    return updatedMean, updatedCovarianceP


#Don't think we need to jit since only called from jitted functions
def computeRelativeLikelihood(observation,predictedObservation, predictedCovarianceP,sensingMatrix,measurementNoiseMatrixR):
    """
    Function that gives the relative likelihood of an observation for the given filter.

    Parameters
    ----------
    observation : array, shape(numSen)
        The observation to get relative likelihood of
    predictedMean : array, shape(numState)
        The mean of the physical state estimate
    predictedCovarianceP : array, shape(numState,numState)
        The covariance of the physical state estimate
    sensingMatrix : array, shape(numSen, numState)
        C matrix
    measurementNoiseMatrixR : array, shape(numSen,numSen)
        Covariance of the sensing noise.
    discretization : float (default=None)

    Returns
    -------
    relativeLikelihood : float
        Relative likelihood of this observation
    """

    #Get the mean of our posterior
    #predictedObservation = jnp.matmul(sensingMatrix,predictedMean)
    #Get covariance of our posterior
    measurementCovariance = jnp.matmul(sensingMatrix,jnp.matmul(predictedCovarianceP,sensingMatrix.T)) + measurementNoiseMatrixR
    #Get relative likelihood
    return multivariateNormalPDF(x=observation, mean=predictedObservation, cov=measurementCovariance)

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

    #We are using a convention where the filter is a n+1xn array, where the filter[0] = mean, filter[1:] = covariance!
    return multivariateNormal(rngKey, filterToSample[0], filterToSample[1:])
