"""
Module containing methods common to more than one model.
"""

import jax.numpy as jnp
from jax import random as jaxRandom

def observationModel(physicalState,failureState,sensingMatrix,getRealizedNominalMeasurementF,covarianceR,rngKey,noisyPropagationBooleanInt):
    """
    Generates observations for most all models, subject to the models nominal measurement model.

    Parameters
    ----------
    physicalState : array, shape(numState)
        Current physical state of the system to propagate
    failureState : array, shape(numAct+numSen)
        Failure afflicting the s/c
    sensingMatrix : array, shape(numSen, numState)
        C matrix
    covarianceR : array, shape(numSen,numSen)
        Covariance of the sensing noise.
    rngKey : JAX PRNG key
        Key for the PRNG, to be used to create pseudorandomness
    noisyPropagationBooleanInt : int
        When 0, noise free propagation is used. Should be 1 or 0 for defined behavior!


    Returns
    -------
    nextObservation : array, shape(numSenors)
        Next observation of the model
    """

    nominalMeasurement = getRealizedNominalMeasurementF(physicalState,failureState,sensingMatrix)

    #Add Gaussian noise
    sensingNoise = jaxRandom.multivariate_normal(rngKey,jnp.zeros(len(sensingMatrix)), covarianceR) #rng.normal(0,self.sigma_v,size=len(mean))

    return nominalMeasurement + noisyPropagationBooleanInt * sensingNoise #when noisyPropagationBooleanInt is 0, noise is zeroed out!


def makeSingleAxisSensingMatrix(dim,numSen=None):
    """
    Make the redundant single axis sensing matrix of the model.
    This is for models that assume redundant sensing on each degree of freedom

    Parameters
    ----------
    dim : int
        Number of dimensions in the model
    numSen : int (default=None)
        Number of sensors measuring the model. By default we assume 2 in each dimension
        The number of sensors will be rounded (down) to the nearest value that can be evenly distributed among each dimension
        We further assume no direct sensing of velocities
    Returns
    -------
    singleAxisSensingMatrix : array, shape(numSen,2)
        Repeated Block of C matrix
    """

    #Default behavior
    if numSen is None:
        singleAxisSensingMatrix = jnp.array([[1,0],
                                               [1,0]])
    #We assume a symmetric distribution of the actuators, round down to achieve
    else:
        numSenPerAxis = int(numSen/(dim))
        singleAxisSensingMatrix = jnp.zeros((numSenPerAxis,2))
        #Set to observe the base state (not derivative in each axis)
        singleAxisSensingMatrix = singleAxisSensingMatrix.at[:,0].set(1)

    return singleAxisSensingMatrix
