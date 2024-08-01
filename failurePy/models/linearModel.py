"""
Linear dynamics and observing model

Currently implemented in a non-OOP approach to allow for JIT compiling
"""

import math
import jax
import jax.numpy as jnp
from jax import random as jaxRandom
from failurePy.models.modelsCommon import observationModel, makeSingleAxisSensingMatrix

@jax.jit
def simulateSystemWrapper(physicalState,failureState,action,rngKey,systemParametersTuple,noisyPropagationBooleanInt=1):
    """
    Wrapper for the linear model system, so all the systems look the same up to a systemParametersTuple which is unique to each system
    Currently no error checking!!

    Parameters
    ----------
    physicalState : array, shape(numState)
        Current physical state of the system to propagate
    failureState : array, shape(numAct+numSen)
        Failure afflicting the s/c
    action : array, shape(numAct)
        Current action to take
    rngKey : JAX PRNG key
        Key for the PRNG, to be used to create pseudorandomness
    systemParametersTuple : tuple
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
    noisyPropagationBooleanInt : int (default=1)
        When 0, noise free propagation is used. Should be 1 or 0 for defined behavior!

    Returns
    -------
    nextPhysicalState : array, shape(numState)
        Next physical state of the model
    nextFailureState : array, shape(numAct+numSen)
        Failure afflicting the s/c (unchanging by this model)
    nextObservation : array, shape(numSenors)
        Next observation of the model
    """

    return simulateSystem(physicalState,failureState, action, systemParametersTuple[0],
                            systemParametersTuple[1],systemParametersTuple[2],
                            systemParametersTuple[3],systemParametersTuple[4], rngKey, noisyPropagationBooleanInt)

def simulateSystem(physicalState,failureState, action, stateTransitionMatrix,controlTransitionMatrix,sensingMatrix,covarianceQ,covarianceR, rngKey, noisyPropagationBooleanInt=1):
    """
    Iterates the model forward one time step dt based on the action chosen and generates resulting observation

    Parameters
    ----------
    physicalState : array, shape(numState)
        Current physical state of the system to propagate
    failureState : array, shape(numAct+numSen)
        Failure afflicting the s/c
    action : array, shape(numAct)
        Current action to take
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
    rngKey : JAX PRNG key
        Key for the PRNG, to be used to create pseudorandomness
    noisyPropagationBooleanInt : int
        When 0, noise free propagation is used. Should be 1 or 0 for defined behavior!


    Returns
    -------
    nextPhysicalState : array, shape(numState)
        Next physical state of the model
    nextFailureState : array, shape(numAct+numSen)
        Failure afflicting the s/c (unchanging by this model)
    nextObservation : array, shape(numSenors)
        Next observation of the model
    """

    #Split rngKey for two different noise processes. These keys are consumed on use! So we need to split first
    rngKey, rngSubKey = jaxRandom.split(rngKey)

    #First we note that by assumption, the failure state doesn't change, so only propagate the physical state
    nextPhysicalState = propagateDynamics(physicalState,failureState,action,stateTransitionMatrix,controlTransitionMatrix,covarianceQ,rngSubKey,noisyPropagationBooleanInt)

    nextObservation = observationModel(nextPhysicalState,failureState,sensingMatrix,getRealizedNominalMeasurement,covarianceR,rngKey,noisyPropagationBooleanInt)

    return (nextPhysicalState,failureState,nextObservation)


def propagateDynamics(physicalState,failureState,action,stateTransitionMatrix,controlTransitionMatrix,covarianceQ,rngKey,noisyPropagationBooleanInt):
    """
    Iterates the dynamics forward a time step

    Parameters
    ----------
    physicalState : array, shape(numState)
        Current physical state of the system to propagate
    failureState : array, shape(numAct+numSen)
        Failure afflicting the s/c
    action : array, shape(numAct)
        Current action to take (constant)
    stateTransitionMatrix : array, shape(numState,numState)
        State transition matrix from previous state to the next state. This depends on dt, and should be recreated if dt changes
    controlTransitionMatrix : array, shape(numState,numAct)
        State Transition matrix resulting from constant control input
    covarianceQ : array, shape(numState,numState)
        Covariance of the dynamics noise.
    rngKey : JAX PRNG key
        Key for the PRNG, to be used to create pseudorandomness
    noisyPropagationBooleanInt : int
        When 0, noise free propagation is used. Should be 1 or 0 for defined behavior!


    Returns
    -------
    nextState : tuple (nextPhysicalState, nextPhi)
        Next state of the model
    """

    realizedAction = getRealizedAction(failureState,action)

    #Create Noise
    dynamicsNoise = jaxRandom.multivariate_normal(rngKey,jnp.zeros(len(physicalState)), covarianceQ)

    #Convolute the state (solution to forced first order diff eq)
    nominalNextState =  jnp.matmul(stateTransitionMatrix,physicalState) + \
                            jnp.matmul(controlTransitionMatrix,realizedAction)
    return nominalNextState + noisyPropagationBooleanInt * dynamicsNoise #when noisyPropagationBooleanInt is 0, noise is zeroed out!

def getRealizedAction(failureState,action):
    """
    Helper method to apply failure and get realized action
    """
    numAct = len(action)

    #Create failure Matrix on the actuators Phi_B
    actuatorFailureMatrix = jnp.diag(failureState[0:numAct])

    realizedAction = jnp.matmul(actuatorFailureMatrix,action)

    return realizedAction

def getRealizedNominalMeasurement(physicalState,failureState,sensingMatrix):
    """
    Helper method to apply failure and get realized observation
    """
    #Make matrix of sensor failures Phi_C
    sensorFailureMatrix = jnp.diag(failureState[-len(sensingMatrix):])

    #Compute the probability of seeing the observation
    #This is a Gaussian around the mean value (given failures)
    nominalMeasurement = jnp.matmul(sensorFailureMatrix,jnp.matmul(sensingMatrix,physicalState))

    return nominalMeasurement

def makeDynamicsMatrix(dim):
    """
    Make the dynamics matrix of the model

    Parameters
    ----------
    dim : int
        Number of dimensions in the model

    Returns
    -------
    dynamicsMatrix : array, shape(numState,numState)
        A matrix
    """

    singleAxisDynamicsMatrix = jnp.array([[0,1],
                                          [0,0]])

    return jnp.kron(jnp.eye(dim,dtype=int),singleAxisDynamicsMatrix)

def makeInfluenceMatrix(dim,spacecraftMass,numAct=None):
    """
    Make the influence matrix of the model

    Parameters
    ----------
    dim : int
        Number of dimensions in the model
    systemMass : float
        Mass of s/c
    numAct : int (default=None)
        Number of actuators affecting the model. By default we assume 2 in each axis (+/-)
        The number of actuators will be rounded (down) to the nearest value that can be evenly distributed among each axis (ie, divided by dim*2)

    Returns
    -------
    influenceMatrix : array, shape(numState,numAct)
        B matrix
    """

    #Default behavior
    if numAct is None:
        singleAxisInfluenceMatrix = jnp.array([[0,0,0,0],
                                               [-1,-1,1,1]])
    #We assume a symmetric distribution of the actuators, round down to achieve
    else:
        numActPerDirection = int(numAct/(dim*2))
        singleAxisInfluenceMatrix = jnp.zeros((2,2*numActPerDirection))
        #Set half actuators as negative, half as positive
        singleAxisInfluenceMatrix = singleAxisInfluenceMatrix.at[1,:numActPerDirection].set(-1)
        singleAxisInfluenceMatrix = singleAxisInfluenceMatrix.at[1,numActPerDirection:].set(1)

    return jnp.kron(jnp.eye(dim,dtype=int),singleAxisInfluenceMatrix/spacecraftMass) #Divide by s/c mass to scale influence!


def makeSensingMatrix(dim,numSen=None):
    """
    Make the sensing matrix of the model

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
    sensingMatrix : array, shape(numSen,numState)
        C matrix
    """

    singleAxisSensingMatrix = makeSingleAxisSensingMatrix(dim, numSen)

    return jnp.kron(jnp.eye(dim,dtype=int),singleAxisSensingMatrix)


def makeStateTransitionMatrices(dynamicsMatrix,dt,influenceMatrix): #dt is pretty universal, so making an exception for it pylint: disable=invalid-name
    """
    Function that makes the state transition matrix and it's integral, assuming
    A is nilpotent and the dimension is less than or equal to 12 (as otherwise we won't be able to compute explicitly)
    This is true for the double integrator systems we are considering. More sophisticated methods can be used for other systems

    Parameters
    ----------
    dynamicsMatrix : array, shape(numState,numState)
        A matrix
    dt : float
        The time between time steps of the experiment

    Returns Matrices
    -------
    stateTransitionMatrix : array, shape(numState,numState)
        State transition matrix from previous state to the next state. This depends on dt, and should be recreated if dt changes
    controlTransitionMatrix : array, shape(numState,numAct)
        State Transition matrix resulting from constant control input
    """

     #Explicitly exponentiate the matrix using a Taylor matrix and relying on nilpotence
    dynamicsMatrixPowers = [jnp.eye(jnp.shape(dynamicsMatrix)[0]),dynamicsMatrix]

    for iMultiplications in range(13):
        dynamicsMatrixPower = jnp.matmul(dynamicsMatrixPowers[iMultiplications+1],dynamicsMatrix)
        #Check if we have hit the zero matrix
        if jnp.all(dynamicsMatrixPower == 0):
            break
        dynamicsMatrixPowers.append(dynamicsMatrixPower)
    if iMultiplications > 11:
        raise ValueError("dynamics matrix (A matrix) not sufficiently  nilpotent (A^12 !=0)")
    #Create the state transition matrix and it's derivative
    stateTransitionMatrix = jnp.zeros(jnp.shape(dynamicsMatrix))
    stateTransitionMatrixIntegral = jnp.zeros(jnp.shape(dynamicsMatrix))

    for iMatrixPowerMinusOne,dynamicsMatrixPower in enumerate(dynamicsMatrixPowers):
        #Taylor series for matrix exponential
        stateTransitionMatrix = stateTransitionMatrix + dynamicsMatrixPower * dt**iMatrixPowerMinusOne/math.factorial(iMatrixPowerMinusOne)
        #Now we compute the integration of the state transition matrix, as if we have this
        #we can compute the next state, leveraging invariance of Amat, B, and the control input
        #Integrated taylor series (note we want integral of e^(-A tau), hence sign)
        stateTransitionMatrixIntegral = stateTransitionMatrixIntegral + dynamicsMatrixPower * (-1)**iMatrixPowerMinusOne * (dt)**(iMatrixPowerMinusOne+1)/math.factorial(iMatrixPowerMinusOne+1)

    controlTransitionMatrix = jnp.matmul(stateTransitionMatrix,jnp.matmul(\
                                stateTransitionMatrixIntegral,influenceMatrix))

    return (stateTransitionMatrix,controlTransitionMatrix)
