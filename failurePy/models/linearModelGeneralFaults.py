"""
Linear dynamics and observing model

Currently implemented in a non-OOP approach to allow for JIT compiling
"""
import jax
import jax.numpy as jnp
from jax import random as jaxRandom
from failurePy.models.modelsCommon import observationModel


@jax.jit
def simulateSystemWrapper(physicalState,failureState,action,rngKey,systemParametersTuple,noisyPropagationBooleanInt=1):
    """
    Wrapper for the linear model system, so all the systems look the same up to a systemParametersTuple which is unique to each system
    Currently no error checking!!

    Parameters
    ----------
    physicalState : array, shape(numState)
        Current physical state of the system to propagate
    failureState : array, shape(2*numAct+2*numSen)
        Failure afflicting the s/c. Now includes constant biases and notation change. 1 = failed, 0 = nominal
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
    failureState : array, shape(2*numAct+2*numSen)
        Failure afflicting the s/c. Now includes constant biases and notation change. 1 = failed, 0 = nominal
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
    failureState : array, shape(2*numAct+2*numSen)
        Failure afflicting the s/c. Now includes constant biases and notation change. 1 = failed, 0 = nominal
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

    Uses general fault model
    """

    numAct = len(action)

    #Create degradation Matrix on the actuators Phi_B. Note notation is now x_k = A x_{k-1} + B u_k + B [-Phi_B Phi_x] [u_k, 1s]^T + w_k
    actuatorDegradationMatrix = jnp.diag(1-failureState[0:numAct])

    #Create bias. Note we are assuming positive bias for these actuators, since we assume thrusters where negative bias doesn't make a lot of sense.
    actuatorBias = failureState[numAct:2*numAct]

    #Apply degradation and bias to the actuation. Note because we assume 0/1 actuation, this all scales cleanly.
    realizedAction = jnp.matmul(actuatorDegradationMatrix,action) + actuatorBias

    return realizedAction

def getRealizedNominalMeasurement(physicalState,failureState,sensingMatrix):
    """
    Helper method to apply failure and get realized observation

    Uses a general observation model, subject to biases and partial degradation

    Parameters
    ----------
    physicalState : array, shape(numState)
        Current physical state of the system to propagate
    failureState : array, shape(numAct+numSen)
        Failure afflicting the s/c
    sensingMatrix : array, shape(numSen, numState)
        C matrix

    Returns
    -------
    nominalMeasurement : array, shape(numSenors)
        Nominal next observation of the model
    """

    numSen = len(sensingMatrix)

    #Make matrix of sensor failures Phi_C
    sensorDegradationMatrix = jnp.diag(1-failureState[-numSen:])

    sensorBias = failureState[-2*numSen:-numSen]

    #Compute the probability of seeing the observation
    #This is a Gaussian around the mean value (given failures)
    nominalMeasurement = jnp.matmul(sensorDegradationMatrix,jnp.matmul(sensingMatrix,physicalState)) + sensorBias

    return nominalMeasurement
