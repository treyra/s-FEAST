"""
Non-linear planar dynamics and observing model, using general fault model

Currently implemented in a non-OOP approach to allow for JIT compiling
"""

import jax
import jax.numpy as jnp
from jax import random as jaxRandom
from failurePy.models.modelsCommon import observationModel
from failurePy.models.linearModelGeneralFaults import getRealizedNominalMeasurement, getRealizedAction
from failurePy.models.threeDOFModel import rk4, thetaRotationMatrixDerivative
from failurePy.models.threeDOFModel import actionDynamicsJacobian as baseActionDynamicsJacobian

@jax.jit #Slight speed up, not the main slow down it appears (already jitting somewhere?)
def simulateSystemWrapper(physicalState,failureState,action,rngKey,systemParametersTuple,noisyPropagationBooleanInt=1):
    """
    Wrapper for the 3DOF model system, so all the systems look the same up to a systemParametersList which is unique to each system
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
            positionInfluenceMatrix : array, shape(2, numThrusters)
                Bp matrix. Influence of the thrusters on the position scaled by mass. We choose to break these out so we can deal with rotation of body frame
            rotationInfluenceMatrix : array, shape(numThrusters)
                Br matrix. Influence of the thrusters on the rotation scaled by moment. We choose to break these out so we can deal with rotation of body frame
            reactionWheelInfluenceMatrix : array, shape(numWheels)
                G*Jw matrix. Influence of reaction wheels on the rotation (they don't affect position). We choose to break these out so we can deal with rotation of body frame and stored momentum.
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

    return simulateSystem(physicalState,failureState, action, systemParametersTuple[0], systemParametersTuple[1],
                            systemParametersTuple[2], systemParametersTuple[3], systemParametersTuple[4], systemParametersTuple[5],
                            systemParametersTuple[6],systemParametersTuple[7],systemParametersTuple[8],
                            rngKey, noisyPropagationBooleanInt=noisyPropagationBooleanInt)

#dt is pretty universal, so making an exception for it
def simulateSystem(physicalState,failureState, action,positionInfluenceMatrix,rotationInfluenceMatrix,reactionWheelInfluenceMatrix,positionCoefficientOfFriction, # pylint: disable=invalid-name
                   rotationalCoefficientOfFriction,sensingMatrix, dt, covarianceQ,covarianceR, rngKey, noisyPropagationBooleanInt=1):
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
    positionInfluenceMatrix : array, shape(2, numThrusters)
        Bp matrix. Influence of the thrusters on the position scaled by mass. We choose to break these out so we can deal with rotation of body frame
    rotationInfluenceMatrix : array, shape(numThrusters)
        Br matrix. Influence of the thrusters on the rotation scaled by moment. We choose to break these out so we can deal with rotation of body frame
    reactionWheelInfluenceMatrix : array, shape(numWheels)
        G*Jw matrix. Influence of reaction wheels on the rotation (they don't affect position). We choose to break these out so we can deal with rotation of body frame and stored momentum.
        For 3DOF this is just the ones array
    positionCoefficientOfFriction : float
        Coefficient of friction that would be in the Ap matrix. Each axis is assumed to be the same coefficient of friction
    rotationalCoefficientOfFriction : float
        Coefficient of friction that would be in the Ar matrix. Since this can vary independent of the position coefficient (for example, based on pad spacing) as it yields a torque,
        we report it separately. We could instead deal with the lever arms, but this is easier and more robust
    dt : float
        The time between time steps of the experiment
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
    nextPhysicalState = propagateDynamics(physicalState,failureState,action,positionInfluenceMatrix,rotationInfluenceMatrix,reactionWheelInfluenceMatrix,
                                          positionCoefficientOfFriction,rotationalCoefficientOfFriction,covarianceQ, dt, rngSubKey,noisyPropagationBooleanInt)

    nextObservation = observationModel(nextPhysicalState,failureState,sensingMatrix,getRealizedNominalMeasurement,covarianceR,rngKey,noisyPropagationBooleanInt)

    return (nextPhysicalState,failureState,nextObservation)

#dt is pretty universal, so making an exception for it
def propagateDynamics(physicalState,failureState,action,positionInfluenceMatrix,rotationInfluenceMatrix,reactionWheelInfluenceMatrix,positionCoefficientOfFriction, # pylint: disable=invalid-name
                      rotationalCoefficientOfFriction,covarianceQ, dt, rngKey,noisyPropagationBooleanInt):
    """
    Iterates the dynamics forward a time step

    Parameters
    ----------
    physicalState : array, shape(numState)
        Current physical state of the system to propagate. physicalState = [x, vx, y, vy, theta, omega, w1, w2, ...]
        Where w1, w2 ... are the speeds of the wheels (if any, determined by length of the state vector)
    failureState : array, shape(2*numAct+2*numSen)
        Failure afflicting the s/c. Now includes constant biases and notation change. 1 = failed, 0 = nominal
    action : array, shape(numAct)
        Current action to take (constant)
    positionInfluenceMatrix : array, shape(2, numThrusters)
        Bp matrix. Influence of the thrusters on the position scaled by mass. We choose to break these out so we can deal with rotation of body frame
    rotationInfluenceMatrix : array, shape(numThrusters)
        Br matrix. Influence of the thrusters on the rotation scaled by moment. We choose to break these out so we can deal with rotation of body frame
    reactionWheelInfluenceMatrix : array, shape(numWheels)
        G*Jw matrix. Influence of reaction wheels on the rotation (they don't affect position). We choose to break these out so we can deal with rotation of body frame and stored momentum.
        For 3DOF this is just the ones array
    positionCoefficientOfFriction : float
        Coefficient of friction that would be in the Ap matrix. Each axis is assumed to be the same coefficient of friction
    rotationalCoefficientOfFriction : float
        Coefficient of friction that would be in the Ar matrix. Since this can vary independent of the position coefficient (for example, based on pad spacing) as it yields a torque,
        we report it separately. We could instead deal with the lever arms, but this is easier and more robust
    covarianceQ : array, shape(numState,numState)
        Covariance of the dynamics noise.
    dt : float
        The time between time steps of the experiment
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

    #Use an rk4 solver for good enough accuracy (I think) that can be jitted
    nominalNextState =  rk4(physicalState,realizedAction,positionInfluenceMatrix,rotationInfluenceMatrix,reactionWheelInfluenceMatrix,positionCoefficientOfFriction,rotationalCoefficientOfFriction, dt)

    return nominalNextState + noisyPropagationBooleanInt * dynamicsNoise #when noisyPropagationBooleanInt is 0, noise is zeroed out!

def actionDynamicsJacobian(physicalState,failureState,systemParametersTuple):
    """
    The jacobian of the dynamics with respect to the control input.
    Used by the scp solver

    Parameters
    ----------
    physicalState : array, shape(numState)
        Current physical state of the system to propagate
    failureState : array, shape(2*numAct+2*numSen)
        Failure afflicting the s/c. Now includes constant biases and notation change. 1 = failed, 0 = nominal
    action : array, shape(numAct)
        Current action to take
    systemParametersTuple : tuple
        Contents are in order:
            positionInfluenceMatrix : array, shape(2, numThrusters)
                Bp matrix. Influence of the thrusters on the position scaled by mass. We choose to break these out so we can deal with rotation of body frame
            rotationInfluenceMatrix : array, shape(numThrusters)
                Br matrix. Influence of the thrusters on the rotation scaled by moment. We choose to break these out so we can deal with rotation of body frame
            reactionWheelInfluenceMatrix : array, shape(numWheels)
                G*Jw matrix. Influence of reaction wheels on the rotation (they don't affect position). We choose to break these out so we can deal with rotation of body frame and stored momentum.
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

    Returns
    -------
    jacobianMatrix : array(numState,numAct)
        Derivative of the each element of the dynamics with respect to each element of the state
    """

    #Only difference between this implementation and previous is failure convention change, so pass in adjusted failure
    # (note we don't need to remove bias, as the previous implementation only considers first numAct failures)

    return baseActionDynamicsJacobian(physicalState,1-failureState,systemParametersTuple)

@jax.jit
def dynamicsJacobianWrapper(physicalState,failureState, nominalAction,systemParametersTuple):
    """
    Wrapper for the 3DOF model system, so all the systems look the same up to a systemParametersList which is unique to each system
    Currently no error checking!!

    Parameters
    ----------
    physicalState : array, shape(numState)
        Current physical state of the system to propagate
    failureState : array, shape(2*numAct+2*numSen)
        Failure afflicting the s/c. Now includes constant biases and notation change. 1 = failed, 0 = nominal
    action : array, shape(numAct)
        Current action to take
    systemParametersTuple : tuple
        Contents are in order:
            positionInfluenceMatrix : array, shape(2, numThrusters)
                Bp matrix. Influence of the thrusters on the position scaled by mass. We choose to break these out so we can deal with rotation of body frame
            rotationInfluenceMatrix : array, shape(numThrusters)
                Br matrix. Influence of the thrusters on the rotation scaled by moment. We choose to break these out so we can deal with rotation of body frame
            reactionWheelInfluenceMatrix : array, shape(numWheels)
                G*Jw matrix. Influence of reaction wheels on the rotation (they don't affect position). We choose to break these out so we can deal with rotation of body frame and stored momentum.
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

    Returns
    -------
    jacobianMatrix : array(numState,numState)
        Derivative of the each element of the dynamics with respect to each element of the state
    """

    #Return the jacobian after pulling out the needed parameters (skip those unneeded)
    return dynamicsJacobian(physicalState, failureState, nominalAction, systemParametersTuple[0], systemParametersTuple[2],systemParametersTuple[3],systemParametersTuple[4])

def dynamicsJacobian(physicalState, failureState, nominalAction, positionInfluenceMatrix,reactionWheelInfluenceMatrix,positionCoefficientOfFriction,rotationalCoefficientOfFriction):
    """
    Compute the Jacobian of the dynamics at the given state

    Parameters
    ----------
    physicalState : array, shape(numState)
        Current physical state of the system to propagate. physicalState = [x, vx, y, vy, theta, omega, w1, w2, ...]
        Where w1, w2 ... are the speeds of the wheels (if any, determined by length of the state vector)
    failureState : array, shape(2*numAct+2*numSen)
        Failure afflicting the s/c. Now includes constant biases and notation change. 1 = failed, 0 = nominal
    nominalAction : array, shape(numAct)
        Current action to take before applying the failure (constant)
    positionInfluenceMatrix : array, shape(2, numThrusters)
        Bp matrix. Influence of the thrusters on the position scaled by mass. We choose to break these out so we can deal with rotation of body frame
    reactionWheelInfluenceMatrix : array, shape(numWheels)
        G*Jw matrix. Influence of reaction wheels on the rotation (they don't affect position). We choose to break these out so we can deal with rotation of body frame and stored momentum.
        For 3DOF this is just the ones array
    positionCoefficientOfFriction : float
        Coefficient of friction that would be in the Ap matrix. Each axis is assumed to be the same coefficient of friction
    rotationalCoefficientOfFriction : float
        Coefficient of friction that would be in the Ar matrix. Since this can vary independent of the position coefficient (for example, based on pad spacing) as it yields a torque,
        we report it separately. We could instead deal with the lever arms, but this is easier and more robust

    Returns
    -------
    jacobianMatrix : array(numState,numState)
        Derivative of the each element of the dynamics with respect to each element of the state
    """

    idxVx = 1
    idxVy = 3
    idxTheta = 4
    idxOmega = 5

    #Get action after faults applied
    realizedAction = getRealizedAction(failureState,nominalAction)

    numThrusters = len(positionInfluenceMatrix[0])
    numWheels = len(reactionWheelInfluenceMatrix)

    #Get rotation matrix derivative
    theta = physicalState[4]
    rotationMatrixDerivative = thetaRotationMatrixDerivative(theta)

    #Construct jacobianMatrix

    #Velocity derivatives with theta (only nonlinear term in 3DOF)
    jacobianMatrixSpatialVelDTheta = jnp.matmul(rotationMatrixDerivative,jnp.matmul(positionInfluenceMatrix,realizedAction[0:numThrusters]))

    #First 4x4 corner is just the original A matrix from 2DOF (Need friction coefficients!)
    singleAxisDynamicsMatrix = jnp.array([[0,1],
                                          [0,-positionCoefficientOfFriction]])
    jacobianSpatialPosVelDPosVel = jnp.kron(jnp.eye(2,dtype=int),singleAxisDynamicsMatrix)

    #First 4 rows
    jacobianSpatialPosVel = jnp.concatenate((jacobianSpatialPosVelDPosVel,jnp.zeros((4,2+numWheels))),axis=1)
    #Set velocity derivatives explicitly
    jacobianSpatialPosVel = jacobianSpatialPosVel.at[idxVx,idxTheta].set(jacobianMatrixSpatialVelDTheta[0])
    jacobianSpatialPosVel = jacobianSpatialPosVel.at[idxVy,idxTheta].set(jacobianMatrixSpatialVelDTheta[1])

    #Remaining rows (only non zero is theta derivative (and friction coefficient!), will set that explicitly)
    jacobianMatrix = jnp.concatenate((jacobianSpatialPosVel,jnp.zeros((2+numWheels,6+numWheels))))
    jacobianMatrix = jacobianMatrix.at[idxTheta,idxOmega].set(1)
    jacobianMatrix = jacobianMatrix.at[idxOmega,idxOmega].set(-rotationalCoefficientOfFriction)


    return jacobianMatrix
