"""
Non-linear planar dynamics and observing model

Currently implemented in a non-OOP approach to allow for JIT compiling
"""

import jax
import jax.numpy as jnp
from jax import random as jaxRandom
from failurePy.models.modelsCommon import observationModel, makeSingleAxisSensingMatrix
from failurePy.models.linearModel import getRealizedAction, getRealizedNominalMeasurement

@jax.jit #Slight speed up, not the main slow down it appears (already jitting somewhere?)
def simulateSystemWrapper(physicalState,failureState,action,rngKey,systemParametersTuple,noisyPropagationBooleanInt=1):
    """
    Wrapper for the 3DOF model system, so all the systems look the same up to a systemParametersList which is unique to each system
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
    failureState : array, shape(numAct+numSen)
        Failure afflicting the s/c
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

    nextObservation = observationModel(nextPhysicalState,failureState,sensingMatrix,getRealizedNominalMeasurement,
                                       covarianceR,rngKey,noisyPropagationBooleanInt)

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
    failureState : array, shape(numAct+numSen)
        Failure afflicting the s/c
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

    #Apply to the actuation
    realizedAction = getRealizedAction(failureState,action)

    #Create Noise
    dynamicsNoise = jaxRandom.multivariate_normal(rngKey,jnp.zeros(len(physicalState)), covarianceQ)

    #Use an rk4 solver for good enough accuracy (I think) that can be jitted
    nominalNextState =  rk4(physicalState,realizedAction,positionInfluenceMatrix,rotationInfluenceMatrix,reactionWheelInfluenceMatrix,positionCoefficientOfFriction,rotationalCoefficientOfFriction, dt)

    return nominalNextState + noisyPropagationBooleanInt * dynamicsNoise #when noisyPropagationBooleanInt is 0, noise is zeroed out!

def rk4(physicalState,realizedAction,positionInfluenceMatrix,rotationInfluenceMatrix,reactionWheelInfluenceMatrix,positionCoefficientOfFriction,rotationalCoefficientOfFriction, dt): #dt is pretty universal, so making an exception for it pylint: disable=invalid-name
    """
    Use an rk4 approximator to accurately propagate the non-linear dynamics

    Parameters
    ----------
    physicalState : array, shape(numState)
        Current physical state of the system to propagate. physicalState = [x, vx, y, vy, theta, omega, w1, w2, ...]
        Where w1, w2 ... are the speeds of the wheels (if any, determined by length of the state vector)
    realizedAction : array, shape(numAct)
        Current action to take after applying the failure (constant)
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

    Returns
    -------

    """

    #RK4 algorithm is y_{n+1} = y_n + 1/6 (k_1 + 2k_2 + 3k_3 + k_4)dt
    #We'll compute each intermediate step (note no time dependence in dynamics!)
    #Since these are local math variables, we will make an exception to the naming convention

    k1 = dynamicsDerivatives(physicalState,realizedAction,positionInfluenceMatrix,rotationInfluenceMatrix,reactionWheelInfluenceMatrix, # pylint: disable=invalid-name
                             positionCoefficientOfFriction,rotationalCoefficientOfFriction)

    k2 = dynamicsDerivatives(dt/2 *k1 + physicalState,realizedAction,positionInfluenceMatrix,rotationInfluenceMatrix,reactionWheelInfluenceMatrix, # pylint: disable=invalid-name
                             positionCoefficientOfFriction,rotationalCoefficientOfFriction)

    k3 = dynamicsDerivatives(dt/2 *k2 + physicalState,realizedAction,positionInfluenceMatrix,rotationInfluenceMatrix,reactionWheelInfluenceMatrix, # pylint: disable=invalid-name
                             positionCoefficientOfFriction,rotationalCoefficientOfFriction)

    k4 = dynamicsDerivatives(dt * k3 + physicalState,realizedAction,positionInfluenceMatrix,rotationInfluenceMatrix,reactionWheelInfluenceMatrix, # pylint: disable=invalid-name
                             positionCoefficientOfFriction,rotationalCoefficientOfFriction)

    return physicalState + 1/6 * (k1 + 2* k2 + 2* k3 + k4) * dt

def dynamicsDerivatives(physicalState,realizedAction,positionInfluenceMatrix,rotationInfluenceMatrix,reactionWheelInfluenceMatrix,positionCoefficientOfFriction,rotationalCoefficientOfFriction):
    """
    Computes the dynamics derivatives at this time step

    Parameters
    ----------
    physicalState : array, shape(numState)
        Current physical state of the system to propagate. physicalState = [x, vx, y, vy, theta, omega, w1, w2, ...]
        Where w1, w2 ... are the speeds of the wheels (if any, determined by length of the state vector)
    realizedAction : array, shape(numAct)
        Current action to take after applying the failure (constant)
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

    Returns
    -------
    stateDerivatives : array(numState)
        Derivative of each state for the given inputs
    """

    #NOTE u is inertial, tau is body

    numThrusters = len(positionInfluenceMatrix[0])

    #Get velocities
    velocityX = physicalState[1]
    velocityY = physicalState[3]
    angularVelocityOmega = physicalState[5]

    #Get rotation matrix
    theta = physicalState[4]
    rotationMatrix = thetaRotationMatrix(theta)


    #Because things are non-linear, we compute the derivatives for spatial, angular and wheel states separately, to make the math easier to follow.
    # This isn't as critical for the 3DOF model, but will make generalizing easier
    inertialSpatialAccelerations = jnp.matmul(rotationMatrix,jnp.matmul(positionInfluenceMatrix,realizedAction[0:numThrusters])) #Already scaled influence by mass.
    accelerationX = inertialSpatialAccelerations[0] - positionCoefficientOfFriction*velocityX #USING ASSUMPTION FRICTION IS THE SAME IN EACH AXIS. If we don't, need to do this before rotation
    #This friction force is to capture average residual friction in our spacecraft simulator, so can be somewhat orientation dependent
    accelerationY = inertialSpatialAccelerations[1] - positionCoefficientOfFriction*velocityY

    #Note that since there is only one axis, reactionWheelInfluenceMatrix is a vector
    #Torque (in body frame) due to thrusters and RW (note - sign is due to Newton's third law, torque is opposite of acceleration) wheels are listed last in the action
    #Already scaled by moment #NOTE: We don't want to do this for higher dimensions, it's a matrix (maybe we still can?)
    bodyAngularAcceleration = -rotationalCoefficientOfFriction*angularVelocityOmega + (
                                jnp.matmul(rotationInfluenceMatrix,realizedAction[0:numThrusters])+jnp.matmul(-reactionWheelInfluenceMatrix,realizedAction[numThrusters:]))

    #This should be positive, b/ torque is opposite of wheel speed change
    wheelAccelerations = realizedAction[numThrusters:]

    #Derivative is [vx, ax, vy, ay, omega, alpha, w1dot, w2dot]
    stateDerivativesMinusWheels = jnp.array([velocityX,accelerationX,velocityY,accelerationY,angularVelocityOmega,bodyAngularAcceleration])

    return jnp.concatenate((stateDerivativesMinusWheels,wheelAccelerations))

def actionDynamicsJacobian(physicalState,failureState,systemParametersTuple):
    """
    The jacobian of the dynamics with respect to the control input.
    Used by the scp solver

    Parameters
    ----------
    physicalState : array, shape(numState)
        Current physical state of the system to propagate
    failureState : array, shape(numAct+numSen)
        Failure afflicting the s/c
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

    #Get needed system parameters
    positionInfluenceMatrix = systemParametersTuple[0]
    rotationInfluenceMatrix = systemParametersTuple[1]
    reactionWheelInfluenceMatrix = systemParametersTuple[2]

    numThrusters = len(positionInfluenceMatrix[0])
    numWheels = len(reactionWheelInfluenceMatrix)

    #Get rotation matrix
    theta = physicalState[4]
    rotationMatrix = thetaRotationMatrix(theta)

    #Set up output
    jacobianMatrix = jnp.zeros((6+numWheels,numThrusters+numWheels)) #Wheels are tracked! (Maybe arguably shouldn't, it's for compatibility with future multi-axis rotation)
    #print(jnp.shape(jacobianMatrix),numThrusters,numWheels)

    #Dynamics are all linear in control, but not in state, so need to factor that in
    # RTheta * Bp * phi_B[thrusters] -> 2 x numThrusters (we left out spatial states here)
    inertialSpatialAccelerationInfluence = jnp.matmul(rotationMatrix,jnp.matmul(positionInfluenceMatrix,jnp.diag(failureState[0:numThrusters])))
    #Br * phi_B[thrusters] -> 1 x numThrusters
    bodyAngularAccelerationThrusterInfluence = jnp.matmul(rotationInfluenceMatrix,jnp.diag(failureState[0:numThrusters]))
    #(G*Jw) * phi_B[thrusters] -> 1x numWheels
    bodyAngularAccelerationWheelInfluence = jnp.matmul(-reactionWheelInfluenceMatrix,jnp.diag(failureState[numThrusters:numThrusters+numWheels]))

    #Spatial portions, state is x, vx, y, vy, theta, omega
    jacobianMatrix = jacobianMatrix.at[1,0:numThrusters].set(inertialSpatialAccelerationInfluence[0,:])
    jacobianMatrix = jacobianMatrix.at[3,0:numThrusters].set(inertialSpatialAccelerationInfluence[1,:])
    #Angular portions
    jacobianMatrix = jacobianMatrix.at[5,0:numThrusters].set(bodyAngularAccelerationThrusterInfluence)
    jacobianMatrix = jacobianMatrix.at[5,numThrusters:].set(bodyAngularAccelerationWheelInfluence)
    #Wheel influence
    jacobianMatrix = jacobianMatrix.at[6:,numThrusters:].set(jnp.eye(numWheels))

    return jacobianMatrix

@jax.jit
def dynamicsJacobianWrapper(physicalState,failureState, nominalAction,systemParametersTuple):
    """
    Wrapper for the 3DOF model system, so all the systems look the same up to a systemParametersList which is unique to each system
    Currently no error checking!!

    Parameters
    ----------
    physicalState : array, shape(numState)
        Current physical state of the system to propagate
    failureState : array, shape(numAct+numSen)
        Failure afflicting the s/c
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
    failureState : array, shape(numAct+numSen)
        Failure afflicting the s/c
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

    #Create failure Matrix on the actuators Phi_B
    actuatorFailureMatrix = jnp.diag(failureState[0:len(positionInfluenceMatrix[0])+len(reactionWheelInfluenceMatrix)])

    #Apply to the actuation
    realizedAction = jnp.matmul(actuatorFailureMatrix,nominalAction)

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


def makeInfluenceMatrices(spacecraftMass,systemMoment,leverArm,numWheels,numAct=None,reactionWheelMoment=None):
    """
    Make the influence matrix of the model, scaled by inertia and moment

    Parameters
    ----------
    systemMass : float
        Mass of s/c
    leverArm : float
        Distance from thrusters to center of mass in each axis, in m (assumed to be uniform for now)
    numWheels : int
        Number of wheels present
    numAct : int (default=None)
        Number of actuators affecting the model. By default we assume 2 in each axis (+/-)
        The number of actuators will be rounded (down) to the nearest value that can be evenly distributed among each axis (ie, divided by dim*2)
    reactionWheelMoment : array, shape(numWheels) (default=None)
        Moment of each reaction wheel affecting the model. By default we assume a uniform value of .05 for each

    Returns
    -------
    positionInfluenceMatrix : array, shape(2, numThrusters)
        Bp matrix. Influence of the thrusters on the position scaled by mass. We choose to break these out so we can deal with rotation of body frame
    rotationInfluenceMatrix : array, shape(numThrusters)
        Br matrix. Influence of the thrusters on the rotation scaled by moment. We choose to break these out so we can deal with rotation of body frame
    reactionWheelInfluenceMatrix : array, shape(numWheels)
        G*Jw matrix. Influence of reaction wheels on the rotation (they don't affect position). We choose to break these out so we can deal with rotation of body frame and stored momentum.
        For 3DOF this is just the ones array
    """

    dim = 2 #Always 2 positional dimensions and one rotational

    #Default behavior
    if numAct is None:
        numAct = 8
        singleAxisInfluenceMatrix = jnp.array([[-1,-1,1,1]]) #NOTE: We now just consider the effects on velocity, don't handle position terms to make stacking up easier later.
    #We assume a symmetric distribution of the actuators, round down to achieve
    else:
        numActPerDirection = int(numAct/(dim*2))
        singleAxisInfluenceMatrix = jnp.zeros((1,2*numActPerDirection))
        #Set half actuators as negative, half as positive
        singleAxisInfluenceMatrix = singleAxisInfluenceMatrix.at[0,:numActPerDirection].set(-1)
        singleAxisInfluenceMatrix = singleAxisInfluenceMatrix.at[0,numActPerDirection:].set(1)

    positionInfluenceMatrix =  jnp.kron(jnp.eye(dim,dtype=int),singleAxisInfluenceMatrix/spacecraftMass)

    #We will alternate the torque of each actuator
    rotationInfluenceMatrix = leverArm/systemMoment * jnp.ones(numAct)
    for iActuator in range(numAct):
        if iActuator % 2 == 1:
            rotationInfluenceMatrix = rotationInfluenceMatrix.at[iActuator].set(-1 * leverArm/systemMoment)

    #All reaction wheels are aligned with the z-axis, so G is identity
    if reactionWheelMoment is None:
        reactionWheelInfluenceMatrix = .05/systemMoment*jnp.ones(numWheels)
    else:
        reactionWheelInfluenceMatrix = reactionWheelMoment/systemMoment

    return positionInfluenceMatrix, rotationInfluenceMatrix, reactionWheelInfluenceMatrix

def makeCalibratedInfluenceMatricesNoWheels(positionInfluence,rotationInfluence):

    """
    Makes the positionInfluenceMatrix and rotationInfluenceMatrix for given thruster position and rotation influence

    Parameters
    ----------
    positionInfluence : float
        Ratio of thrust/mass for each thruster
    rotationInfluence : float
        Ratio of torque/moment of inertia for each thruster

    Returns
    -------
    positionInfluenceMatrix : array, shape(2, numThrusters)
        Bp matrix. Influence of the thrusters on the position scaled by mass. We choose to break these out so we can deal with rotation of body frame
    rotationInfluenceMatrix : array, shape(numThrusters)
        Br matrix. Influence of the thrusters on the rotation scaled by moment. We choose to break these out so we can deal with rotation of body frame
    """

    dim = 2 #Always 2 positional dimensions and one rotational
    numAct = 8
    singleAxisInfluenceMatrix = positionInfluence * jnp.array([[-1,-1,1,1]]) #NOTE: We now just consider the effects on velocity, don't handle position terms to make stacking up easier later.

    positionInfluenceMatrix =  jnp.kron(jnp.eye(dim,dtype=int),singleAxisInfluenceMatrix)

    #We will alternate the torque of each actuator
    rotationInfluenceMatrix = rotationInfluence * jnp.ones(numAct)
    for iActuator in range(numAct):
        if iActuator % 2 == 1:
            rotationInfluenceMatrix = rotationInfluenceMatrix.at[iActuator].set(-rotationInfluence)

    return positionInfluenceMatrix,rotationInfluenceMatrix

def makeSensingMatrix(numWheels, numSen=None):
    """
    Make the sensing matrix of the model

    Parameters
    ----------
    numWheels : int
        Number of reaction wheels. Part of the state, but we don't sense this directly (for now)
    numSen : int (default=None)
        Number of sensors measuring the model. By default we assume 2 in each dimension
        The number of sensors will be rounded (down) to the nearest value that can be evenly distributed among each dimension
        We further assume no direct sensing of velocities
    Returns
    -------
    sensingMatrix : array, shape(numSen,numState)
        C matrix
    """

    dim = 3

    singleAxisSensingMatrix = makeSingleAxisSensingMatrix(dim, numSen)

    sensingMatrixMinusWheels = jnp.kron(jnp.eye(dim,dtype=int),singleAxisSensingMatrix)
    #Need to add zeros for the wheel states

    return jnp.concatenate((sensingMatrixMinusWheels, jnp.zeros((len(sensingMatrixMinusWheels),numWheels))),axis=1)

def thetaRotationMatrix(theta):
    """
    Function that returns the 2D rotation matrix from theta used
    for a matrix multiplication to transform from the inertial
    to body frames

    Parameters
    ----------
    theta : float
        angle in radians of the s/c's rotation

    Returns
    -------
    thetaRotationMatrix : array, shape(2,2)
        Rotation matrix from inertial to body frame
    """

    return jnp.array([[jnp.cos(theta),-jnp.sin(theta)],
                      [jnp.sin(theta), jnp.cos(theta)]])

def thetaRotationMatrixDerivative(theta):
    """
    Function that returns the derivative with respect to theta
    2D rotation matrix from theta used for a matrix multiplication
    to transform from the inertial to body frames

    Parameters
    ----------
    theta : float
        angle in radians of the s/c's rotation

    Returns
    -------
    thetaRotationMatrixDerivative : array, shape(2,2)
        Rotation matrix from inertial to body frame
    """

    return jnp.array([[-jnp.sin(theta),-jnp.cos(theta)],
                      [ jnp.cos(theta),-jnp.sin(theta)]])
