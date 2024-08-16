"""
Sequential Convex Programming (SCP) Baseline.

Adapted from an implementation originally authored by Ben Riviere
"""
#Baselines were not developed to coding style
# pylint:skip-file

import jax.numpy as jnp
from jax import random as jaxRandom
import numpy as onp
import cvxpy as cp

#import plotter
#from controllers.controller import Controller
#from util import make_interp

#Hack to make this work, not generally applicable
from failurePy.models.threeDOFModel import dynamicsDerivatives as threeDOFDynamicDerivatives
from failurePy.models.threeDOFModel import actionDynamicsJacobian as threeDOFActionDynamicsJacobian


def solveForNextAction(beliefTuple,solverParametersTuple,possibleFailures,systemF,systemParametersTuple,rewardF,estimatorF,
                       physicalStateSubEstimatorF,physicalStateJacobianF,physicalStateSubEstimatorSampleF,nSimulationsPerTree,rngKey):
    """
    Function that takes in the current belief tuple, parameters, possible failures and system to determine the next best action to take.
    Uses the SFEAST algorithm

    Parameters
    ----------
    beliefTuple : tuple
        The belief is represented by an array of failureWeights and the filter array corresponding to each possibleFailure. Contains:
            failureWeights : array, shape(maxPossibleFailures)
                Normalized weighting of the relative likelihood of each failure
            filters : array
                Array of conditional filters on the physical state
    solverParametersTuple : tuple
        List of solver parameters needed. Contents are:
            numActuators : int

            safetyFunctionF : float

    possibleFailures : array, shape(maxPossibleFailures,numAct+numSen)
        Array of possible failures to be considered by the solver. If not exhaustive, these are a random subset. Belief failure weights are are over these possibilities
    systemF : function
        Function reference of the system to call to run experiment. Not used here, but provided to make compatible with marginal filter
    systemParametersTuple : tuple
        Tuple of system parameters needed. See the model being used for details. (ie, linearModel)
        Abstracted for the system function
    rewardF : function
        Reward function to evaluate the beliefs with
    estimatorF : function
        Estimator function to update the beliefs with. Takes batch of filters
    physicalStateSubEstimatorF : function
        Function to update all of the conditional position filters
    physicalStateJacobianF : function
        Jacobian of the model for use in estimating.
    physicalStateSubEstimatorSampleF : function
        Samples from the belief corresponding to this estimator
    nSimulationsPerTree : int
        Number of max simulations per tree for the solver to search
    rngKey : JAX PRNG key
        Key for the PRNG, to be used to create pseudorandomness

    Returns
    -------
    action : array, shape(numAct)
        Action to take next
    rootNode : BeliefNode
        Root node of the tree that is now expanded to amount requested (N trajectories)
    """

    #Unpack
    safetyFunctionF,numAct,horizon, actuationLimit = unpackSolverParameters(solverParametersTuple)


    #We sample over possible actions that might satisfy cbf

    #Outline of function.
    # Compute ML failure. In event of a tie (such as at initial state), select randomly
    # Sample within actuation limits (thinking setting any actuator to +/- 5 m/s^2 influence, since empirically, velocity usually below 10m/s^2
    # Compute nominal next state given the most likely failure
    # see if satisfies hcbf(next state) > hcbf(current state), hcbf = h * (10-||v||) -> safe + velocity bounded
    # If fails bound, keep if violation is less than previous best. Resample up to 100 times
    # Use smallest violation if failed to find an action that satisfies the bound.
    # Actually, just set hcbf to h, and progressively widen a until we find one that works or until we have widened 100 times.


    #Get ML failure
    failureWeights = beliefTuple[0]

    mostLikelyFailuresIdxes = jnp.argwhere(failureWeights == jnp.amax(failureWeights))
    #print(len(mostLikelyFailuresIdxes),mostLikelyFailuresIdxes)
    #Dealing with possible ties/lack there of
    if len(mostLikelyFailuresIdxes) > 1:
        #Pick random most likely failure and assume this to be the dynamics
        mostLikelyFailuresIdx = jaxRandom.choice(rngKey,mostLikelyFailuresIdxes)[0]
    #When there are nans (estimator has diverged), the length will be zero, so we'll just pick a failure at random again
    #Using isnan over len(mostLikelyFailuresIdxes) == 0, as we dont' want to ignore a different sort of error
    elif jnp.isnan(jnp.amax(failureWeights)):
        mostLikelyFailuresIdx = jaxRandom.choice(rngKey,jnp.arange(len(failureWeights)))
    else:
        mostLikelyFailuresIdx = mostLikelyFailuresIdxes[0,0]

    mostLikelyFailure = possibleFailures[mostLikelyFailuresIdx]
    #Need to assume a state. Idea, propagate posterior of this failure and require up to 2 sigma (95%!) to be safe?
    # Downside: why not just use full failure and propagate action? Maybe we should? Could re-use our chebyshev bound? With random observations/noise?

    #Best plan so far, select action, simulate 100 times on state/failure sampled from initial belief, evaluate average hcbf as approximate hcbf, check if

    #Now get ML state (first row of associated filter)
    assumedState = beliefTuple[1][mostLikelyFailuresIdx,0]

    action = scpSolveForNextAction(assumedState,mostLikelyFailure, systemF, systemParametersTuple, safetyFunctionF,physicalStateJacobianF,numAct,horizon,actuationLimit)
    return action, None #No tree to return

def scpSolveForNextAction(assumedState,mostLikelyFailure, systemF,systemParametersTuple, safetyFunctionF,physicalStateJacobianF,numAct,horizon,actuationLimit):
    """
    Re-implementation of the scp solver to re-solve at every time step. Not constrained on the terminal state

    """

    #No goal state
    xSolution, uSolution, tSolution, scpPlotInfo = solveFiniteHorizonSafeOptControlProb(assumedState, 0, mostLikelyFailure, systemF, systemParametersTuple, safetyFunctionF,physicalStateJacobianF,numAct,horizon,actuationLimit)
    #Now always returning valid control, but it might be empty
    #if xSolution is None:
    #    return onp.nan * onp.array(numAct)

    #self.scp_plot_info = scpPlotInfo
    #if self.debugPlotFlag: self.plot_scp(scpPlotInfo)
    return uSolution[0]


# finite horizon terminal constraint optimal control problem: FHTCOCP
def solveFiniteHorizonSafeOptControlProb(x0, t0, mostLikelyFailure, systemF, systemParametersTuple, safetyFunctionF,physicalStateJacobianF,numAct,horizon,actuationLimit, eta=2, verbose=False):
    """
    Solve the scp problem

    Parameters
    ----------
    x0 : array, shape(numStates)
        Initial condition
    t0 : float
        Initial time
    systemParametersTuple : tuple
        Tuple of system parameters needed. Assumed to be 3DOF nonlinear system
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

    """

    dt = systemParametersTuple[6]

    tSolution = onp.arange(t0, horizon, dt)
    nonLinearCorrectionFlag = False

    #Guess unforced trajectory
    xReference, uReference = initialNoForceGuess(tSolution, x0, numAct,mostLikelyFailure,systemF,systemParametersTuple)
    xSolutionsList = [xReference]
    uSolutionsList = [uReference]
    virtualControlMagnitudeList = []
    xDifferenceList = []
    status = "running"
    maxNumIter = 20
    #Start with no virtual control, since initial guess shouldn't need it
    #print(onp.shape(xReference),onp.shape(uReference),onp.shape(x0))
    if verbose:
        print("xRef: ",xReference)
        print("uRef: ",uReference)
    virtualControl = onp.zeros((len(tSolution),len(x0)))
    for k in range(maxNumIter):
        if verbose:
            print("scp iter: {}/{}".format(k, maxNumIter))
        xSolution, uSolution, virtualControl, dummyTimeSolution = solveLinearizedFiniteHorizonOptControlProb(x0, xReference, uReference, tSolution, eta,systemParametersTuple,actuationLimit,mostLikelyFailure, safetyFunctionF,physicalStateJacobianF,horizon,virtualControl)
        if xSolution is None:
            status = "failed"
            if verbose:
                print("scp status: ",status)
                print("nextAction: ", uSolution[0])
            break
        if nonLinearCorrectionFlag:
            xSolution, uSolution = scpRollout(x0, tSolution, uSolution)
        if is_converged(xSolution, xReference):
            status = "solved"
        xDifference = onp.linalg.norm(onp.array(xSolution) - onp.array(xReference), axis=1)
        virtualControlMagnitude = onp.linalg.norm(onp.array(virtualControl), axis=1)
        xReference, uReference = xSolution, uSolution
        #eta = beta * eta
        xSolutionsList.append(xReference)
        uSolutionsList.append(uReference)
        virtualControlMagnitudeList.append(virtualControlMagnitude)
        xDifferenceList.append(xDifference)
        if status in ["failed","solved"]:
            if verbose:
                print("scp status: ",status)
                print("nextAction: ", uSolution[0])
            break
    scp_plot_info = {
        "x0" : x0,
        "xd" : onp.nan,
        "ts" : tSolution,
        "xss" : xSolutionsList,
        "uss" : uSolutionsList,
        "exss" : xDifferenceList,
        "vss" : virtualControlMagnitudeList,
    }
    return xSolution, uSolution, tSolution, scp_plot_info

def initialNoForceGuess(tSolution, x0, numAct,mostLikelyFailure,systemF,systemParametersTuple):
    """
    Makes initial guess of the solution by applying provided control sequences
    or modeling as a straight line with no forcing

    Parameters
    ----------
    tSolution : float
        Time steps the solution should be defined along
    x0 : array, shape(numStates)
        Initial state
    xGoal : array, shape(numStates)
        Desired end state

    Returns
    -------
    xReference : array,
    """

    # guess (0) straight line, (1) unforced trajectory
    # Either way, will put zeros as the reference control
    uZero = onp.zeros(numAct)

    K = len(tSolution)

    uReference = [uZero for dummyIdx in range(K)]
    xReference, dummyControls = scpRollout(x0, tSolution, uReference,mostLikelyFailure,systemF,systemParametersTuple)
    return onp.array(xReference), onp.array(uReference)

def is_converged(xSolution, xReference):
    scale = 1
    tolerance = 1e-2
    scaled_abs_errors = [onp.abs(scale*x - \
       scale*xbar) for (x,xbar) in zip(xSolution, xReference)]
    scaled_abs_errors_arr = onp.array(scaled_abs_errors) # nt x n x 1
    return not (onp.sum(scaled_abs_errors_arr > tolerance) > 0)

# linearized finite horizon terminal constraint optimal control problem: LFHTCOCP
def solveLinearizedFiniteHorizonOptControlProb(x0, xReference, uReference, tSolution, eta,systemParametersTuple,actuationLimit,mostLikelyFailure, safetyFunctionF,physicalStateJacobianF,horizon, virtualControl = None, verbose=False):
    """
    Set up and solve cvxpy problem

    Parameters
    ----------
    systemParametersTuple : tuple
        Tuple of system parameters needed. Assumed to be 3DOF nonlinear system
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
    """

    dt = systemParametersTuple[6]
    lambdaV = 1
    #set to 1.28155*sigmaW, as this would be 90% threshold if all uncertainty was due to sigmaW
    safetyBuffer = 2*1.28155*systemParametersTuple[-2][0,0] #Will take x sigma

    K = len(tSolution)
    xSolutionCp = cp.Variable((K, len(x0)))
    uSolutionCp = cp.Variable((K, len(uReference[0])))
    virtualControlSolutionCp = cp.Variable((K, len(x0))) # virtual control

    #Warm start to use reference as feasible solution
    xSolutionCp.value = xReference
    uSolutionCp.value = uReference
    if virtualControl is not None:
        virtualControlSolutionCp.value = virtualControl
    else:
        #Might be redundant, but being clear here what the warm start is
        virtualControlSolutionCp.value = onp.zeros((K,len(x0)))

    circularObstacleRadius = 10
    circularObstaclePosition = onp.array([0,-20])
    linearSafeZoneNormalMatrix = onp.array([[1,0],[-1,0],[0,1],[0,-1]])
    linearSafeZoneOffsets = onp.array([25,25,25,25])

    # s cvx
    constraints = []
    for k in range(K-1):
        # dynamics (virtual control)
        A_k, B_k, C_k = singleTaylorExpansion(xReference[k], uReference[k],systemParametersTuple,mostLikelyFailure,physicalStateJacobianF)
        Xdot_k = A_k @ xSolutionCp[k,:] + B_k @ uSolutionCp[k,:] + C_k + virtualControlSolutionCp[k,:]
        constraints.append(
            xSolutionCp[k+1,:] == eulerStep(xSolutionCp[k,:], Xdot_k, dt))
        # trust region
        deltaX = cp.norm(xSolutionCp[k,:]-xReference[k,:])
        deltaU = cp.norm(uSolutionCp[k,:]-uReference[k,:])
        constraints.append(deltaX + deltaU <= eta)
        # state space (None)
        # control space
        constraints.append(uSolutionCp[k,:] <= actuationLimit)
        constraints.append(uSolutionCp[k,:] >= 0)

        # Safety, buffer is hyperparameter (set to 1.28155*sigmaW, as this would be 90% threshold if all uncertainty was due to sigmaW)
        constraints.append(convexifiedCircularObstacleConstraint(xSolutionCp[k,:],xReference[k,:],circularObstacleRadius,circularObstaclePosition) <= -safetyBuffer)
        constraints.append(linearSafeZoneConstraint(xSolutionCp[k,:],linearSafeZoneNormalMatrix,linearSafeZoneOffsets) <= -safetyBuffer)

    # initial/terminal constraints
    constraints.append(xSolutionCp[0,:] == x0)
    #constraints.append(xSolutionCp[-1,:] == xGoal) #unconstrained terminal state
    #constraints.append(uSolutionCp[-1,:] == onp.zeros(len(uReference[0])))

    # augmented cost function
    objective = cp.Minimize(cp.sum_squares(uSolutionCp) + lambdaV * cp.norm(virtualControlSolutionCp))
    # objective = cp.Minimize(cp.sum_squares(Vs))
    # objective = cp.Minimize(cp.sum_squares(Us))

    cvxPyProb = cp.Problem(objective, constraints)
    try:
        sol = cvxPyProb.solve(solver=cp.CVXOPT, warm_start=True, verbose=verbose)
        # sol = prob.solve(solver=cp.GUROBI)
        # sol = prob.solve(solver=cp.MOSEK, verbose=True)
        if cvxPyProb.status in ["optimal"]: #Is this necessary? Think this might fail now that we removed squeezes
            xSolution = xSolutionCp.value
            uSolution = uSolutionCp.value
            virtualControlSolution = virtualControlSolutionCp.value
            #xSolution = [xSolutionCp[k,:][:,onp.newaxis] for k in range(xSolutionCp.shape[0])]
            #uSolution = [uSolutionCp[k,:][:,onp.newaxis] for k in range(uSolutionCp.shape[0])]
            #vSolution = [virtualControlSolutionCp[k,:][:,onp.newaxis] for k in range(virtualControlSolutionCp.shape[0])]
            return xSolution, uSolution, virtualControlSolution, tSolution
        else:
            if verbose:
                print("scp cvx failed to solve, maximizing obstacle safety instead")
                print("prob.status: ", cvxPyProb.status)
                print("x0: ", x0)
                #print("last x: ",xSolutionCp.value)
                #print("last u: ", uSolutionCp.value)
                #print("last virtualU: ", virtualControlSolutionCp.value)
                # print("xbars: ", xbars)
                # print("ubars: ", ubars)
                # print("ts: ", ts)
    except cp.error.SolverError as e:
        if verbose:
            print("error", e)
            #Don't want to return None, as that will just lead to a less helpful error. Will raise instead
            #raise ValueError("Solver failed to run") from e
            print("scp cvx failed to solve, maximizing obstacle safety instead")



    #Try again, maximizing the safety of each time step instead.
    # scvx
    constraints = []
    for k in range(K-1):
        # dynamics (virtual control)
        A_k, B_k, C_k = singleTaylorExpansion(xReference[k], uReference[k],systemParametersTuple,mostLikelyFailure,physicalStateJacobianF)
        Xdot_k = A_k @ xSolutionCp[k,:] + B_k @ uSolutionCp[k,:] + C_k + virtualControlSolutionCp[k,:]
        constraints.append(
            xSolutionCp[k+1,:] == eulerStep(xSolutionCp[k,:], Xdot_k, dt))
        # trust region
        deltaX = cp.norm(xSolutionCp[k,:]-xReference[k,:])
        deltaU = cp.norm(uSolutionCp[k,:]-uReference[k,:])
        constraints.append(deltaX + deltaU <= eta)
        # state space (None)
        # control space
        constraints.append(uSolutionCp[k,:] <= actuationLimit)
        constraints.append(uSolutionCp[k,:] >= 0)

        # Safety, buffer is hyperparameter (set to 1.28155*sigmaW, as this would be 90% threshold if all uncertainty was due to sigmaW)
        # Now obstacle is the objective, not a constraint
        constraints.append(linearSafeZoneConstraint(xSolutionCp[k,:],linearSafeZoneNormalMatrix,linearSafeZoneOffsets) <= -safetyBuffer)

    # initial/terminal constraints
    constraints.append(xSolutionCp[0,:] == x0)
    #constraints.append(xSolutionCp[-1,:] == xGoal) #unconstrained terminal state
    #constraints.append(uSolutionCp[-1,:] == onp.zeros(len(uReference[0])))

    # augmented cost function
    objective = cp.Minimize(convexifiedCircularObstacleConstraint(xSolutionCp[k,:],xReference[k,:],circularObstacleRadius,circularObstaclePosition)
                                + lambdaV * cp.norm(virtualControlSolutionCp))
    # objective = cp.Minimize(cp.sum_squares(Vs))
    # objective = cp.Minimize(cp.sum_squares(Us))

    cvxPyProb = cp.Problem(objective, constraints)
    try:
        sol = cvxPyProb.solve(solver=cp.CVXOPT, warm_start=True, verbose=verbose)
        # sol = prob.solve(solver=cp.GUROBI)
        # sol = prob.solve(solver=cp.MOSEK, verbose=True)
        if cvxPyProb.status in ["optimal"]: #Is this necessary? Think this might fail now that we removed squeezes
            xSolution = xSolutionCp.value
            uSolution = uSolutionCp.value
            virtualControlSolution = virtualControlSolutionCp.value
            #xSolution = [xSolutionCp[k,:][:,onp.newaxis] for k in range(xSolutionCp.shape[0])]
            #uSolution = [uSolutionCp[k,:][:,onp.newaxis] for k in range(uSolutionCp.shape[0])]
            #vSolution = [virtualControlSolutionCp[k,:][:,onp.newaxis] for k in range(virtualControlSolutionCp.shape[0])]
            return xSolution, uSolution, virtualControlSolution, tSolution
    except cp.error.SolverError as e:
        if verbose: print("error", e)

    if verbose: print("Trying maximizing safe zone safety.")


    #Finally try solving with obstacle as constraint, and safety zone as objective

    # scvx
    constraints = []
    for k in range(K-1):
        # dynamics (virtual control)
        A_k, B_k, C_k = singleTaylorExpansion(xReference[k], uReference[k],systemParametersTuple,mostLikelyFailure,physicalStateJacobianF)
        Xdot_k = A_k @ xSolutionCp[k,:] + B_k @ uSolutionCp[k,:] + C_k + virtualControlSolutionCp[k,:]
        constraints.append(
            xSolutionCp[k+1,:] == eulerStep(xSolutionCp[k,:], Xdot_k, dt))
        # trust region
        deltaX = cp.norm(xSolutionCp[k,:]-xReference[k,:])
        deltaU = cp.norm(uSolutionCp[k,:]-uReference[k,:])
        constraints.append(deltaX + deltaU <= eta)
        # state space (None)
        # control space
        constraints.append(uSolutionCp[k,:] <= actuationLimit)
        constraints.append(uSolutionCp[k,:] >= 0)

        # Safety, buffer is hyperparameter (set to 1.28155*sigmaW, as this would be 90% threshold if all uncertainty was due to sigmaW)
        # Now obstacle is the objective, not a constraint
        constraints.append(convexifiedCircularObstacleConstraint(xSolutionCp[k,:],xReference[k,:],circularObstacleRadius,circularObstaclePosition) <= -safetyBuffer)
        #constraints.append(linearSafeZoneConstraint(xSolutionCp[k,:],linearSafeZoneNormalMatrix,linearSafeZoneOffsets) <= -safetyBuffer)

    # initial/terminal constraints
    constraints.append(xSolutionCp[0,:] == x0)
    #constraints.append(xSolutionCp[-1,:] == xGoal) #unconstrained terminal state
    #constraints.append(uSolutionCp[-1,:] == onp.zeros(len(uReference[0])))

    # augmented cost function
    objective = cp.Maximize(cp.sum(linearSafeZoneConstraint(xSolutionCp[k,:],linearSafeZoneNormalMatrix,linearSafeZoneOffsets))
                                - lambdaV * cp.norm(virtualControlSolutionCp))
    # objective = cp.Minimize(cp.sum_squares(Vs))
    # objective = cp.Minimize(cp.sum_squares(Us))

    cvxPyProb = cp.Problem(objective, constraints)
    try:
        sol = cvxPyProb.solve(solver=cp.CVXOPT, warm_start=True, verbose=verbose)
        # sol = prob.solve(solver=cp.GUROBI)
        # sol = prob.solve(solver=cp.MOSEK, verbose=True)
    except cp.error.SolverError as e:
        if verbose: print("error", e)
        #Don't want to return None, as that will just lead to a less helpful error. Will raise instead
    if verbose: print("Giving up and keeping uReference")

    if cvxPyProb.status in ["optimal"]: #Is this necessary? Think this might fail now that we removed squeezes
        xSolution = xSolutionCp.value
        uSolution = uSolutionCp.value
        virtualControlSolution = virtualControlSolutionCp.value
        #xSolution = [xSolutionCp[k,:][:,onp.newaxis] for k in range(xSolutionCp.shape[0])]
        #uSolution = [uSolutionCp[k,:][:,onp.newaxis] for k in range(uSolutionCp.shape[0])]
        #vSolution = [virtualControlSolutionCp[k,:][:,onp.newaxis] for k in range(virtualControlSolutionCp.shape[0])]
        return xSolution, uSolution, virtualControlSolution, tSolution
    else:
        #Give up and return the reference control and nones
        return None, uReference, None, None


def convexifiedCircularObstacleConstraint(physicalState,physicalStateRef,radiusObstaclePlusRadiusSpacecraft,center):
    """
    Function that enforces a circular obstacle constraint.

    Parameters
    ----------
    physicalState : array, shape(numState)
        Physical state of the system to evaluate constraints against.
        NOTE: Assumed to be a double integrator state, so for example, we have x, vx, y, vy, ...
        Only the position for each dimension (x,y,...) will be used to determine collision.
        The first len(center) dimensions will be checked for collision, in the same order
    radiusObstaclePlusRadiusSpacecraft : float
        Radius of obstacle AND the spacecraft. This is because this is the closest the centers can come to each other.
    center : array, shape(numDimensionsObstacle)
        The center of the obstacle. Length of this will be used to determine how many dimensions to check

    Returns
    -------
    constraintReturn : int
        If 0 or greater, constraint is violated.
    """
    #Get the positions out of the physical state
    positionSpaceCraft = getLinearPositionSpaceCraft(physicalState)
    positionSpaceCraftRef = getLinearPositionSpaceCraft(physicalStateRef)

    #Convexified constraint
    return radiusObstaclePlusRadiusSpacecraft*cp.norm(positionSpaceCraftRef-center) - (positionSpaceCraftRef-center).T @ (positionSpaceCraft - center)
    #Non-convex
    ##If within radius, positive, so violated
    #return radiusObstaclePlusRadiusSpacecraft - cp.norm(center-positionSpaceCraft)

def getLinearPositionSpaceCraft(physicalState):
    """
    Gets the dimensions of the spacecraft's position that are relevant to a constraint we are evaluating, which is on linear position
    Assumes 2 wheels, 2 linear axes, and 1 rotation axis

    Parameters
    ----------
    physicalState : array, shape(numState)
        Physical state of the system to evaluate constraints against.
        NOTE: Assumed to be a double integrator state, so for example, we have x, vx, y, vy, ...
        Only the position for each dimension (x,y,...) will be used to determine collision.
        The first len(center) dimensions will be checked for collision, in the same order

    Returns
    -------
    positionSpaceCraft : array, shape(numDimensionsObstacle)
        The relevant position of the SpaceCraft to evaluate the constraint against
    """

    pickOutLinearPositionMatrix = onp.array([[1,0,0,0,0,0,0,0],
                                             [0,0,1,0,0,0,0,0]])
    #For hardware w/o wheels. TODO: make this configurable
    #pickOutLinearPositionMatrix = onp.array([[1,0,0,0,0,0],
    #                                         [0,0,1,0,0,0]])

    return pickOutLinearPositionMatrix @ physicalState #Need to use @ for cvxpy


def linearSafeZoneConstraint(physicalState,normalMatrix,offsetVector):
    """
    Function that enforces a (convex) composition of linear safe zone constraints on the position.
    Defined as Ax - b < 0 for each constraint. We require every constraint is satisfied for safety,
    so return the worst case (most positive or max) from each line


    Parameters
    ----------
    physicalState : array, shape(numState)
        Physical state of the system to evaluate constraints against.
        NOTE: Assumed to be a double integrator state, so for example, we have x, vx, y, vy, ...
        Only the position for each dimension (x,y,...) will be used to determine collision.
        The first len(center) dimensions will be checked for collision, in the same order
    normalMatrix : array, shape(numConstraints,numDimensionsObstacle)
        A Matrix defining the normal vector of each constraint
    offsetVector : array, shape(numConstraints)
        b Vector defining the offset from the origin of each constraint

    Returns
    -------
    worstConstraint : float
        If 0 or greater, at least one constraint is violated.
    """
    #Get the positions out of the physical state
    positionSpaceCraft = getLinearPositionSpaceCraft(physicalState)

    constraintEvaluation = normalMatrix @ positionSpaceCraft - offsetVector

    #If outside safe zone, one of these will be positive
    return constraintEvaluation



def singleTaylorExpansion(xReference, uReference,systemParametersTuple,mostLikelyFailure,physicalStateJacobianF):
    """
    Taylor expansion of the dynamics
    A = jacobian of the dynamics with respect to x, at xReference, uReference
    B = jacobian of the dynamics with respect to u, at xReference, uReference
    C = difference between non-linear and linearized dynamics. xdot = f(x,u) = Ax + Bu + C. Used for "virtual control" to provide a non-linear correction, I think.

    Parameters
    ----------
    systemParametersTuple : tuple
        Tuple of system parameters needed. Assumed to be 3DOF nonlinear system
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

    """
    #print(onp.shape(xReference),onp.shape(mostLikelyFailure),onp.shape(uReference))
    A = physicalStateJacobianF(xReference,mostLikelyFailure, uReference,systemParametersTuple)
    B = threeDOFActionDynamicsJacobian(xReference,mostLikelyFailure,systemParametersTuple)
    numAct = len(uReference)
    realizedAction = jnp.matmul(jnp.diag(mostLikelyFailure[0:numAct]),uReference)
    positionInfluenceMatrix = systemParametersTuple[0]
    rotationInfluenceMatrix = systemParametersTuple[1]
    reactionWheelInfluenceMatrix = systemParametersTuple[2]

    C = threeDOFDynamicDerivatives(xReference,realizedAction,positionInfluenceMatrix,rotationInfluenceMatrix,reactionWheelInfluenceMatrix,0,0)\
        - A @ xReference\
            - B @ uReference
    return onp.array(A), onp.array(B), onp.array(C)

def eulerStep(X_k, Xdot_k, dt):
    """
    Forward Euler approximation of system dynamics
    """

    X_kp1 = X_k + Xdot_k * dt
    return X_kp1

def scpRollout(x0, tSolution, uSequence,mostLikelyFailure,systemF,systemParametersTuple):
    """
    Rolls out the sequence of controls provided

    Parameters
    ----------
    x0 : array, shape(numStates)
        Initial state
    tSolution : array, shape(numTimeSteps)
        Times steps to roll out at
    uSequence : array, shape(numTimeSteps,numActuators)
        Controls to apply at each timeStep

    Returns
    -------
    xSolution : array, shape(numTimeSteps,numStates)
        Resulting sequence of states from the control sequence and initial conditions
    uSequence : array, shape(numTimeSteps,numActuators)
        Controls to apply at each timeStep
    """
    # xs = [x0]
    # for k, t in enumerate(ts[0:-1]):
    for kTimeStep, t in enumerate(tSolution):
        if kTimeStep == 0:
            xSolution = [x0]
        else:
            #Nominal propagation, so dummy key
            rngKey = jaxRandom.PRNGKey(0)
            nextPhysicalState,dummyFailureState,dummyNextObservation = systemF(xSolution[-1],mostLikelyFailure,uSequence[kTimeStep-1],rngKey,systemParametersTuple,noisyPropagationBooleanInt=0)
            xSolution.append(nextPhysicalState)
    return onp.array(xSolution), uSequence

def unpackSolverParameters(solverParametersTuple):
    """
    Helper method to unpack parameters for readability
    """

    #Get solver parameters
    numAct = solverParametersTuple[0]
    safetyFunctionF = solverParametersTuple[1]
    horizon = solverParametersTuple[2]
    actuationLimit = solverParametersTuple[3]

    return safetyFunctionF,numAct,horizon, actuationLimit
