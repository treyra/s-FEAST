"""
Module of helper methods called by pipeline.py
"""

import warnings

import jax.numpy as jnp

from failurePy.utility.comb import comb
from failurePy.models.linearModel import simulateSystemWrapper as linearSystemF
from failurePy.models.linearModelGeneralFaults import simulateSystemWrapper as linearSystemGeneralF
from failurePy.models.threeDOFModel import simulateSystemWrapper as threeDOFSystemF
from failurePy.models.threeDOFGeneralFaultModel import simulateSystemWrapper as threeDOFSystemGeneralF

def getExperimentParameters(experimentParamsDict):
    """
    Helper function to get common parameters for each experiment
    """
    numState, numAct, numSen, numAgents = getDimensions(experimentParamsDict["systemF"],experimentParamsDict["systemParametersTuple"],experimentParamsDict["networkFlag"])

    numNFailures = getNumNFailures(experimentParamsDict["nMaxComponentFailures"],numAct,numSen,numAgents=numAgents)

    numWarmStart = experimentParamsDict["numWarmStart"]

    return numState, numAct, numSen, numNFailures, numWarmStart, numAgents

#Conditioning on value of numAct and numSens isn't ideal, but that probably won't change, so could be fine
#Low priority because this is only run once per trial, also need to figure out jitting failureCombinationGenerator first
def generateAllPossibleFailures(numAct,numSen,numNFailures,possibleFailuresIdxes,numAgents=1):
    """
    Method that creates the possible failures

    Parameters
    ----------
    numAct : int
        Number of actuators
    numSen : int
        Number of sensors
    numNFailures : array, shape(nMaxComponentFailures+1)
        Number of failure combinations for each level of failed components
    possibleFailuresIdxes : array, shape(nMaxPossibleFailures)
        The indexes of each failure to generate
    numAgents : int (default=1)
        How many agents are present (distinguishes single from multi-agent)

    Returns
    -------
    possibleFailures : array, shape(nMaxPossibleFailures,numAct+numSen)
        List of possible failures to be considered by the solver. If not exhaustive, these are a random subset. Belief failure weights are are over these possibilities
    """
    #Note we will squeeze at end, so if numAgents=1, we will remove that axis
    numFallibleComponents = numAct+numSen
    possibleFailures = jnp.zeros((len(possibleFailuresIdxes),numAgents,numFallibleComponents))
    for iFailure,possibleFailuresIdx in enumerate(possibleFailuresIdxes):
        possibleFailure = failureCombinationGenerator(numAgents*numAct,numAgents*numSen,numNFailures,possibleFailuresIdx)
        #If more than one agent, need to divide the failure across them to parallel process later
        #If one agent, this will get flattened
        for jAgent in range(numAgents):
            #NEED TO TAKE INTO CONSIDERATION THAT SENSOR FAILURES ARE GENERATED AT THE END
            # Otherwise, nominal sensing flag is broken. TEST THIS
            actuatorFailure = possibleFailure[jAgent*numAct:(jAgent+1)*numAct]
            #Ugly, but negative indexing fails when the index is -0
            sensorFailure = possibleFailure[len(possibleFailure)-(jAgent+1)*numSen:len(possibleFailure)-jAgent*numSen]
            possibleFailures = possibleFailures.at[iFailure,jAgent].set(jnp.concatenate((actuatorFailure,sensorFailure)))
    #Remove numAgents axis if numAgents = 1. Catch value error if numAgents !=1
    try:
        possibleFailures = jnp.squeeze(possibleFailures,axis=1)
    except ValueError:
        pass #Just return the original array then!

    return possibleFailures #Remove numAgents axis if numAgents=1

#but could maybe lead to low memory implementation (store int instead of array)
#Or just jit allowing failureIdx to be conditional? Doesn't seem very efficient though
#Low priority because this is only run once per trial
def failureCombinationGenerator(numAct,numSen,numNFailures,failureIdx):
    """
    Method that generates a failure given the number of components that can fail, max failures, and which failure index to consider.

    Parameters
    ----------
    numAct : int
        Number of actuators
    numSen : int
        Number of sensors
    numNFailures : array, shape(nMaxComponentFailures+1)
        Number of failure combinations for each level of failed components
    failureIdx : int
        Index of the failure in the lexicographic ordering of the failures. This order is defined as lower numbers of failures occurring first, and within
        the same number of failures, failures corresponding to smaller binary numbers occurring first.

    Returns
    -------
    failure : array, shape(numAct+numSen)
        The failure state corresponding to the failureIdx
    """

    #Create nominal failure state (no failures), we'll modify as needed
    failure = jnp.ones(numAct+numSen)

    #Find where failureIdx is relative to the total number of failures, and branch
    #failureIdx runs sequentially over all combinations, starting from 0 failures
    if failureIdx == 0:
        return failure

    if failureIdx < numNFailures[1] + 1:
        failure = failure.at[failureIdx-1].set(0)
        return failure

    #For more than one failure, we use failure combination generator
    #Loop through number of total failures > 2
    for numTotalComponentFailures in jnp.arange(2,len(numNFailures)):
        #Count all possible failures up to and including this numTotalComponentFailures
        # The failureIndexLimit is the last index with numTotalComponentFailures)
        failureIndexLimit = 0
        for iComponentFailureCount in range(numTotalComponentFailures+1):
            failureIndexLimit += numNFailures[iComponentFailureCount]
        #If the failure index is below this, we know how many failures we need!
        #Else, loop further, as we need more component failures!
        if failureIdx < failureIndexLimit:
            #note reverseLexicographicIdx is failureIdx - numNFailures of previous totals, which is failureIdx-failureIndexLimit + numNFailures[numTotalComponentFailures]
            return failureCombinationFromLexicographicOrder(numAct,numSen,numTotalComponentFailures,failureIdx-failureIndexLimit + numNFailures[numTotalComponentFailures])
    #If failureIdx is improperly set above totalPossibleFailures, just return everything failed
    return jnp.zeros(numAct+numSen)

#but could maybe lead to low memory implementation (store int instead of array)
#Low priority because this is only run once per trial
def failureCombinationFromLexicographicOrder(numAct,numSen,numFailures,reverseLexicographicIdx):
    """
    Adapts the algorithm from: https://math.stackexchange.com/questions/1368526/fast-way-to-get-a-combination-given-its-position-in-reverse-lexicographic-or/1368570#1368570

    Parameters
    ----------
    numAct : int
        Number of actuators
    numSen : int
        Number of sensors
    numFailures : int
        Number of failures to return
    failureIdx : int
        Index of the failure in the lexicographic ordering of the failures (for this number of failures!).
        This order is defined as lower numbers of failures occurring first, and within
        the same number of failures, failures corresponding to smaller binary numbers occurring first.

    Returns
    -------
    failure : array, shape(numAct+numSen)
        The failure state corresponding to the reverseLexicographicIdx for this numFailures
    """
    #Create nominal failure state (no failures), we'll modify as needed
    #Since sensors are always last, we only set them as failed if there are enough possible combinations before rolling over
    failure = jnp.ones(numAct+numSen)
    #Loop through the bits in reverse order (MSB first, right side of array)
    for iBit in reversed(range(numAct+numSen)):
        #Compute combinatorial value, or zero if this doesn't make sense (choosing more than we have, which would through an error)
        if iBit >= numFailures >=0:
            combinatorialPlaceValue = comb(iBit,numFailures)
        else:
            combinatorialPlaceValue = 0
        #Check if we should set a failure
        if reverseLexicographicIdx >= combinatorialPlaceValue:
            #If so, decrease the index value
            reverseLexicographicIdx = reverseLexicographicIdx - combinatorialPlaceValue
            failure = failure.at[iBit].set(0)
            #Decrease the number of failures left to set
            numFailures = numFailures - 1
    return failure

def getNumNFailures(nMaxComponentFailures,numAct,numSen,nominalSensing=False,numAgents=1):
    """
    Computes the number of failures for each level of simultaneous failures

    Parameters
    ----------
    nMaxComponentFailures : int
        Max number of simultaneous failures
    numAct : int
        Number of actuators
    num Sen : int
        Number of sensors
    nominalSensing : boolean (default=False)
        If true, no sensor failures will be generated. Achieved by limiting combinations considered,
        and noting that sensors are always last in the failure array.
    numAgents : int (default=1)
        How many agents are present (distinguishes single from multi-agent)

    Returns
    -------
    numNFailures : array, shape(nMaxComponentFailures+1)
        Number of combinations of failures for each level of simultaneous failures
    """

    #Guard against too high an nMaxComponentFailures
    if not nominalSensing:
        numFallibleComponents = (numAct+numSen)*numAgents
    else:
        numFallibleComponents = numAct*numAgents
    if nMaxComponentFailures > numFallibleComponents:
        nMaxComponentFailuresTooHigh = f"{nMaxComponentFailures} maximum component failures exceeds the number of fallible components ({numFallibleComponents})!"
        warnings.warn(nMaxComponentFailuresTooHigh)
        nMaxComponentFailures = numFallibleComponents

    #Get numbers of failures
    numNFailures = jnp.ones(nMaxComponentFailures+1)
    numNFailures = numNFailures.at[1].set(numFallibleComponents)
    for iComponentsFailed in range(nMaxComponentFailures-1): #Faster to just set it (I think for 0 and 1 failures)
        numNFailures = numNFailures.at[iComponentsFailed+2].set(comb(numFallibleComponents,iComponentsFailed+2))

    return numNFailures

def checkFailureIsValid(failure,numSen):
    """
    Checks for double sensor failures, as these can't be solved. (Assumes redundant sensing)

    Parameters
    ----------
    failure : array, shape(numAct+numSen)
        The failure state to validate
    numSen : int
        Number of sensors

    Returns
    -------
    newFailureIfNeeded : array, shape(numAct+numSen)
        The failure state with no double sensor failures if needed, otherwise returns None
    """

    #Check to see if double sensing makes sense. Even number of sensors, needed
    if not numSen % 2 == 0:
        return None
    # Flag for multiple failures on an axis
    oneSensorFailedFlag = False
    newFailureIfNeeded = None

    #Checking for two sensors failed on the same axis, as this is impossible to solve
    for iSen in range(numSen):
        #New axis, reset
        if iSen % 2 == 0:
            oneSensorFailedFlag = False
        #This sensor is failed. Backwards indexing! Watch for off by one errors!
        if failure[-iSen-1] == 0:
            #Already had a sensor failed this axis!
            if oneSensorFailedFlag:
                #Remove this sensor failure. Need to make a new failure if we haven't yet
                if newFailureIfNeeded:
                    newFailureIfNeeded = newFailureIfNeeded.at[-iSen-1].set(1)
                else:
                    newFailureIfNeeded = failure.at[-iSen-1].set(1)
            else:
                oneSensorFailedFlag = True
    return newFailureIfNeeded

def getDimensions(systemF,systemParametersTuple,networkFlag):
    """
    Method that gets the dimensions of the system

    Parameters
    ----------
    systemF : function
        Function reference of the system to call to run experiment
    systemParametersTuple : tuple
        Tuple  containing all system parameters. Relevant parameters are given below (for linear system):
            influenceMatrix : array, shape(numState, numAct)
                B matrix
            sensingMatrix : array, shape(numSen, numState)
                C matrix Future
    networkFlag : bool
        Whether we are in a distributed network or not

    Returns
    -------
    numState : int
        Number of states
    numAct : int
        Number of actuators
    numSen : int
        Number of sensors
    numAgents : int
        Number of agents in system (distributed vs single agent)
    """
    covarianceRIdx = -1
    #Need to determine system type here
    if systemF is linearSystemF or systemF is linearSystemGeneralF:
        #Get size of physical state and num actuators and sensors
        influenceMatrixIdx = 2


        numState = len(systemParametersTuple[influenceMatrixIdx])
        numAgents = 1
        numAct = len(systemParametersTuple[influenceMatrixIdx][0])
        numSen = len(systemParametersTuple[covarianceRIdx])
    elif systemF is threeDOFSystemF or systemF is threeDOFSystemGeneralF:
        positionInfluenceMatrixIdx = 0
        reactionWheelInfluenceMatrixIdx = 2


        numState = 6 + len(systemParametersTuple[reactionWheelInfluenceMatrixIdx]) #Need to add in the wheels to the state
        numAct = len(systemParametersTuple[positionInfluenceMatrixIdx][0]) + len(systemParametersTuple[reactionWheelInfluenceMatrixIdx]) #Wheels count as actuators
        numSen = len(systemParametersTuple[covarianceRIdx])
        numAgents = 1
    elif networkFlag:
        nodePositionsIdx = -1
        numState = len(systemParametersTuple[nodePositionsIdx])
        numAgents = numState
        numAct = 3  #hg, lg, act
        numSen = 1  #sen
    else:
        systemDimensionsNotDefined = f"system function {systemF} does not yet have defined dimensions."
        raise NotImplementedError(systemDimensionsNotDefined)

    return numState,numAct,numSen,numAgents


def diagnoseFailure(beliefTuple,currentFailureParticles):
    """
    Returns the failure diagnosis.

    Known issue. If the failure weights are all nan (such as if the EKF diverged). Then argmax will return 0, and we will erroneously diagnose the correct failure.
    Currently addressed in pipeline as well as saving. Saving is so that reprocessed experiments can fix this.

    Parameters
    ----------
    beliefTuple : tuple
        The belief is represented by an array of failureWeights and the filter array corresponding to each possibleFailure. Contains:
            failureWeights : array, shape(maxFailureParticles)
                Normalized weighting of the relative likelihood of each failure
            filters : array
                Array of conditional filters on the physical state
    currentFailureParticles : array, shape(nMaxFailureParticles,numAct+numSen)
        List of possible failures to be considered by the solver. If not exhaustive, these are a random subset. Belief failure weights are are over these possibilities

    Returns
    -------
    diagnosis : array, shape(numAct+numSen)
        Failure believed to be afflicting the s/c
    """
    failureWeightsIdx = 0
    failureIdx = jnp.argmax(beliefTuple[failureWeightsIdx])

    return currentFailureParticles[failureIdx]
