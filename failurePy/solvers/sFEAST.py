"""
POMCP Algorithm adapted to carry the updated belief state forward instead of the approximation,
as this is what our reward is conditioned on. Uses a marginalized filter to do so.

Goal of this implementation is to be purely functional, to allow for jitting of the algorithm.
Currently the tree has to be represented as objects, but there are ideas to do this with arrays instead.
"""
import time
from functools import partial
import jax
import jax.numpy as jnp
from jax import random as jaxRandom
import numpy as onp #Following Jax conventions to be explicit about which implementation of numpy we use

#But feels like if we aren't jitting, we are wasting memory with device arrays? helper methods seem to fix

#Previously used list (*not* dict!) of nodes, coupled with (mutable) dict in decision nodes mapping to child belief nodes
#Idea is that this will allow for rapid traversal down the tree without needing to explicitly store the histories (just need to know what child to go to, and make and and new node when needed)
#Ideas for speed ups, lists instead of dicts for nodes? Immutable nodes? Store history in separate list so don't need mutable dicts in decision nodes? ...

#Can't jit because of logic flow


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
            availableActions : array, shape(maxNumActions,numAct)
                Array of actions that can be taken. First action is always null action
            discretization : float
                Discretization level or scheme
            maxSimulationTime : float
                Max simulation time (can be infinite). NOTE: Currently implemented by breaking loop after EXCEEDING time, NOT a hard cap
            explorationParameter : float
                Weighting on exploration vs. exploitation
            nMaxDepth : int
                Max depth of the tree
            discountFactor : float
                Discount on future rewards, should be in range [0,1]
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
    startTime,availableActions,discretization,maxSimulationTime,explorationParameter,nMaxDepth,discountFactor = unpackSolverParameters(solverParametersTuple)


    #We perform a tree search over possible futures. Create the root node (no observation used to get here)
    #We now allow for probabilistic rewards (usually b/ of safety constraint)
    rngKey, rngSubKey = jaxRandom.split(rngKey)
    rootNode = BeliefNode(None,rewardF(beliefTuple,rngSubKey),beliefTuple,availableActions)

    #Make array of rng keys, to avoid needing to split every loop (in GPU mode this might cause some hanging as we wait for a dispatch)
    rngKeys = jaxRandom.split(rngKey,num=nSimulationsPerTree)

    #Now search repeatedly until we either time out or use all of our simulations NOTE: Time out is NOT immediate
    #for iSimulation in tqdm(range(nSimulationsPerTree)):
    for iSimulation in range(nSimulationsPerTree):
        #Time out
        if (time.time() - startTime) > maxSimulationTime:
            print("Timeout")
            break
        #No longer used
        ##Make rngSubKey, as consumed on use
        #rngKey,rngSubKey = jaxRandom.split(rngKey)
        dummyDiscountedReward = simulate(rootNode, nMaxDepth,discretization,explorationParameter,
                                    availableActions,possibleFailures,systemF,systemParametersTuple,discountFactor,
                                    rewardF,estimatorF,physicalStateSubEstimatorF,physicalStateJacobianF,physicalStateSubEstimatorSampleF,
                                    depth=0,rngKey=rngKeys[iSimulation]) #Get rngKey from precomputed array
        #Simulate returns the discounted reward, currently we don't use if for anything

    #Now select child action that we expect to give us the best value. Do so by having no exploration term
    #NOTE: This will return an untried action if nSimulationsPerTree < maxNumActions. This behavior is undefined!
    return availableActions[rootNode.getBestActionIdx(0)],rootNode


#Don't think I can, have to check if a belief node exists. Look into further, might be able to jit higher level or otherwise combine with previous method to make this work?
#Currently using a helper function to jit what we can do.
def simulate(rootNode, nMaxDepth,discretization,explorationParameter,availableActions,possibleFailures,systemF,systemParametersTuple,
             discountFactor,rewardF,estimatorF,physicalStateSubEstimatorF,physicalStateJacobianF,physicalStateSubEstimatorSampleF,depth,rngKey):
    """
    Recursively simulates the system out to a horizon, branching and/or creating new nodes according to POMCP value function
    Returns new versions of most past parameters

    Parameters
    ---------
    rootNode : BeliefNode
        Node to expand out from
    nMaxDepth : int
        Max depth of the tree
    discretization : float
        Discretization level or scheme
    explorationParameter : float
        Weighting on exploration vs. exploitation
    availableActions : array, shape(maxNumActions,numAct)
        Array of actions that can be taken. First action is always null action
    possibleFailures : array, shape(maxPossibleFailures,numAct+numSen)
        Array of possible failures to be considered by the solver. If not exhaustive, these are a random subset. Belief failure weights are are over these possibilities
    systemF : function
        Function reference of the system to call to run experiment. Not used here, but provided to make compatible with marginal filter
    systemParametersTuple : tuple
        Tuple of system parameters needed. See the model being used for details. (ie, linearModel)
        Abstracted for the system function
    discountFactor : float
        Discount on future rewards, should be in range [0,1]
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
    depth : int
        Depth of the tree so far
    rngKey : JAX PRNG key
        Key for the PRNG, to be used to create pseudorandomness

    Returns
    -------
    discountedReward : float
        Discounted reward for the sequence of actions taken
    """


    #return if we've hit the bottom of the recursion stack
    if depth > nMaxDepth:
        return 0

    #Get belief at this node
    beliefTuple = rootNode.beliefTuple

    #Update visits at this belief

    #Select the action to take
    rngKey,rngSubKey = jaxRandom.split(rngKey) #Make rngSubKey, as consumed on use. Used for randomly returning if multiple actions haven't been trying
    actionIdx = rootNode.getBestActionIdx(explorationParameter,rngSubKey)
    action = availableActions[actionIdx]
    decisionNode = rootNode.children[actionIdx]

    rngKey,rngSubKey = jaxRandom.split(rngKey) #Make rngSubKey, as consumed on use
    nextObservation = simulateHelperFunction(action,possibleFailures,beliefTuple,systemF,systemParametersTuple,physicalStateSubEstimatorSampleF,discretization,rngSubKey)
    #End block to jit

    #Now see if this observation is any of the existing children of the decision node, or make new node. Get reward
     #We now allow for probabilistic rewards (usually b/ of safety constraint)
    rngKey, rngSubKey = jaxRandom.split(rngKey)
    reward, nextBeliefNode = getNextBeliefNodeAndReward(availableActions, possibleFailures, systemF, systemParametersTuple, rewardF, estimatorF,
                                                        physicalStateSubEstimatorF, physicalStateJacobianF, beliefTuple, action, decisionNode, nextObservation,rngSubKey)

    ##THIS ISN'T NEEDED ANYMORE.... and isn't used in the safety proof. Need to re-run?
    ##Safety early return
    #if reward<=0: #If we violate safety constraints, the reward is set to 0 (or a negative penalty)
    #    maxReward=1 #Always true at time of writing, in future if we want to allow rewards to be greater than 1,
    #    return reward - maxReward * depth #subtracts off max possible reward from previous states on this trajectory, ensuring it is negative overall
    #    #So any unsafe trajectory is automatically worse than any safe one.

    #Simulate forward and collect future reward
    futureReward = simulate(nextBeliefNode,nMaxDepth,discretization,explorationParameter,
                            availableActions,possibleFailures,systemF,systemParametersTuple,discountFactor,
                            rewardF,estimatorF,physicalStateSubEstimatorF,physicalStateJacobianF,physicalStateSubEstimatorSampleF,
                            depth=depth+1,rngKey=rngKey) #Don't need to split key since we don't re-use it or loop

    #Get discounted reward for future beliefs
    reward = reward + futureReward*discountFactor

    #Update visit at this root node and chosen decision node by 1 NOTE: THIS IS A CHANGE FROM V1, WHICH UPDATED THE BELIEF NODE, BEFORE THE ACTION WAS PICKED
    rootNode.nVisits += 1
    decisionNode.nVisits += 1

    #Update value of the decision node
    decisionNode.value += (reward-decisionNode.value)/decisionNode.nVisits

    return reward

def getNextBeliefNodeAndReward(availableActions, possibleFailures, systemF, systemParametersTuple, rewardF, estimatorF, physicalStateSubEstimatorF,
                               physicalStateJacobianF, beliefTuple, action, decisionNode, nextObservation,rngKey):
    """
    Helper method to get look through the existing belief nodes, creating a new one as needed, to match the new observation.

    Returns the next belief node and the corresponding reward.
    """
    observationIdx = decisionNode.getIdxOfBeliefChild(nextObservation)

    #Check if this is a new observation!
    if observationIdx == -1:
        #Add to the decision node's children a new belief node!
        #Update belief tuple
        nextBeliefTuple = estimatorF(action,nextObservation,beliefTuple,possibleFailures,systemF,systemParametersTuple,physicalStateSubEstimatorF,physicalStateJacobianF)
        #We now allow for probabilistic rewards (usually b/ of safety constraint)
        reward = rewardF(nextBeliefTuple,rngKey)

        nextBeliefNode = BeliefNode(nextObservation,reward,nextBeliefTuple,availableActions)
        decisionNode.children.append(nextBeliefNode)

    else:
        #Get the belief node
        nextBeliefNode = decisionNode.children[observationIdx]
        reward = nextBeliefNode.reward
    return reward,nextBeliefNode

#MASSIVE speed up, 2.5x faster (leverages the filter arrays)
@partial(jax.jit, static_argnames=['systemF','physicalStateSubEstimatorSampleF'])
def simulateHelperFunction(action,possibleFailures,beliefTuple,systemF,systemParametersTuple,physicalStateSubEstimatorSampleF,discretization,rngKey):
    """
    Performs the simulation step of POMCP in a compiled fashion

    Parameters
    ----------
    values : array, shape(len(availableActions))
        Values of each possible action's decision node.
    visits : array, shape(len(availableActions))
        Visits to each possible action's decision node.
    explorationParameter : float
        Weighting on exploration vs. exploitation
    nVisitsParent : int
        Number of visits to the parent belief node

    Returns
    -------
    actionIdx : int
        Index of the action to take. Since decisionNodeIdxes is sorted the same as availableActions, this corresponds to the index of decisionNodeIdxes that gives the node index of the decision node.
    """

    failureWeightsIdx = 0
    filtersIdx = 1

    #Pick failure from possible, weighted by current belief
    rngKey,rngSubKey = jaxRandom.split(rngKey) #Make rngSubKey, as consumed on use
    failureIdx = jaxRandom.choice(rngSubKey,len(possibleFailures),p=beliefTuple[failureWeightsIdx])
    failureStateSample = possibleFailures[failureIdx]

    #Now use the filter to sample x
    rngKey,rngSubKey = jaxRandom.split(rngKey) #Make rngSubKey, as consumed on use
    physicalStateSample = physicalStateSubEstimatorSampleF(beliefTuple[filtersIdx][failureIdx],rngSubKey)
    #IDEA: Make filters a 3D array. 1st axis, filter, 2nd and 3rd are concatenated array of mean AND covariance. Sub index later!

    #Now simulate the action forward
    rngKey,rngSubKey = jaxRandom.split(rngKey) #Make rngSubKey, as consumed on use
    dummy,dummy,nextObservation = systemF(physicalStateSample,failureStateSample,action,rngSubKey,systemParametersTuple) #We don't care about next physical state or the next failure state

    #Discretize the observation
    discretizationArray = jax.vmap(discretize, in_axes=[0, None])
    nextObservation = discretizationArray(nextObservation,discretization)
    return nextObservation

@jax.jit
def selectBestActionIndex(values,visits,explorationParameter,nVisitsParent,):
    """
    Choses best action to take according to the POMCP value function

    Parameters
    ----------
    values : array, shape(len(availableActions))
        Values of each possible action's decision node.
    visits : array, shape(len(availableActions))
        Visits to each possible action's decision node.
    explorationParameter : float
        Weighting on exploration vs. exploitation
    nVisitsParent : int
        Number of visits to the parent belief node

    Returns
    -------
    actionIdx : int
        Index of the action to take. Since decisionNodeIdxes is sorted the same as availableActions, this corresponds to the index of decisionNodeIdxes that gives the node index of the decision node.
    """

    #Get values using auto vectorization (note nVisitsParent and exploration parameter are not batched)
    batchValue = jax.vmap(pomcpValueFunction, in_axes=[None, 0, 0, None])
    pomcpValues = batchValue(nVisitsParent,visits,values,explorationParameter)

    #Get argmax and return. If there is an exact tie, biases to earlier actions. Because rewards probably aren't exactly the same, not a concern
    return jnp.argmax(pomcpValues)

def pomcpValueFunction(nVisitsParent,nVisitsChild,value,explorationParameter):
    """
    Compute the POMCP value function

    Parameters
    ----------
    nVisitsParent : int
        Number of visits to the parent node so far
    nVisitsChild : int
        Number of visits to the child node so far (if zero returns inf)
    value : float
        The average value of the child node so far
    explorationParameter : float
        Weighting on exploration vs. exploitation

    Returns
    -------
    augmentedValue : float
        The average value of the child node so far augmented by the exploration term
    """

    #Use where to avoid divide by zero error but remain jittable!
    # NOTE: Since we now select zero visit nodes *before* calling pomcpValueFunction, this is probably outdated
    # Since it's a public method though, I think it is wise to have this a guard
    return jnp.where(nVisitsChild == 0, jnp.inf, value + explorationParameter * jnp.sqrt(jnp.log(nVisitsParent)/nVisitsChild))



def discretize(value,discretization):
    """
    Function to discretize the value to specified level

    Parameters
    ----------
    value : float
        Value to be discretized
    discretization : float
        Discretization size

    Returns
    -------
    discreteValue : float
        Discretized value
    """

    return discretization * jnp.round(value/discretization)

#UNUSED
#@jax.jit
def hashableRepresentation(array):
    """
    Method that turns an array to a format that is hashable.

    Parameters
    ----------
    array : array
        Array to turn to a hashable representation

    Returns
    -------
    hashableRepresentation : hashable
        Hashable object that represents the array
    """

    return array.tobytes()

@jax.jit
def nextObservationInNode(nextObservation,decisionNodeObservationList):
    """
    UNUSED, one way to check membership

    Parameters
    ----------
    nextObservation : array, shape(numSen)
        The observation observed after taking the action at this decision node
    decisionNodeObservationList : list
        List of observations seen after taking the action of this decision node.

    Returns
    -------
    nextObservationInNode : boolean
        Whether nextObservation has already been seen (in decisionNodeObservationList) as a result of the action of this decision node
    """
    for observation in decisionNodeObservationList:
        if jnp.array_equal(observation,nextObservation):
            return True

    return False

def unpackSolverParameters(solverParametersTuple):
    """
    Helper method to unpack parameters for readability
    """
    #Get start time (for timeout based execution)
    startTime = time.time()

    #Get solver parameters
    availableActions = solverParametersTuple[0]
    discretization = solverParametersTuple[1]
    maxSimulationTime = solverParametersTuple[2]
    explorationParameter = solverParametersTuple[3]
    nMaxDepth = solverParametersTuple[4]
    discountFactor = solverParametersTuple[5]
    return startTime,availableActions,discretization,maxSimulationTime,explorationParameter,nMaxDepth,discountFactor

#Adding back in class for stateful computation, tree structure seems unnaturally difficult to work out with out when it can grow arbitrarily, will revisit
class Node: #Base class, we're creating a data structure so we don't need public methods here. pylint: disable=too-few-public-methods
    """
    Class representing a Node of the tree search. Parent of decision and belief nodes
    """

    def __init__(self):
        """
        Create a node of the tree
        """
        self.nVisits = 0.0 #Number of visits to this node
        #self.value = 0.0  # average reward
        self.children = []


class BeliefNode(Node):
    """
    Class representing a belief node in the tree search
    """

    def __init__(self, observation, reward, beliefTuple, availableActions):
        """
        Create a belief node of the tree

        Parameters
        ----------
        observation : array, shape(numSen)
            The observation that led to this belief node (used to map back to this belief node)
        reward : float
            Value of the belief at this node (according to confidence metric)
        beliefTuple : tuple
            The belief is represented by an array of failureWeights and the filter array corresponding to each possibleFailure. Contains:
                failureWeights : array, shape(maxPossibleFailures)
                    Normalized weighting of the relative likelihood of each failure
                filters : array
                    Array of conditional filters on the physical state
        availableActions : array, shape(maxNumActions,numAct)
            Array of actions that can be taken. First action is always null action.
        """
        super().__init__()

        self.observation = observation
        self.reward = reward
        self.availableActions = availableActions
        self.beliefTuple = beliefTuple

        #Make children actions. NOTE: We DON'T need to explicitly track the action in the decision nodes,
        # as lists are ordered! The index of the node in children is the index of the action in availableActions
        for iAction in range(len(availableActions)): #pylint: disable=unused-variable
            self.children.append(DecisionNode())

    def getBestActionIdx(self, explorationParameter=0.0, rngKey=None):
        """
        Returns the action with the best (augmented) value function

        Parameters
        ----------
        explorationParameter : float (default = 0)
            Weighting on exploration vs. exploitation. If 0, no exploration, value function is not augmented.
        rngKey : JAX PRNG key (default=None)
            Key for the PRNG, to be used to create pseudorandomness. Note, MUST be provided if any actions are untried

        Returns
        -------
        actionIdx : int
            Index of the action to take. Since decisionNodeIdxes is sorted the same as availableActions,
            this corresponds to the index of decisionNodeIdxes that gives the node index of the decision node.
        """

        #Get the values and visits to each child (use numpy for mutability then convert to jax)
        values = onp.zeros(len(self.availableActions))
        visits = onp.zeros(len(self.availableActions))
        #Iterate over the children and add to the list
        for iChild, child in enumerate(self.children):
            values[iChild] = child.value
            visits[iChild] = child.nVisits

        values = jnp.array(values)
        visits = jnp.array(visits)

        #Check if any have zero visits, if so return one of those randomly. Seems to cause about a 25% slow down in POMCP speed, but much better performance. 63 sims/s vs 85 sims/s
        #Jitting recovers almost all of the speed 83 sims/s, 85 sims/s, with improved performance.
        if jnp.any(visits == 0):
            numZero = int(len(visits) - jnp.count_nonzero(visits))
            return getRandomZeroVisitsIdxHelperFunction(rngKey,visits,numZero)

        #jitted POMCP value function
        return selectBestActionIndex(values,visits,explorationParameter,self.nVisits,)

    def getMostVisitedAction(self):
        """
        Returns the action that was selected the most

        Returns
        -------
        bestAction : array, shape(numAct)
            Action to take next based on POMCP value function
        """
        #Iterate over every child node
        mostVisitedActionIdx = None #Strictly I don't think this line is needed
        mostVisits = -1*jnp.inf
        for iChild, child in enumerate(self.children):
            visits = child.nVisits
            if visits > mostVisits:
                mostVisits = visits
                mostVisitedActionIdx = iChild

        return self.availableActions[mostVisitedActionIdx]

@partial(jax.jit, static_argnames=['numZero'])
def getRandomZeroVisitsIdxHelperFunction(rngKey,visits,numZero):
    """
    Helper function to randomly select an idex from the decision nodes with zero visits

    Parameters
    ----------
    rngKey : JAX PRNG key
        Key for the PRNG, to be used to create pseudorandomness
    visits : array, shape(len(availableActions))
        Visits to each possible action's decision node.
    numZero : int
        Number of zero visit actions

    Return
    ------
    indexAction : int
        Index of the action that hasn't been visited to try
    """
     #jnp.asarray(visits == 0).nonzero() finds sets elements to 1 if visits == 0, 0 if not, and then returns the indices of the nonzero elements, in a tuple, so access it (since 1D)
    return jaxRandom.choice(rngKey,jnp.asarray(visits == 0).nonzero(size=numZero)[0])

class DecisionNode(Node): #We're creating a data structure so we don't need many public methods here. pylint: disable=too-few-public-methods
    """
    Class representing a decision node in the tree search
    """

    def __init__(self):
        """
        Create a decision node of the tree
        """
        super().__init__()

        #Initially we have no idea of the value
        self.value = 0

    def getIdxOfBeliefChild(self, observation):
        """
        Method to determine if this observation has been seen before from this decision node, and return the resulting child if so

        Parameters
        ----------
        observation : array, shape(numSen)
            The observation seen from this decision node

        Returns
        -------
        beliefChildIdx : int
            Index of the child this observation corresponds to. -1 if no child matches
        """

        #Loop through children
        for iChild, child in enumerate(self.children):
            if jnp.all(child.observation == observation):
                return iChild

        #If we fail to find it, new child!
        return -1
