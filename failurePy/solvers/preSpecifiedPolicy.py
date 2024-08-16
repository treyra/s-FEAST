"""
Debug policy module
"""
import jax.numpy as jnp

#Debug class, so disable much of pylint
class PreSpecifiedPolicy: # pylint: disable=too-few-public-methods
    """
    Debug policy, uses class object to deterministically apply a set of specified actions
    """
    def __init__(self, actionList=None):
        """
        Create the pre-specified list of actions to take

        Parameters
        ----------
        actionList : List (default=None)
            List of actions to take, if any. If not provided hard-coded default used
        """

        if actionList is None:
            #actionList = []
            ##1D pre-specified
            #actionList = [ jnp.array([1,1,0,0]),
            #              jnp.array([0,0,0,1]),
            #              jnp.array([0,0,1,0]),
            #              jnp.array([0,0,1,1]),
            #              jnp.array([1,1,0,0]),
            #              jnp.array([0,1,0,0]),
            #              jnp.array([1,0,0,0]),
            #              jnp.array([1,1,0,0]),
            #              jnp.array([0,1,0,0]),
            #              jnp.array([0,0,1,0]),
            #              jnp.array([0,0,1,1]),
            #              jnp.array([0,0,1,1]),
            #              jnp.array([1,1,0,0]),
            #              jnp.array([1,1,0,0]),
            #              jnp.array([1,1,0,0]),
            #                ]


            #Sequential test
            #actionList = [ jnp.array([1,0,0,0,0,0,0,0,0,0]),
            #                 jnp.array([0,1,0,0,0,0,0,0,0,0]),
            #                 jnp.array([0,0,1,0,0,0,0,0,0,0]),
            #                 jnp.array([0,0,0,1,0,0,0,0,0,0]),
            #                 jnp.array([0,0,0,0,1,0,0,0,0,0]),
            #                 jnp.array([0,0,0,0,0,1,0,0,0,0]),
            #                 jnp.array([0,0,0,0,0,0,1,0,0,0]),
            #                 jnp.array([0,0,0,0,0,0,0,1,0,0]),
            #                 jnp.array([0,0,0,0,0,0,0,0,1,0]),
            #                 jnp.array([0,0,0,0,0,0,0,0,0,1]),
            #                ]

            #All rotations (if no failures of course)
            actionList = [ jnp.array([1,0,1,0,0,0,0,0,0,0]),
                             jnp.array([1,0,1,0,0,0,0,0,0,0]),
                            jnp.array([0,0,0,0,0,0,0,0,0,0]),
                            jnp.array([0,0,0,0,0,0,0,0,0,0]),
                             jnp.array([0,1,0,1,0,0,0,0,0,0]),
                             jnp.array([0,1,0,1,0,0,0,0,0,0]),
                            jnp.array([0,0,0,0,0,0,0,0,0,0]),
                            jnp.array([0,0,0,0,0,0,0,0,0,0]),
                             jnp.array([0,0,0,0,1,0,1,0,0,0]),
                             jnp.array([0,0,0,0,1,0,1,0,0,0]),
                            jnp.array([0,0,0,0,0,0,0,0,0,0]),
                            jnp.array([0,0,0,0,0,0,0,0,0,0]),
                             jnp.array([0,0,0,0,0,1,0,1,0,0]),
                             jnp.array([0,0,0,0,0,1,0,1,0,0]),
                            jnp.array([0,0,0,0,0,0,0,0,0,0]),
                            jnp.array([0,0,0,0,0,0,0,0,0,0]),
                             jnp.array([0,0,0,0,0,0,0,0,1,1]),
                             jnp.array([0,0,0,0,0,0,0,0,1,1]),
                             jnp.array([0,0,0,0,0,0,0,0,-1,-1]),
                             jnp.array([0,0,0,0,0,0,0,0,-1,-1]),
                            ]
            #Tom's actuation idea
           # actionList = [Action(.1* np.array([1,1,1,1,1,1,1,1,50,50])),
           #               Action(.1* np.array([1,1,1,1,1,1,1,1,50,50])),
           #               Action(.1* np.array([1,1,1,1,1,1,1,1,50,50])),
           #               Action(.1* np.array([1,0,0,0,0,0,0,0,0,0])),
           #               Action(.1* np.array([0,1,0,0,0,0,0,0,0,0])),
           #               Action(.1* np.array([0,0,1,0,0,0,0,0,0,0])),
           #               Action(.1* np.array([0,0,0,1,0,0,0,0,0,0])),
           #               Action(.1* np.array([0,0,0,0,1,0,0,0,0,0])),
           #               Action(.1* np.array([0,0,0,0,0,1,0,0,0,0])),
           #               Action(.1* np.array([0,0,0,0,0,0,1,0,0,0])),
           #               Action(.1* np.array([0,0,0,0,0,0,0,1,0,0])),
           #               Action(.1* np.array([0,0,0,0,0,0,0,0,50,0])),
           #               Action(.1* np.array([0,0,0,0,0,0,0,0,0,50])),
           #               ]

        self.actionList = actionList
        self.tree = {():None}

    #Made to be compatible, so all arguments ignored
    def takeNextAction(self,beliefTuple,solverParametersTuple,possibleFailures,systemF,systemParametersTuple,rewardF,estimatorF, # pylint: disable=unused-argument
                       physicalStateSubEstimatorF,physicalStateJacobianF,physicalStateSubEstimatorSampleF,nSimulationsPerTree,rngKey): # pylint: disable=unused-argument
        """
        Compatible with SFEAST, but pre determined result

        Parameters
        ----------
        beliefTuple : tuple
            The belief is represented by an array of failureWeights and the filter array corresponding to each possibleFailure. Contains:
                failureWeights : array, shape(maxPossibleFailures)
                    Normalized weighting of the relative likelihood of each failure
                filters : array
                    Array of conditional filters on the physical state
        solverParametersTuple : tuple
            List of solver parameters needed. None needed, but availableActions included for completeness
                availableActions : array, shape(maxNumActions,numAct)
                    Array of actions that can be taken. First action is always null action
        possibleFailures : array, shape(maxPossibleFailures,numAct+numSen)
            Array of possible failures to be considered by the solver. If not exhaustive, these are a random subset. Belief failure weights are are over these possibilities
        systemF : function
            Function reference of the system to call to run experiment. Not used here, but provided to make compatible with marginal filter
        systemParametersTuple : tuple
            Tuple of system parameters needed
            Abstracted for the system function
        rewardF : function
            Reward function to evaluate the beliefs with
        estimatorF : function
            Estimator function to update the beliefs with. Takes batch of filters
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
        rootNode : None
            No belief node for random policy
        """

        return self._takeNextAction()

    def _takeNextAction(self):
        """
        Internal action method, takes no arguments
        """

        if len(self.actionList) > 0:

            action =  self.actionList.pop(0)
        else:
            #HACK to allow repeated actions
            self.__init__() # pylint: disable=unnecessary-dunder-call
            action, dummyRootNode = self._takeNextAction()
            ##action = jnp.array([0,0,0,0,0,0,0,0,0,0])
            #noMoreSpecifiedActions = "Ran out of pre-specified actions."
        #raise IndexError(noMoreSpecifiedActions)

        return action,None
