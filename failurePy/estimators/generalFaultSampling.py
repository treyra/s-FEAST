"""
Module of functions that sample in fault space for new particles and check resampling conditions.

Also includes initialization functions

Not fully implemented yet
"""

#Still under development pylint: skip-file
from functools import partial

import jax
import jax.random as jaxRandom
import jax.numpy as jnp

#Do we need to saturate the Gaussian at 0/1? I think so, could this cause issues? Unsure.
@partial(jax.jit, static_argnames=['timeStepK','estimatorF','systemF','physicalStateSubEstimatorF','physicalStateJacobianF','singleParticleResampleMethodF'])
def failureParticleResample(beliefTuple,timeStepK,estimatorF,initialBelief,actionHistory,observationHistory,systemF,systemParametersTuple,
                              physicalStateSubEstimatorF,physicalStateJacobianF,rngKey, sigma=.1,singleParticleResampleMethodF=None):
    """
    Takes in a belief and performs a weighted resample based on gaussian noise.

    Parameters
    ----------
    beliefTuple : tuple
        The belief is represented by an array of failureWeights and the filter array corresponding to each possibleFailure. Contains:
            failureWeights : array, shape(numFailureParticlesP)
                Normalized weighting of the relative likelihood of each failure
            filters : array
                Array of conditional filters on the physical state
            possibleFailureParticles : array, shape(nMaxPossibleFailures,2*numAct+2*numSen)
                Array of the current failure particles (locations in fault space, effectively). Belief failure weights are are over these possibilities
    timeStepK : int
        How many time steps from initial the states we need to propagate filters forward to catch up to current time
    estimatorF : function
        Estimator function to update the beliefs with. Takes batch of filters
    initialBelief : array
        The initial belief state of our estimator. ASSUMPTION: All particles start with uniform probability and the same initial state estimate.
        We are only interested in these 2 components of the initial belief.
        TODO: Relax this assumption
    actionHistory : tuple
        Tuple of actions taken to fast-forward filters
    observationHistory : tuple
        Tuple of actions taken to fast-forward filters
    systemF : function
        Function reference of the system to call to run experiment
    systemParametersTuple : tuple
        Tuple of system parameters needed. See the model being used for details. (ie, linearModel)
    physicalStateSubEstimatorF : function
        Function to update all of the conditional position filters
    physicalStateJacobianF : function
        Jacobian of the model for use in estimating. Can be none if physicalStateSubEstimatorF doesn't need it.
    rngKey : JAX PRNG key
        Key for the PRNG, to be used to create pseudorandomness
    sigma : float (default=.1)
        The sigma on the resampling. Should be provided when constructed if a different value is desired
    singleParticleResampleMethodF : function (default=None)
        Function to perform resampling with. Should be provided when constructed

    Returns
    -------
    resampledFailureParticles : array, shape(nMaxPossibleFailures,2*numAct+2*numSen)
        Array of the updated failure particles (locations in fault space, effectively). Belief failure weights are are over these possibilities
    updatedBeliefTuple : tuple
        The tuple consists of an updated array of weights over possibleFailures and corresponding conditional filters
    """

    resampledFailureParticles = resampledFailureParticlesJitHelper(beliefTuple,singleParticleResampleMethodF,sigma,rngKey)

    #Now rebuild filters with the update particles. USE THE ESTIMATOR FUNCTION
    updatedBeliefTuple = initialBelief #Note this could have the failure particles provided, but the base estimatorF doesn't use these anyways
    for iPastTimeStep in range(timeStepK):
        updatedBeliefTuple = estimatorF(actionHistory[iPastTimeStep],observationHistory[iPastTimeStep],updatedBeliefTuple,resampledFailureParticles,systemF,systemParametersTuple,physicalStateSubEstimatorF,physicalStateJacobianF)

    #Combine into one tuple (mostly for saving the changing particles)
    return updatedBeliefTuple + (resampledFailureParticles, )

@partial(jax.jit, static_argnames=['singleParticleResampleMethodF'])
def resampledFailureParticlesJitHelper(beliefTuple,singleParticleResampleMethodF,sigma,rngKey):
    """
    Helper that can jit without needing to know timeStepK
    """

    failureWeights = beliefTuple[0]
    possibleFailureParticles = beliefTuple[2]

    numFailureParticlesP = len(failureWeights)
    failureIdxs = jnp.arange(numFailureParticlesP)
    rngKey,rngSubKey = jaxRandom.split(rngKey)
    #Sample with replacement from our particles. This will lead to some particle extinction.
    # #IDEA! WHAT IF THE DIFFUSION SIZE IS THE INVERSE OF THE FAILURE WEIGHT? UNLIKELY PARTICLES MOVE MORE? NO PARTICLE EXTINCTION? NEED TO READ PAPERS!!!
    # This idea would probably lead to more likely to search uncovered space, though it is a little hard to conceptualize
    #More ideas: Some sort of probability gradient? Is this too strong of a convergence?

    #Use choices to select the particles
    particlesToResampleIdxs = jaxRandom.choice(rngSubKey,failureIdxs,shape=(numFailureParticlesP,),p=failureWeights,replace=True)

    #Parallelize the diffusion step
    batchGaussianDiffusion = jax.vmap(singleParticleResampleMethodF,in_axes=[0,None,0]) #IDEA!!! We can swap this line out! Get polymorphic behavior! (Can specify re-sample method)
    batchRngKeys = jaxRandom.split(rngKey,num=numFailureParticlesP)
    covariance = sigma * jnp.eye(len(possibleFailureParticles[0]))

    resampledFailureParticles = batchGaussianDiffusion(possibleFailureParticles[particlesToResampleIdxs],covariance,batchRngKeys)

    return resampledFailureParticles


def makeFailureParticleResampleF(singleParticleResampleMethodF,sigma=.1):
    """
    Constructor for failureParticleResample functions. Binds resample method and sigma
    """
    def failureParticleResampleF(beliefTuple,timeStepK,estimatorF,initialFilters,actionHistory,observationHistory,systemF,systemParametersTuple,
                              physicalStateSubEstimatorF,physicalStateJacobianF,rngKey):
        return failureParticleResample(beliefTuple,timeStepK,estimatorF,initialFilters,actionHistory,observationHistory,systemF,systemParametersTuple,
                              physicalStateSubEstimatorF,physicalStateJacobianF,rngKey, sigma,singleParticleResampleMethodF)
    return failureParticleResampleF




def gaussianDiffusion(failureParticle,covariance,rngKey,clipMin=0,clipMax=1):
    """
    vmap helper for gaussianDiffusionResample
    """

    particle = jaxRandom.multivariate_normal(rngKey,failureParticle,covariance)
    return jnp.clip(particle,clipMin,clipMax)






@jax.jit
def maxRatioResampleCheck(beliefTuple,threshold):
    """
    Takes in a belief and checks if highest versus lowest weight is above a threshold

    Parameters
    ----------
    beliefTuple : tuple
        The belief is represented by an array of failureWeights and the filter array corresponding to each possibleFailure. Contains:
            failureWeights : array, shape(numFailureParticlesP)
                Normalized weighting of the relative likelihood of each failure
            filters : array
                Array of conditional filters on the physical state
    threshold : float
        maximum ratio that can exist between max and min particle weights

    Returns
    -------
    resampleFlag : boolean
        Whether the maxRatio threshold is exceeded
    """

    weights = beliefTuple[0]
    maxWeight = jnp.max(weights)
    minWeight = jnp.min(weights)
    return maxWeight/minWeight > threshold

    #NEED TO INCORPORATE CONVERGENCE INTO REWARD: Tree needs to see how "close" the particles are together. Need to maybe add regularization or something like that
    #to make the tree avoid overfitting, or otherwise guarantee we don't converge to local minimum? Stabilizing noise probably has a role here

def makeMaxRatioResampleCheck(threshold):
    """
    Makes maxRatioResampleCheck for given threshold
    """

    def failureParticleResampleCheckF(beliefTuple):
        return maxRatioResampleCheck(beliefTuple,threshold)
    return failureParticleResampleCheckF

def effectiveNumberOfParticlesResampleCheck(thresholdPercent = .66):
    """
    Uses the condition suggested here: https://ieeexplore.ieee.org/abstract/document/5546308
    """
    pass

def neverResampleCheck(beliefTuple):
    """
    Dummy function that never performs resampling (for static particle testing)
    """

    return False


def randomInitialFailureParticles(numFailureParticles, numAct,numSen,rngKey,dummyProvidedFailure):
    """
    Simple way of initializing the failure particles by randomly sampling belief space.
    """
    numFallibleComponents = numAct + numSen
    #Each has degradation and bias
    return jaxRandom.uniform(rngKey,shape=(numFailureParticles,2*numFallibleComponents))

def biasedRandomInitialFailureParticles(numFailureParticles, numAct,numSen,rngKey,dummyProvidedFailure):
    """
    Simple way of initializing the failure particles by randomly sampling belief space and biasing towards nominal
    """

    numFallibleComponents = numAct + numSen
    #Each has degradation and bias, but bias these towards 0, 1/2 chance of being nominal
    scaledFailureParticles =  2* jaxRandom.uniform(rngKey,shape=(numFailureParticles,2*numFallibleComponents))
    scaledFailureParticles -= 1
    return jnp.clip(scaledFailureParticles,a_min=0,a_max=1)

def biasedRandomInitialFailureParticlesRedundantBiases(numFailureParticles, numAct,numSen,rngKey,providedFailure):
    """
    Simple way of initializing the failure particles by randomly sampling belief space and biasing towards nominal

    Also ensures the act/sen biases are repeated 5x times
    """
    numFallibleComponents = numAct + numSen
    #Each has degradation and bias, but bias these towards 0, 1/2 chance of being nominal
    scaledFailureParticles =  2* jaxRandom.uniform(rngKey,shape=(numFailureParticles,2*numFallibleComponents))
    scaledFailureParticles -= 1
    failureParticles = jnp.clip(scaledFailureParticles,a_min=0,a_max=1)

    #Add provided failure to the list
    if providedFailure is not None:
        failureParticles = failureParticles.at[0].set(providedFailure)

    #For validating
    #jnp.set_printoptions(threshold=jnp.inf)
    #jnp.set_printoptions(linewidth=jnp.inf)

    #Now make each bias have 5x redundancy
    numDiffBiases = int(numFailureParticles/5)
    for iFailureParticle in range(numFailureParticles-numDiffBiases):
        failureParticles = failureParticles.at[iFailureParticle+numDiffBiases,numAct:2*numAct].set(failureParticles[iFailureParticle % numDiffBiases,numAct:2*numAct])
        failureParticles = failureParticles.at[iFailureParticle+numDiffBiases,2*numAct+numSen:2*numAct+2*numSen].set(failureParticles[iFailureParticle % numDiffBiases,2*numAct+numSen:2*numAct+2*numSen])
        #print(failureParticles[iFailureParticle+numDiffBiases]) For validating

    return failureParticles


@jax.jit
def binaryToDegradationFaults(binaryTrueFaultParticle,rngKey):
    """
    Takes a binary fault in, and changes it to randomly degraded fault on the components identified as failed.

    Note we are using the old convention here of 1 = nominal, 0 = completely degraded. THESE FAULTS SHOULD BE USED WITH THE OLD DYNAMICS MODEL

    Parameters
    ----------
    binaryTrueFaultParticle : array, shape(numAct+numSen)
        A fault particle with a 0/1 for each particle, with 1=nominal, 0=totally failed.
    """

    #Only need to do this every now and then, so I this computation is fast enough
    randomDegradations = jaxRandom.uniform(rngKey,shape=(len(binaryTrueFaultParticle),)) #Create random vals between 0 and 1
    #Only affect components with current fault state = 0
    return jnp.where(binaryTrueFaultParticle == 1, 1, randomDegradations)

#def specifiedInitialFailureParticles()
