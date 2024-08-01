"""
File the makes a visualization of the evolution of the position and failure belief over time,
In a one dimensional experiment. Used to validate


"""
import os
import warnings
import matplotlib.pyplot as plt
from matplotlib import colormaps
import cv2
import numpy as onp
import jax.numpy as jnp

from failurePy.visualization.visualizationUtilityFunctions import loadBeliefData, setColorCyclerDistinct
from failurePy.visualization.renderPlanarVis import evalMultivariateGaussian

#Ignore matplotlib warnings'
warnings.filterwarnings( "ignore", module = r"matplotlib\..*" )

#Going to just make the plots manually after the experiment, no need to modify pipeline this
def main(savedDataDirPath,experimentName,solverName,nSimulationsPerTree,outputPath,outputSubDirectory,nTrial=0):
    """
    Function that gets the data from the specified experiment and generates a video of the belief evolution over time.
    Intended to visualize the hardware demos

    Parameters
    ----------
    savedDataDirPath : str
        String of the absolute path to the saveDirectory for we are loading the experiment from.
    experimentName : str
        Name of the experiment we are loading
    solverName : str
        The solver to get the data from
    nSimulationsPerTree: int
        The  number of simulations per tree to get data from.
    outputPath : str
        Path to where the output data should be saved.
    outputSubDirectory : str
        Sub directory to save the output data in
    nTrial : int (default=None)
        The trial to get the data from
    """
    #Get the data loaded
    beliefList,initialFailureParticles,failureState,physicalStateList = loadBeliefData(savedDataDirPath,experimentName,solverName,nSimulationsPerTree,nTrial)

    #Visualizing 1D belief

    makeBeliefTimeSeries(beliefList,initialFailureParticles,failureState,physicalStateList,dt=1,fps=10,outputPath=outputPath,outputSubDirectory=outputSubDirectory)



#dt is universal, so making exception
def makeBeliefTimeSeries(beliefList,initialFailureParticles,failureState,physicalStateList,dt=1,fps=10,outputPath="~/Documents/BeliefVisualization",outputSubDirectory="TimeSeries"): # pylint: disable=invalid-name
    """
    Makes time series of provided data, with transitions added in for smoothness.
    Save video to the provided directory. Also uses this directory for temporary storage of the generated frames.

    Parameters
    ----------
    beliefList : list
        List of the beliefs at each time step, the failure particles (if there is any variation) are part of these tuples!
    initialFailureParticles : array, shape(nMaxInitialFailureParticles,numAct+numSen)
        List of possible failures to be considered by the solver. If not exhaustive, these are a random subset. Belief failure weights are are over these possibilities
    failureState : array, shape(numAct+numSen)
        True failure state
    physicalStateList : list
        List of the physical states
    dt : float (default=1)
        The time between time steps of the experiment
    fps : int (default=10)
        The number of frames per second the video should be made with
    outputPath : str (default="~/Documents/BeliefVisualization")
        Path to where the output data should be saved.
    outputSubDirectory : str (default="TimeSeries")
        Sub directory to save the output data in
    """

    #Get the save directory
    outputSubDirectory = os.path.join(outputPath,outputSubDirectory)

    if not os.path.exists(outputSubDirectory):
        os.mkdir(outputSubDirectory)

    currentFailureParticles = initialFailureParticles

    #Loop through and make time step data, we'll average as we transition
    for iData,beliefTuple in enumerate(beliefList): #Need to index directly for transitioning pylint: disable=consider-using-enumerate
        #Check if we updated belief particles
        if len(beliefTuple) == 3:
            currentFailureParticles = beliefTuple[2]
        #Plot each belief, will create transition below
        fig = plotBeliefAndFailureParticles(beliefList[iData],currentFailureParticles,failureState,physicalStateList[iData],iData)

        #Save the figure as a png
        figName = str(iData) + ".png"

        figPath = os.path.join(outputSubDirectory,figName)

        fig.savefig(figPath)
        plt.close()


    #for jFrame in range(dt*fps):
    #    #Transition smoothly if not the first frame
    #    if iData != 0 and jFrame/fps <= 1/4:
    #        beliefWeights = (.5 - 2*jFrame/fps) * beliefList[iData-1] + (2*jFrame/fps +.5) * beliefList[iData]
    #    elif iData != len(beliefList)-1 and jFrame/fps >= 3/4:
    #        beliefWeights = (1 - 2*(jFrame/fps - 3/4)) * beliefList[iData] + (2*(jFrame/fps - 3/4)) * beliefList[iData+1]
    #    else:
    #        beliefWeights = beliefList[iData]





    videoName = os.path.join(outputSubDirectory,'video.mp4')

    #This returns unsorted!! #[img for img in os.listdir(outputSubDirectory) if img.endswith(".png")]

    images = []
    for iImage in range(len(beliefList)):
        imageSubPath = os.path.join(outputSubDirectory,str(iImage) + ".png")
        images.append(cv2.imread(os.path.join(outputSubDirectory, imageSubPath)))
    #Just get one frame to initialize video
    height, width, dummyLayers = images[0].shape
    video = cv2.VideoWriter(videoName, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width,height)) # pylint: disable=no-member

    for iImage, currentImage in enumerate(images):
        for jFrame in range(int(fps*dt)):
            #Check for cross fades
            if iImage != 0 and jFrame < fps/4:
                previousImage = images[iImage-1]
                frame = (.5 - 2*jFrame/fps) * previousImage + (2*jFrame/fps +.5) * currentImage
                frame = frame.astype(onp.uint8) #CONVERT TO UINT8 here!
            elif iImage != len(images)-1 and jFrame/fps >= 3/4:
                nextImage = images[iImage+1]
                frame = (1 - 2*(jFrame/fps - 3/4)) * currentImage + (2*(jFrame/fps - 3/4)) * nextImage
                frame = frame.astype(onp.uint8) #CONVERT TO UINT8 here!
            else:
                frame = currentImage


            video.write(frame)

            cv2.imshow('Frame',currentImage) # pylint: disable=no-member

    cv2.destroyAllWindows() # pylint: disable=no-member
    video.release()


#Plot element wise failure weights (single frame). DON'T show, let calling function handle that, but return fig to save/close
def plotBeliefAndFailureParticles(beliefTuple,currentFailureParticles,failureState,physicalState,iExperimentStep): #Main plotting function, so pylint: disable=too-many-statements,too-many-branches
    """
    Plots the likelihood of each possible failure scenario at the given time step

    Parameters
    ----------
    beliefTuple : tuple
        Tuple of failureParticleWeights, filters, (and failureParticles, if different from the initial failures)
    currentFailureParticles : array, shape(nMaxInitialFailureParticles,numAct+numSen)
        List of possible failures to be considered by the solver. If not exhaustive, these are a random subset. Belief failure weights are are over these possibilities
    failureState : array, shape(numAct+numSen)
        True failure state
    physicalState: array, shape(2)
        The true position and velocity

    Returns
    -------
    fig : matplotlib figure
        Figure object with the failure likelihoods plotted.
    """

    #Assuming 4 actuators, 2 sensors, each which can fail or have bias, so we have 12 failure dimensions
    #Going to visualize it as 6 different 2d plots for each dimension and colored particles, could get messy

    #Allows us to specify non-uniformly sized axes, A is a merged box for the belief
    fig = plt.figure(layout="constrained",figsize=(15,10))
    axDict = fig.subplot_mosaic(
                                """
                                AABDF
                                AACEG
                                """
                                )

    failureWeights = beliefTuple[0]
    #print(failureWeights)
    #filters = beliefTuple[1] #We know this is a 2d gaussian on position and velocity.

    positionX, positionY, pdfPositionBelief = evalBeliefGaussian(beliefTuple, physicalState, 100)

    beliefColorMap = colormaps["viridis"] #.copy()
    #beliefColorMap.set_under('w')
    #beliefColorMap.mappable.set_clim(vmin=.01)
    #Masking out small values of belief to make it so the color map fades to white.
    #pdfPositionBelief = pdfPositionBelief.at[jnp.where(pdfPositionBelief<.005)].set(-jnp.inf)
    pdfPositionBelief = pdfPositionBelief.at[jnp.where(pdfPositionBelief<.005)].set(-jnp.inf)
    beliefPlot = axDict["A"].contourf(positionX, positionY, pdfPositionBelief, cmap=beliefColorMap,zorder=-2)#,vmin=1000) #Should be under unsafe (-1)
    beliefPlot.cmap.set_under('w')
    beliefExtent = 7
    #Move frame as needed to follow belief
    xOffset = (physicalState[0]+3.5)//beliefExtent
    yOffset = (physicalState[1]+3.5)//beliefExtent

    axDict["A"].set_xlim(xmin=beliefExtent*(xOffset-1),xmax=beliefExtent*(xOffset+1))
    axDict["A"].set_ylim(ymin=beliefExtent*(yOffset-1),ymax=beliefExtent*(yOffset+1))
    axDict["A"].set_aspect("equal")
    axDict["A"].set_xlabel("x [m]")
    axDict["A"].set_ylabel("$v_x$ [m]")
    axDict["A"].set_title(f"Time step {iExperimentStep}")
    #Ground truth
    axDict["A"].text(.05,.95,f"x={onp.round(physicalState[0],1)}, \nvx={onp.round(physicalState[1],1)}",
                     horizontalalignment='left',verticalalignment='center',transform = axDict["A"].transAxes)
    axDict["A"].plot(physicalState[0],physicalState[1],"k+",markersize=20,alpha=.25)

    #Now plot the failure particles. Will use a tab color map
    failureParticleBeliefAxes = ["B","C","D","E","F","G"]
    components = ["-X1","-X2","+X1","+X2","S1","S2",]
    componentIdxs = jnp.array([[0,4],[1,5],[2,6],[3,7],[8,10],[9,11]])

    #Determine weighting of belief sizes. Min is going to be non zero so we can see it
    minParticleSize = 0
    particleSizeRange = 15
    particleSizes = particleSizeRange* failureWeights + minParticleSize
    print(failureWeights)
    #Other particle visualization data
    particleAlpha = .5
    numFailureParticles = len(currentFailureParticles)
    #In this case, we don't have bias and are using old fault model will set biases all to zero, and convert models
    if len(currentFailureParticles[0]) == 6:
        zeroFailures = jnp.zeros((numFailureParticles,12))
        #Actuators
        currentFailureParticles = zeroFailures.at[:,0:4].set(1-currentFailureParticles[:,0:4])
        #Sensors
        currentFailureParticles = zeroFailures.at[:,8:10].set(1-currentFailureParticles[:,4:6])
        zeroFailure = jnp.zeros(12)
        failureState = zeroFailure.at[0:6].set(1- failureState)

    #Get 5 closest particles
    faultSpaceDistances = jnp.linalg.norm(currentFailureParticles - failureState,axis=1)
    averageDistance = jnp.mean(faultSpaceDistances)
    #This returns the array partitioned into 3 (unsorted) idxes of the smallest values, and the remaining (unsorted) idxes
    partitionedDistanceIdxes = jnp.argpartition(faultSpaceDistances,4)
    #So sort
    idxesSortedPartition = jnp.argsort(faultSpaceDistances[partitionedDistanceIdxes[:5]])
    closestIdxes = partitionedDistanceIdxes[idxesSortedPartition]

    for iComponent in range(6):
        ax = axDict[failureParticleBeliefAxes[iComponent]]
        ax.set_xlabel("Degradation")
        ax.set_ylabel("Bias")
        ax.set_title(components[iComponent])
        setColorCyclerDistinct(numFailureParticles,ax)

        #Plot five closest particles in faults space, biggest is closest
        maxMarkerRadius = 20
        for idx in closestIdxes:
            closeFaultParticle = currentFailureParticles[idx]
            ax.scatter(closeFaultParticle[componentIdxs[iComponent,0]],closeFaultParticle[componentIdxs[iComponent,1]],
                    s=(maxMarkerRadius * (1-faultSpaceDistances[idx]/averageDistance))**2,alpha=particleAlpha/2,marker="x",c="k")



        #Plot the failure particles for this component
        for jFailureParticle,failureParticle in enumerate(currentFailureParticles):

            #print(currentFailureParticles[jFailureParticle])
            #print(componentIdxs[iComponent,1], currentFailureParticles[jFailureParticle,componentIdxs[iComponent,1]])
            #print(currentFailureParticles[jFailureParticle,10])
            ax.scatter(failureParticle[componentIdxs[iComponent,0]],
                       failureParticle[componentIdxs[iComponent,1]],s=jnp.square(particleSizes[jFailureParticle]),alpha=particleAlpha)
            ax.set_xlim(xmin=-0.05,xmax=1.05)
            ax.set_ylim(ymin=-0.05,ymax=1.05)

        #Plot true failure
        ax.scatter(failureState[componentIdxs[iComponent,0]],failureState[componentIdxs[iComponent,1]],
                    s=20**2,alpha=particleAlpha/2,marker="+",c="k")

    #For examining plots without making movie
    #plt.show()

    return fig


def evalBeliefGaussian(beliefTuple, physicalState, numMeshForPositionGaussian):
    """
    Creates a multi modal Gaussian using each element of the belief tuple.
    evalRegion is the sub set of the plot region that we evaluate the gaussian over, as not all of this region is necessarily interesting

    Parameters
    ----------
    beliefTuple : tuple
        The belief is represented by an array of failureWeights and the filter array corresponding to each possibleFailure. Contains:
            failureWeights : array, shape(maxPossibleFailures)
                Normalized weighting of the relative likelihood of each failure
            filters : array
                Array of conditional filters on the physical state.
    positionGaussianExtent : float
        Interesting region of the Gaussian, set to make computing more efficient. Currently not very well motivated
    plotRegion : array, shape(2,2)
        Bounds of the axis
    physicalState : array, shape(numState)
        Current physical state
    numMeshForPositionGaussian : int
        How fine the Gaussian mesh is.
    rotationFlag : boolean (default=False)
        Whether or not this is a 3DOF spacecraft
    """

    #We want to evaluate the gaussian near the peak of the belief, which we don't know directly
    #We do however, expect the peak to be near the actual physical state of the s/c, so we can use this as our center.
    physicalStateX = physicalState[0]
    physicalStateVX = physicalState[1]

    # eval in "x" coordinates
    beliefExtent=7
    evalX = jnp.linspace(-beliefExtent, beliefExtent, numMeshForPositionGaussian)
    evalVX = jnp.linspace(-beliefExtent, beliefExtent, numMeshForPositionGaussian)
    # Move eval region to area we expect the Gaussian to be significant over
    evalX += physicalStateX
    evalVX += physicalStateVX
    evalMeshX, evalMeshY = jnp.meshgrid(evalX, evalVX)
    evalPos = jnp.empty(evalMeshX.shape + (2,))
    evalPos = evalPos.at[:, :, 0].set(evalMeshX)
    evalPos = evalPos.at[:, :, 1].set(evalMeshY)

    beliefPdf = 0 * evalMeshX

    #No need to mask here, as we have a 2-D posterior

    for (weight, positionFilter) in zip(beliefTuple[0], beliefTuple[1]):
        physicalStateBeliefMean = positionFilter[0]
        physicalStateBeliefCovariance = positionFilter[1:1+len(physicalStateBeliefMean)]
        mean = physicalStateBeliefMean
        sigma = 1e-3 * jnp.eye(2) + physicalStateBeliefCovariance
        beliefPdf += weight * evalMultivariateGaussian(evalPos, mean, sigma)

    return evalX, evalVX, beliefPdf

#If running directly
if __name__ == "__main__":
    #Lab PC
    SAVED_DATA_DIR_PATH = None #SET TO THE DIRECTORY YOU WISH TO VISUALIZE
    visualizationOutputPath = None #SET WHERE YOU WANT THE OUTPUT DATA TO GO

    main(SAVED_DATA_DIR_PATH,experimentName="linearTest",solverName="PreSpecified",nSimulationsPerTree=200,outputPath=visualizationOutputPath,outputSubDirectory="Test",nTrial=2)
