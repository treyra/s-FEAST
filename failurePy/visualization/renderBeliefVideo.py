"""
File the makes a visualization of the evolution of the belief over time, used for visualizing the hardware experiments
"""

import os
import warnings
import matplotlib.pyplot as plt
import numpy as np
import cv2

from failurePy.visualization.visualizationUtilityFunctions import loadBeliefData

#Ignore matplotlib warnings'
warnings.filterwarnings( "ignore", module = r"matplotlib\..*" )

#Going to just make the plots manually
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
    beliefList,possibleFailures,failureState, dummyPhysicalStates = loadBeliefData(savedDataDirPath,experimentName,solverName,nSimulationsPerTree,nTrial)
    beliefWeightsList = []
    for belief in beliefList:
        beliefWeightsList.append(belief[0])

    #Testing movie making
    makeTimeSeries(beliefWeightsList,possibleFailures,failureState,dt=1,fps=10,outputPath=outputPath,outputSubDirectory=outputSubDirectory)

#dt is universal, so making exception
def makeTimeSeries(beliefWeightsList,possibleFailures,failureState,dt=1,fps=10,outputPath="~/Documents/BeliefVisualization",outputSubDirectory="TimeSeries"): # pylint: disable=invalid-name
    """
    Makes time series of provided data, with transitions added in for smoothness.
    Save video to the provided directory. Also uses this directory for temporary storage of the generated frames.

    Parameters
    ----------
    beliefWeightsList : list
        List of the failure probabilities for each time step's belief in beliefList
    possibleFailures : array, shape(nMaxPossibleFailures,numAct+numSen)
        List of possible failures to be considered by the solver. If not exhaustive, these are a random subset. Belief failure weights are are over these possibilities
    failureState : array, shape(numAct+numSen)
        True failure state
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

    #Loop through and make frames, we'll average as we transition
    for iData in range(len(beliefWeightsList)): #Need to index directly for transitioning pylint: disable=consider-using-enumerate
        #Plot each frame
        for jFrame in range(dt*fps):
            #Transition smoothly if not the first frame
            if iData != 0 and jFrame/fps <= 1/4:
                beliefWeights = (.5 - 2*jFrame/fps) * beliefWeightsList[iData-1] + (2*jFrame/fps +.5) * beliefWeightsList[iData]
            elif iData != len(beliefWeightsList)-1 and jFrame/fps >= 3/4:
                beliefWeights = (1 - 2*(jFrame/fps - 3/4)) * beliefWeightsList[iData] + (2*(jFrame/fps - 3/4)) * beliefWeightsList[iData+1]
            else:
                beliefWeights = beliefWeightsList[iData]

            fig = plotElementFailureWeights(beliefWeights,possibleFailures,failureState)

            #Save the figure as a png
            figName = str(iData*dt*fps+jFrame) + ".png"

            figPath = os.path.join(outputSubDirectory,figName)

            fig.savefig(figPath)
            plt.close()



    videoName = os.path.join(outputSubDirectory,'video.mp4')

    images = [img for img in os.listdir(outputSubDirectory) if img.endswith(".png")]
    frame = cv2.imread(os.path.join(outputSubDirectory, images[0])) # pylint: disable=no-member
    height, width, dummyLayers = frame.shape

    video = cv2.VideoWriter(videoName, cv2.VideoWriter_fourcc(*'mp4v'), 10, (width,height)) # pylint: disable=no-member

    for image in images:
        frame = cv2.imread(os.path.join(outputSubDirectory, image)) # pylint: disable=no-member
        video.write(frame)

        cv2.imshow('Frame',frame) # pylint: disable=no-member

    cv2.destroyAllWindows() # pylint: disable=no-member
    video.release()


#Plot element wise failure weights (single frame). DON'T show, let calling function handle that, but return fig to save/close
def plotElementFailureWeights(beliefWeights,possibleFailures,failureState):
    """
    Plots the likelihood of each possible failure scenario at the given time step

    Parameters
    ----------
    beliefWeights : array, shape(nMaxPossibleFailures)
        Likelihood of each possible failure scenario
    possibleFailures : array, shape(nMaxPossibleFailures,numAct+numSen)
        List of possible failures to be considered by the solver. If not exhaustive, these are a random subset. Belief failure weights are are over these possibilities
    failureState : array, shape(numAct+numSen)
        True failure state

    Returns
    -------
    fig : matplotlib figure
        Figure object with the failure likelihoods plotted.
    """
    fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(7.5,5)) #ax is used a lot with matplotlib so pylint: disable=invalid-name
    #fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(11.25,6))

    #Need to calculate the weight on each actuator
    componentFailureWeights = np.zeros(len(failureState))

    for iPossibleFailure,possibleFailure in enumerate(possibleFailures):
        beliefWeight = beliefWeights[iPossibleFailure]
        componentFailureWeights += beliefWeight*possibleFailure

    #Flip to be failure probabilities
    componentFailureWeights = 1 - componentFailureWeights

    #Hard code the labels
    barXLabels = ["-X+Mθ","-X-Mθ","+X+Mθ","+X-Mθ","-Y+Mθ","-Y-Mθ","+Y+Mθ","+Y-Mθ","S1 X","S2 X","S3 Y","S4 Y","S5 θ","S6 θ"]

    xPositions = np.arange(len(barXLabels))  # the label locations

    #Now plot the bar chart
    barPlot = ax.bar(xPositions,componentFailureWeights) #Unused if not annotating, pylint: disable=unused-variable
    ax.set_ylabel('Failure Probability')
    ax.set_title('Failure weight by component')
    ax.set_xticks(xPositions)
    ax.set_xticklabels(barXLabels,rotation = 45)
    #ax.legend()

    ax.set_ylim([0, 1.1])

    #Optional labeling of each bar with the current hight, can make the plot busy.
    #autoLabel(barPlot,ax)

    return fig


def autoLabel(barPlotRectangles,ax): #ax is used a lot with matplotlib so pylint: disable=invalid-name
    """
    Attach a text label above each bar in rectangles, displaying its height.
    Used for annotating bar graphs.

    Parameters
    ----------
    rectangles : matplotlib BarContainer
        Bar plot rectangles to annotate
    ax : matplotlib axis
        Bar plot axis to annotate on.
    """
    for rect in barPlotRectangles:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

#If running directly
if __name__ == "__main__":
    SAVED_DATA_DIR_PATH = None # SET TO PATH OF EXPERIMENT TO RENDER

    main(SAVED_DATA_DIR_PATH,experimentName="12 movie 5",solverName="realTimeSFEAST",nSimulationsPerTree=100,outputPath="SET_AS_ABS_PATH_TO_DESIRED_OUTPUT_DIRECTORY",outputSubDirectory="TimeSeries")
