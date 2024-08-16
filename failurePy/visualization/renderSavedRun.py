"""
File the makes a pdf and png visualization of the experiment.

Useful for when the experiment wasn't originally visualized
"""
import os
import warnings


from failurePy.visualization.renderPlanarVisWrapper import visualizeFirstTrajectory
from failurePy.load.yamlLoader import loadExperimentParamsFromYaml

#Ignore matplotlib warnings'
warnings.filterwarnings( "ignore", module = r"matplotlib\..*" )

#Going to just make the plots manually
def main(savedDataDirPath,experimentName,outputPath):
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
    #Get the exp params loaded
    experimentDataDirPath = os.path.join(savedDataDirPath,experimentName)
    configFilePath = os.path.join(experimentDataDirPath,"config.yaml")
    experimentParamsDict, dummyRelativeSaveDirectoryPath = loadExperimentParamsFromYaml(configFilePath) #We won't use extra data here pylint: disable=unbalanced-tuple-unpacking

    outputFilePath = os.path.join(outputPath,"renderBeliefAndTreeOnly.pdf")

    visualizeFirstTrajectory(experimentDataDirPath,experimentParamsDict,outputFilePath,regenTree=85)


#If running directly
if __name__ == "__main__":
    SAVED_DATA_DIR_PATH = None #SET THIS BEFORE RUNNING!
    main(SAVED_DATA_DIR_PATH,experimentName="hardwareSafetyTest",outputPath="SET_AS_ABS_PATH_TO_DESIRED_OUTPUT_DIRECTORY")
