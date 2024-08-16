"""
File that has functions for comparing data between test results

"""

import os
import subprocess
from tqdm import tqdm

def checkIfResultsAreConsistent(absoluteExperimentSavedDataDirectory1,absoluteExperimentSavedDataDirectory2):
    """
    Method that checks if the data in two saved experiment directories math.
    Useful for checking if two versions of the simulation pipeline are interchangeable
    Note checks at the experiment level directly
    """

    #Check if trial nums match
    trials1 = next(os.walk(absoluteExperimentSavedDataDirectory1))[1]
    trials2 = next(os.walk(absoluteExperimentSavedDataDirectory2))[1]
    consistent = True

    #Only check one way
    print(f"Checking consistency of the trials found in {absoluteExperimentSavedDataDirectory1} with those found in {absoluteExperimentSavedDataDirectory2}")
    if len(trials1) != len(trials2):
        if len(trials1) > len(trials2):
            print(f"{absoluteExperimentSavedDataDirectory1} has more trials than {absoluteExperimentSavedDataDirectory2}")
            consistent = False
        else:
            print(f"{absoluteExperimentSavedDataDirectory1} has less trials than {absoluteExperimentSavedDataDirectory2}")
            consistent = False
    for trial in tqdm(trials1):
        #Check if the same trial is in trials2
        if not trial in trials2:
            print(f"Trial {trial} in {absoluteExperimentSavedDataDirectory1} not found in {absoluteExperimentSavedDataDirectory2}")
            consistent = False
            continue
        try:
            dataDict1StringPath = os.path.join(absoluteExperimentSavedDataDirectory1,trial,"trialData.txt")
            with open(dataDict1StringPath, "rb") as trialDataFile:
                data1DictString = trialDataFile.read()
            #Check if data is in 2, then check data
            try:
                data2DictStringPath = os.path.join(absoluteExperimentSavedDataDirectory2,trial,"trialData.txt")
                with open(data2DictStringPath, "rb") as trialDataFile:
                    data2DictString = trialDataFile.read()
                #Iterate through all the stuff in the dict
                if data1DictString[:-20] != data2DictString[:-20]: #Hack to ignore wall clock time data
                    print(f"Trial {trial}'s data in {absoluteExperimentSavedDataDirectory1} and {absoluteExperimentSavedDataDirectory2} is inconsistent")
                    #print(git.diff("--no-index",data1DictString,data2DictString))
                    #gitOutput = subprocess.Popen([f"git diff --no-index {dataDict1StringPath} {data2DictStringPath}"], cwd=os.getcwd(), stdout=subprocess.PIPE)
                    #NOTE shell=True is unsafe in general (if dataDict1StringPath or dataDict3StringPath are provided by arbitrary, malicious inputs) but safe here since we control input
                    with subprocess.Popen([f"git diff --no-index {dataDict1StringPath} {data2DictStringPath}"], shell=True, stdout=subprocess.PIPE) as gitOutput:
                        while gitOutput.poll() is None: #Loop until terminated
                            print(gitOutput.stdout.readline())
                    consistent = False

            except FileNotFoundError:
                print(f"Trial {trial} in {absoluteExperimentSavedDataDirectory1} has data, but no data exists in {absoluteExperimentSavedDataDirectory2}")
                consistent = False
        except FileNotFoundError:
            try:
                #Just checking if we can read the second file, not going to use it
                data2DictStringPath = os.path.join(absoluteExperimentSavedDataDirectory2,trial,"trialData.txt")
                with open(data2DictStringPath, "rb") as trialDataFile:
                    dummyData2DictString = trialDataFile.read()
                print(f"Trial {trial} in {absoluteExperimentSavedDataDirectory1} does not have any data, but data does exist in {absoluteExperimentSavedDataDirectory2}")
                consistent = False
            except FileNotFoundError:
                pass #consistent at least if they both don't have it

    if consistent:
        print("All data is consistent!")


if __name__ == "__main__":
    EXPERIMENT_SAVED_DATA_PATH_1 = "/home/ragan/CaltechVersionFailurePy/probabilistic_planning_v2/failurePy/SavedData/linearSafetyMultiTest/SFEAST/200"
    EXPERIMENT_SAVED_DATA_PATH_2 = "/home/ragan/CaltechVersionFailurePy/probabilistic_planning_v2/failurePy/SavedData/linearSafetyTest3/SFEAST/200"
    checkIfResultsAreConsistent(EXPERIMENT_SAVED_DATA_PATH_1,EXPERIMENT_SAVED_DATA_PATH_2)
