"""
Function that processes data and gives a success code of 1 to any trial where the most likely final failure scenario is correct (matches the true underlying failure)

Intended for experiments where terminating early is inappropriate/not checked for (like safety), but where the success code was set to nan for not terminating by mistake

WARNING! Deletes data (as we will no longer be able to distinguish if the experiment terminated early or not, although this could be recovered if we know what the threshold was)
"""

import os

from failurePy.utility.reprocessData import reprocessDataAverages, confirmDataDeletion, reprocessTrialDataDicts
from failurePy.utility.saving import updateSuccessStatusAndSafetyOfTrialDataDict, updateSuccessStatusOfTrialDataDict

def updateSuccessStatus(absoluteSavedDataPath):
    """
    Recomputes success based on the last belief in the specified directory

    Parameters
    ----------
    absoluteSavedDataPath : string
        Absolute path to directory where we will recompute success based on the last belief
    """

    #Guard on user confirmation
    if confirmDataDeletion(absoluteSavedDataPath, "early termination success"):
        reprocessTrialDataDicts(absoluteSavedDataPath, updateSuccessStatusOfTrialDataDict)

        #Recompute the averages using the saved config file!
        reprocessDataAverages(os.path.join(absoluteSavedDataPath, "config.yaml"),os.path.dirname(os.path.dirname(absoluteSavedDataPath)))
    else:
        print("The success statuses will not be updated")


def updateSuccessStatusAndSafetyThroughout(absoluteSavedDataPath,force=False):
    """
    Recomputes success based on the last belief in the specified directory, and also logs the first time the system becomes unsafe

    Parameters
    ----------
    absoluteSavedDataPath : string
        Absolute path to directory where we will recompute success based on the last belief
    """
    #Guard on user confirmation
    if force or confirmDataDeletion(absoluteSavedDataPath, "early termination success"):
        reprocessTrialDataDicts(absoluteSavedDataPath, updateSuccessStatusAndSafetyOfTrialDataDict)

        #Recompute the averages using the saved config file!
        reprocessDataAverages(os.path.join(absoluteSavedDataPath, "config.yaml"),os.path.dirname(os.path.dirname(absoluteSavedDataPath)))
    else:
        print("The success statuses will not be updated")

if __name__ == "__main__":

    SAVED_DATA_DIRECTORY = None #SET TO THE DIRECTORY YOU WISH TO COMPUTE THE SUCCESS AT THE END OF


    #updateSuccessStatus(SAVED_DATA_DIRECTORY)
    updateSuccessStatusAndSafetyThroughout(SAVED_DATA_DIRECTORY, force=True)
