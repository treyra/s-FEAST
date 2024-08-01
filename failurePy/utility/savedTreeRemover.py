"""
Script that removes the tree from saved data dictionaries for data compression. Useful when accidentally starting a run with saveTreeFlag=True

WARNING! PERMANENTLY DELETES DATA
"""

from failurePy.utility.reprocessData import confirmDataDeletion, reprocessTrialDataDicts

def removeSavedTree(absoluteSavedDataPath):
    """
    Removes saved tree data from the specified directory

    Parameters
    ----------
    absoluteSavedDataPath : string
        Absolute path to directory where the tree data will be deleted from.
    """

    #Guard on user confirmation
    if confirmDataDeletion(absoluteSavedDataPath,"saved tree data"):
        reprocessTrialDataDicts(absoluteSavedDataPath,removeSavedTreeFromTrialDataDict)

    else:
        print("Saved Tree Data will not be removed")

def removeSavedTreeFromTrialDataDict(trialResultsDict):
    """
    Removes saved tree data from the specified dictionary

    Parameters
    ----------
   trialResultsDict : dict
        trial data we will remove the saved tree from
    """

    #Look if it has a tree (no need to overwrite if it doesn't have it)
    if "treeList" in trialResultsDict:
        trialResultsDict.pop("treeList")
    return trialResultsDict

if __name__ == "__main__":
    SAVED_DATA_DIRECTORY = None #SET TO THE DIRECTORY YOU WISH TO DELETE TREE DATA FROM
    removeSavedTree(SAVED_DATA_DIRECTORY)
