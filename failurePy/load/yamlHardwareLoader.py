"""
Methods for loading the experiment parameters from a yaml file that are specific to hardware emulation
Future TODO: could add more robust input checking/sanitizing, currently assuming user will do this correctly, but typos in the yamls could lead to weird behavior.
"""

from failurePy.load.yamlLoaderUtilityMethods import getInputDict, loadOptionalParameter


def loadRealTimeParams(configFilePath):
    """
    Loads hardware emulation experiment parameters from a .yaml file

    Parameters
    ----------
    configFilePath : String
        Absolute path to the config file for the experiment

    Returns
    -------
    initialAction : array, len(numAct)
        Initial action to take (zeros if none provided)
    """
    inputDict = getInputDict(configFilePath)

    initialAction = loadOptionalParameter("initialAction",inputDict,None)

    return initialAction
