"""
Module with methods for loading files from the package submodules
"""

import sys
import importlib.resources as packageResources

#import yaml

#Guard against importing this with python < 3.9.
# Only this module has issues, so we are doing this instead of setting a package requirement
if sys.version_info < (3,9):
    raise ValueError("Loading of package files is not supported for python < 3.9")


def getPackageSubDirectoryPath(package,subDirectory):
    """
    Helper function to load the files of a specified package to make them available to the program.
    This is in it's own function so we don't need to guard on python version in more than one spot

    Parameters
    ----------
    package : python package
        The package to load the files from
    subDirectory : string
        The subDirectory of the package we want a path too.

    Returns
    -------
    packageSubDirectoryPath : string
        Path to the package's subDirectory
    """

    return packageResources.files(package).joinpath(subDirectory) # pylint: disable=no-member
