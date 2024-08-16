"""
Simple combinatorial implementation that should be integer exact
"""

import math

#Math expression so allowing single letter names
def comb(n,k): # pylint: disable=disallowed-name,invalid-name
    """
    N choose k

    Parameters
    ----------
    N : int
        Number of items to choose from
    k : int
        Number of items to choose

    Returns
    -------
    nChooseK : int
        Number of combinations of k items from N totalW
    """

    return math.factorial(n)/(math.factorial(k)*math.factorial(n-k))
