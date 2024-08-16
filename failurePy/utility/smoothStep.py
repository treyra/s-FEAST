"""
Module to create a smooth step function.

May be extended to have several useful functions in the future/jax implementations
"""

from scipy.special import comb
import numpy as onp


#Credit: Jonas Adler on stackoverflow for the implementation
def smoothStep(xVal, xMin=0, xMax=1, maxDerivative=1):
    """
    Smoothly interpolates between xMin and xMax, with the first
    maxDerivatives being continuous
    """
    xVal = onp.clip((xVal - xMin) / (xMax - xMin), 0, 1)

    result = 0
    for nDerivative in range(0, maxDerivative + 1):
        result += comb(maxDerivative + nDerivative, nDerivative) * comb(2 * maxDerivative + 1, maxDerivative - nDerivative) * (-xVal) ** nDerivative

    result *= xVal ** (maxDerivative + 1)

    return result
