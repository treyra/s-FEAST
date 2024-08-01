"""
File with useful functions for visualizing and understanding bit manipulation in python, especially since it only mimics C integers
"""

import numpy as onp

def displayTwosComp(intValue, nBits):
    """
    Displays the value in twos complement notation for an integer (which python mimics when doing bitwise operations), with the specified number of bits
    """
    print(getTwosCompString(intValue, nBits))

#Adapted from https://stackoverflow.com/questions/1604464/twos-complement-in-python to display easier
def getTwosCompString(intValue, nBits):
    """
    Compute the 2's complement of intValue. Recall the nBits INCLUDES the sign bit, so we will fake it in display

    Checks bit limits, which is a bit of a slow down but displays -1 right now.
    """
    if (intValue > 2**(nBits-1) -1) or (intValue < - (2**(nBits-1)) ):
        print(f"Int {intValue} outside number of range of values for a signed int given the specified nBits {- (2**(nBits-1)) } - {2**(nBits-1) -1}, behavior is undefined")

    #Doesn't work with jax... just going to avoid binary ops here, this isn't speed critical
    ##Check for issues with bit limits
    #atBitLimitFlag = False
    #print(type(intValue), nBits)
    #if nBits >=8:
    #    if isinstance(intValue, (onp.int8,jnp.int8)):
    #        if nBits > 8:
    #            print(f"nBits is greater than the size of the integer (8 bit), errors will occur with negative numbers")
    #        else:
    #            atBitLimitFlag = True
    #elif nBits >=16:
    #    print("Should be here")
    #    if isinstance(intValue, (onp.int16,jnp.int16)):
    #        if nBits > 16:
    #            print(f"nBits is greater than the size of the integer (16 bit), errors will occur with negative numbers")
    #        else:
    #            atBitLimitFlag = True
    #elif nBits >=32:
    #    if isinstance(intValue, (onp.int32,jnp.int32)):
    #        if nBits > 32:
    #            print(f"nBits is greater than the size of the integer (32 bit), errors will occur with negative numbers")
    #        else:
    #            atBitLimitFlag = True
    #elif nBits >=64:
    #    if isinstance(intValue, (onp.int64,jnp.int64)):
    #        if nBits > 64:
    #            print(f"nBits is greater than the size of the integer (64 bit), errors will occur with negative numbers")
    #        else:
    #            atBitLimitFlag = True
    digits = nBits+2
    if intValue < 0:
        #MSB is the sign bit, and need to watch out for overflow, so just add half at a time
        #We will fake the sign bit
        intValue = 2**(nBits-2) + intValue
        intValue = 2**(nBits-2) + intValue
        #This also doesn't work because jax (and maybe numpy too) casts down to the smaller int size, and 2**nBits is outside of that!
        #intValue = 2**nBits + intValue
        #This doesn't work if at limit of integer, so will just non binary op version
        #intValue = (1 << nBits) + intValue
        returnString = format(intValue,f'#0{digits}b')
        #Fake the sign bit
        return returnString[:2] +'1' + returnString[3:]

    if (intValue & (1 << (nBits - 1))) != 0:
        # If sign bit is set.
        # compute negative value.
        intValue = intValue - (1 << nBits)

    return format(intValue,f'#0{digits}b')

def displayVectorTwosComp(intArray, nBits):
    """
    Displays an integer array in twos complement notation
    """
    outputList = []
    for intValue in intArray:
        outputList.append(getTwosCompString(intValue, nBits))
    print(onp.array(outputList))
