import math
import numpy as np # Often useful for numerical operations

# Corresponds to zi.R
def zi(n, thetai, typeDistance):
    psiTotal = 1.0

    if typeDistance == 'K':
        # This loop now correctly implements the R formula:
        # psiTotal = psiTotal * (1 - (exp((-1)*(n-i+1)*(thetai[i]))))/(1 - exp((-1)*thetai[i]))
        for i in range(1, n): # R's i from 1 to n-1
            # R's thetai[i] is Python's thetai[i-1] (0-indexed)
            # R's (n-i+1) is used directly as Python's loop variable 'i' matches R's 'i'.
            numerator = 1 - math.exp(-1 * (n - i + 1) * thetai[i-1])
            denominator = 1 - math.exp(-1 * thetai[i-1])
            # Avoid division by zero if thetai is very small
            if denominator == 0:
                # This case corresponds to the limit as thetai -> 0, which is (n-i+1)
                psiTotal = psiTotal * (n - i + 1)
            else:
                psiTotal = psiTotal * (numerator / denominator)

    elif typeDistance == 'C':
        for i in range(1, n): # R's i from 1 to n-1
             # R: (n-i)*(exp((-1)*(thetai[i])))
             # Python: (n-i)*(exp((-1)*(thetai[i-1])))
            psiTotal = psiTotal * (1 + (n-i) * (math.exp((-1)*thetai[i-1])))

    return psiTotal

# Corresponds to Zvalue.R
def Zvalue(n, m, theta, typeDistance):
    z = np.zeros(m) # Initialize numpy array of zeros

    for i in range(m): # Loop from 0 to m-1 for 0-indexed array
        # theta[i,:] slices the i-th row (0-indexed)
        a = zi(n, theta[i,:], typeDistance)
        z[i] = a

    return z