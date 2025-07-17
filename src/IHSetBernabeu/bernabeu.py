import numpy as np
import numpy as np
from scipy.interpolate import interp1d

def Bernabeu(A, B, C, D, M, h, xo):
    """
    Bernabeu beach profile composed of two expressions depending on depth h.
    The profile is split at the intersection of both expressions.
    """

    # Ensure h is a numpy array and sorted
    h = np.array(h)
    if np.any(np.diff(h) < 0):
        h = np.sort(h)

    # Evaluate both segments
    x1 = (h / A)**(3/2) + B / (A**(3/2)) * h**3 

    h_M = h-M
    h_M = h_M[h_M >= 0]  # Ensure h-M is non-negative for the second segment

    x2 = ((h_M) / C)**(3/2) + D / C**(3/2) * (h_M)**3 + xo

    # we find the intersection considering they have diferent sizes
    x2_interp = np.interp(h-M, h_M, x2)
    intersection_index = np.argwhere(np.isclose(x1, x2_interp, atol=1e-5)).flatten()
    if intersection_index.size == 0:
        x = x1
    else:

        intersection_index = intersection_index[0]
        x1_ = x1[:intersection_index]
        x2_ = x2_interp[intersection_index:]

        x = np.concatenate((x1_, x2_))
    
    return x, x1, x2, h_M
