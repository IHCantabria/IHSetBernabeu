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

    # (1) Calculate the surf zone segment
    x1 = (h / A)**(3/2) + B/(A**(3/2)) * h**3
 
    # (2) Calculate the shoaling zone segment (only where: h >= CM)
    h_M = h - M
    mask2 = h_M >= 0 # Ensure h-M is non-negative for the second segment
    h2 = h_M[mask2]
    x2 = (h2 / C)**(3/2) + D/(C**(3/2)) * h2**3 + xo

    # 3) Interpolate x2 back over the entire vector h (not only in the h_M vector)
    # (for points < M it will use x2_interp <– x2 on the edge, but it will not be used)
    x2_interp = np.interp(h, h[mask2], x2)

    # 4) Always finds the breakpoint by minimizing |x1–x2_interp|
    diff = np.abs(x1 - x2_interp)
    i_break = np.argmin(diff)
    
    #5) Assemble the complete profile where xi = hi
    x = np.empty_like(x1)
    x[:i_break] = x1[:i_break]
    x[i_break:] = x2_interp[i_break:]

    return x, x1, x2, h2
