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

    # (1) Calculate the surf zone segment
    x1 = (h / A)**1.5 + B/(A**1.5) * h**3
 
    # (2) Calculate the shoaling zone segment (only where: h >= M)
    h_M_full = h - M
    mask2 = h_M_full >= 0
    h2 = h_M_full[mask2]
    x2 = (h2 / C)**1.5 + D/(C**1.5) * h2**3 + xo

    # 3) Interpolate x2 back over the entire vector h
    # (for points < M it will use x2_interp <– x2 on the edge, but it will not be used)
    x2_interp = np.interp(h, h[mask2], x2)

    # 4) Finds the breakpoint by minimizing |x1–x2_interp|
    diff = np.abs(x1 - x2_interp)
    i_break = np.argmin(diff)

    #5) Assemble the complete profile
    x_full = np.empty_like(x1)
    x_full[:i_break] = x1[:i_break]
    x_full[i_break:] = x2_interp[i_break:]

    return x_full, x1, x2, h2


