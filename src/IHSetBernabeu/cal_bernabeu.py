import numpy as np
import pandas as pd
from .bernabeu import Bernabeu
from IHSetUtils import wMOORE
from scipy.optimize import least_squares

class cal_Bernabeu(object):
    """
    cal_Bernabeu
    
    Configuration to calibrate and run the Bernabeu profile.
    
    This class reads input datasets, calculates its parameters.
    """
    def __init__(self, CM, Hs50, D50, Tp50, doc, hr, HTL=0):
        self.CM = CM
        self.Hs50 = Hs50
        self.D50 = D50
        self.Tp50 = Tp50
        self.doc = doc
        self.HTL = HTL
        self.hr = hr

        # observed data placeholders
        self.x_raw = None
        self.y_raw = None
        self.x_obs = None
        self.y_obs = None
        self.y_obs_rel = None
        self.x_drift = 0
        self.data = False

        # model segments placeholders
        self.x_full = None
        self.y_full = None
        self.x1 = None
        self.y1 = None
        self.x2 = None
        self.y2 = None

        # Calculate parameters and depth vector
        self.params()
        self.def_hvec()

    def params(self):
        ws = wMOORE(self.D50 / 1000)
        gamma = self.Hs50 / (ws * self.Tp50)
        self.Ar = 0.21 - 0.02 * gamma
        self.B = 0.89 * np.exp(-1.24 * gamma)
        self.C = 0.06 + 0.04 * gamma
        self.D = 0.22 * np.exp(-0.83 * gamma)

    def def_hvec(self):
        self.h = np.arange(0.0, self.CM + self.doc, 0.001)

    def def_xo(self):
        self.xo = (
            (self.hr + self.CM) / self.Ar
        )**1.5 - (self.hr / self.C)**1.5 + (
            self.B / (self.Ar**1.5)
        ) * (self.hr + self.CM)**3 - (
            self.D / (self.C**1.5)
        ) * self.hr**3

    def run(self):
        """
        Run the Bernabeu profile with current parameters.
        Calculates and stores full profile and segment arrays,
        applies x_drift and HTL shifts, returns full x,y.
        """
        self.def_xo()
        # raw computation
        x_raw, x1_raw, x2_raw, h2 = Bernabeu(
            self.Ar, self.B, self.C, self.D,
            self.CM, self.h, self.xo
        )
        y2_raw = self.h + self.HTL

        self.x_full = x_raw
        self.y_full = self.h + self.HTL

        # Surf e Shoalling segments mask
        mask_surf  = self.h <= (self.hr + self.CM)
        mask_shoal = ~mask_surf

        # segment 1: surf zone (h <= hr + CM)
        self.x1_full = x1_raw
        self.y1_full = self.h + self.HTL
        self.x1 = self.x_full[mask_surf]
        self.y1 = self.y_full[mask_surf]
        
        # segment 2: shoaling zone
        #x2_interp    = np.interp(self.h, h2, x2_raw)
        self.x2_full = x2_raw
        self.y2_full = h2 + self.CM + self.HTL
        self.x2 = self.x_full[mask_shoal]
        self.y2 = self.y_full[mask_shoal]

        return self.x_full, self.y_full

    def from_D50(self, D50):
        self.D50 = D50
        self.params()
        return self.run()

    def from_Hs50(self, Hs50):
        self.Hs50 = Hs50
        self.params()
        return self.run()

    def from_Tp50(self, Tp50):
        self.Tp50 = Tp50
        self.params()
        return self.run()

    def change_hr(self, hr):
        self.hr = hr
        self.params()
        return self.run()

    def change_CM(self, CM):
        self.CM = CM
        self.params()
        self.def_hvec()
        return self.run()

    def change_doc(self, doc):
        self.doc = doc
        self.def_hvec()
        return self.run()

    def change_HTL(self, HTL):
        self.HTL = HTL
        self.def_hvec()
        return self.run()

    def add_data(self, path):
        df = pd.read_csv(path)
        self.x_raw = df["X"].values
        self.y_raw = df["Y"].values

        # If topography above water is negative and depths positive, no sign flip needed
        # Compute drift at HTL (find x where elevation equals HTL)
        self.x_drift = np.interp(self.HTL, self.y_raw, self.x_raw)

        # Filter observed between HTL (upper limit) and DoC (lower limit)
        mask = (self.y_raw >= self.HTL) & (self.y_raw <= self.doc)
        if not np.any(mask):
            raise ValueError(
                f"CSV does not contain values between HTL={self.HTL} and DoC={self.doc}."
            )
        x_cut = self.x_raw[mask]
        y_cut = self.y_raw[mask]

        # Align observed so that HTL intersection is at x=0
        self.x_obs = x_cut - self.x_drift
        self.y_obs = y_cut
        # Relative elevation from HTL
        self.y_obs_rel = self.y_obs - self.HTL
        self.data = True

        return self.calibrate()

    def calibrate(self):
        if not self.data:
            raise ValueError("No data loaded. Use add_data() to load data.")

        x_obs_interp = np.interp(self.h, self.y_obs_rel, self.x_obs)

        def resid(log_params):
            A, B, C, D = np.exp(log_params[:4])
            hr = log_params[4]
            xo = (
                (hr + self.CM) / A
            )**1.5 - (hr / C)**1.5 + (
                B / (A**1.5)
            ) * (hr + self.CM)**3 - (
                D / (C**1.5)
            ) * hr**3

            mask1 = self.h <= (hr + self.CM)
            h1 = self.h[mask1]
            x1 = (h1 / A)**1.5 + B / (A**1.5) * h1**3

            mask2 = ~mask1
            h2 = self.h[mask2] - self.CM
            x2 = (h2 / C)**1.5 + D / (C**1.5) * h2**3 + xo

            res1 = x1 - x_obs_interp[mask1]
            res2 = x2 - x_obs_interp[mask2]
            return np.concatenate([res1, res2])

        x0 = np.array([
            np.log(self.Ar), np.log(self.B),
            np.log(self.C), np.log(self.D),
            self.hr
        ])
        lb = [np.log(1e-3)]*4 + [0.0]
        ub = [np.log(10.0)]*4 + [self.h.max()]

        res = least_squares(
            resid, x0, bounds=(lb, ub),
            loss='huber', f_scale=0.1,
            xtol=1e-8, ftol=1e-8, max_nfev=2000
        )

        self.Ar, self.B, self.C, self.D = np.exp(res.x[:4])
        self.hr = res.x[4]
        self.def_hvec()
        return self.run()
