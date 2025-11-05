import numpy as np
import pandas as pd
from .bernabeu import Bernabeu
from IHSetUtils import wMOORE
from scipy.optimize import least_squares

class cal_Bernabeu(object):
    """
    Calibrator/runner for the Bernabeu (2003) profile.
    Supports calibration in X (original) and Y (NEW) via 'fit_mode'.
    """

    # NEW: add fit_mode ("x" or "y")
    def __init__(self, HTL, Hs50, Tp50, D50, CM, hr, doc, fit_mode: str = "x"):

        # --- Input data validation (unchanged) ---
        gamma_break = CM / Hs50
        if not (0.5 <= gamma_break <= 2.0):
            raise ValueError(
                f"CM should be between 0.5 and 2.0 times Hs50 "
                f"(CM/Hs50 given = {gamma_break:.2f})"
            )
        if not (-17.0 <= HTL <= 17.0):
            raise ValueError(f"HTL must be between -17 and 17 m (given {HTL})")
        if not (0.1 <= Hs50 <= 4):
            raise ValueError(f"Hs50 must be between 0.1 and and be 2x smaller than CM (given {Hs50})")
        if not (4.0 <= Tp50 <= 20.0):
            raise ValueError(f"Tp50 must be between 1 and 25 s (given {Tp50})")
        if not (0.06 <= D50 <= 4.0):
            raise ValueError(f"D50 must be between 0.06 and 4.0 mm (given {D50})")
        if not (HTL < doc <= HTL+20.0):
            raise ValueError(f"doc must be between HTL and CM/2 (given doc={doc}, HTL={HTL} and CM/2={CM/2})")
        if not (HTL < hr < CM/2):
            raise ValueError(f"hr must satisfy HTL-0.5 < hr < CM/2 (given hr={hr}, HTL={HTL}, CM/2={CM/2})")

        self.HTL = HTL
        self.Hs50 = Hs50
        self.Tp50 = Tp50
        self.D50 = D50
        self.CM = CM
        self.hr = hr
        self.doc = doc

        # NEW: store fit mode (default "x")
        self.fit_mode = (fit_mode or "x").lower().strip()
        if self.fit_mode not in ("x", "y"):
            raise ValueError("fit_mode must be 'x' or 'y'")

        # observed data placeholders
        self.x_raw = None
        self.y_raw = None
        self.x_obs = None
        self.y_obs = None
        self.y_obs_rel = None
        self.x_drift = 0.0
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
        A_raw = 0.21 - 0.02 * gamma
        self.A = max(A_raw, 1e-3)
        self.B = 0.89 * np.exp(-1.24 * gamma)
        self.C = 0.06 + 0.04 * gamma
        self.D = 0.22 * np.exp(-0.83 * gamma)

    def def_hvec(self):
        # vertical domain from HTL down to doc (plus CM for shoaling)
        self.h = np.arange(0.0, self.CM + self.doc, 0.001)

    def def_xo(self):
        # offshore matching (Bernabeu’s junction term)
        self.xo = (
            (self.hr + self.CM) / self.A
        )**1.5 - (self.hr / self.C)**1.5 + (
            self.B / (self.A**1.5)
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
        x_raw, x1_raw, x2_raw, h2 = Bernabeu(
            self.A, self.B, self.C, self.D,
            self.CM, self.h, self.xo
        )

        # final (apply drift, convert h→y)
        self.x_full = x_raw + self.x_drift
        self.y_full = self.h + self.HTL

        # Surf e Shoaling segments
        mask_surf  = self.h <= (self.hr + self.CM)
        mask_shoal = ~mask_surf

        self.x1_full = x1_raw + self.x_drift
        self.y1_full = self.h + self.HTL
        self.x1 = self.x_full[mask_surf]
        self.y1 = self.y_full[mask_surf]

        self.x2_full = x2_raw + self.x_drift
        self.y2_full = h2 + self.CM + self.HTL
        self.x2 = self.x_full[mask_shoal]
        self.y2 = self.y_full[mask_shoal]

        return self.x_full, self.y_full

    # ------------------- "from ..." updaters (unchanged) -------------------
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

    # ------------------- NEW: drift shift with optional recalibration -------------------
    def shift_HTL(self, new_x0: float, recalibrate: bool = False):
        """
        NEW: set a new shoreline position (x_drift). If recalibrate=True,
        re-fit the parameters keeping HTL/DoC (vertical window) fixed.
        """
        self.x_drift = float(new_x0)
        if self.data and recalibrate:
            self._rebuild_obs_keep_drift()
            self.calibrate()
        return self.run()

    # alias
    def shift_x0(self, new_x0: float, recalibrate: bool = False):
        return self.shift_HTL(new_x0, recalibrate=recalibrate)

    # ------------------- data I/O -------------------
    def add_data(self, path):
        df = pd.read_csv(path, dtype={"X": float, "Y": float})
        self.x_raw = pd.to_numeric(df["X"], errors="coerce").to_numpy()
        self.y_raw = pd.to_numeric(df["Y"], errors="coerce").to_numpy()

        # If terrain is stored with depths positive downward, flip to positive-up (optional)
        # Keep your current convention; only ensure arrays are finite.

        # NEW: robust shoreline (x_drift) at y==HTL — sort by Y before interpolating
        m_valid = np.isfinite(self.x_raw) & np.isfinite(self.y_raw)
        xv = self.x_raw[m_valid]
        yv = self.y_raw[m_valid]
        idx = np.argsort(yv)
        self.x_drift = float(np.interp(self.HTL, yv[idx], xv[idx]))

        # Filter observed between HTL and DoC
        mask = (self.y_raw >= self.HTL) & (self.y_raw <= self.doc) & m_valid
        if not np.any(mask):
            raise ValueError(
                f"CSV does not contain values between HTL={self.HTL} and DoC={self.doc}."
            )
        x_cut = self.x_raw[mask]
        y_cut = self.y_raw[mask]

        # Relative arrays for calibration
        self.x_obs = x_cut - self.x_drift            # observed X relative to shoreline
        self.y_obs = y_cut
        self.y_obs_rel = self.y_obs - self.HTL       # h = Y - HTL
        self.data = True

        return self.calibrate()

    # NEW: rebuild obs after changing drift, without recomputing x_drift from CSV
    def _rebuild_obs_keep_drift(self):
        if self.x_raw is None or self.y_raw is None:
            raise ValueError("No CSV loaded.")
        m_valid = np.isfinite(self.x_raw) & np.isfinite(self.y_raw)
        mask = (self.y_raw >= self.HTL) & (self.y_raw <= self.doc) & m_valid
        if not np.any(mask):
            raise ValueError("CSV does not contain values within [HTL, DoC].")
        x_cut = self.x_raw[mask]
        y_cut = self.y_raw[mask]
        self.x_obs = x_cut - float(self.x_drift)
        self.y_obs = y_cut
        self.y_obs_rel = self.y_obs - self.HTL

    # ------------------- calibration -------------------
    def calibrate(self):
        """
        Calibrate A,B,C,D and hr by least_squares.
        - fit_mode="x" (default): minimize residuals in X (original behavior, robustified)
        - fit_mode="y" (NEW):     minimize residuals in Y
        """
        if not self.data:
            raise ValueError("No data loaded. Use add_data() to load data.")

        # Interpolators need monotonic x or y. Build safe, masked views:
        # X-fit: we need x_obs(h). Build x_obs_interp(h) from y_obs_rel (h).
        def _xfit_residuals(params_log):
            A, B, C, D = np.exp(params_log[:4])
            hr = params_log[4]

            # model x(h)
            xo = ((hr + self.CM) / A)**1.5 - (hr / C)**1.5 + (B / (A**1.5)) * (hr + self.CM)**3 - (D / (C**1.5)) * hr**3
            mask1 = self.h <= (hr + self.CM)
            h1 = self.h[mask1]
            x1 = (h1 / A)**1.5 + B / (A**1.5) * h1**3
            mask2 = ~mask1
            h2 = self.h[mask2] - self.CM
            x2 = (h2 / C)**1.5 + D / (C**1.5) * h2**3 + xo
            x_model = np.empty_like(self.h)
            x_model[mask1] = x1
            x_model[mask2] = x2

            # observed x(h): sort by h (y_obs_rel)
            m = np.isfinite(self.x_obs) & np.isfinite(self.y_obs_rel)
            if not np.any(m):
                return np.array([1e6])
            yr = self.y_obs_rel[m]
            xr = self.x_obs[m]
            idx = np.argsort(yr)
            # only use nonnegative h (seaward from shoreline)
            yr = yr[idx]
            xr = xr[idx]
            mm = yr >= 0.0
            if not np.any(mm):
                return np.array([1e6])
            yr = yr[mm]; xr = xr[mm]

            x_obs_interp = np.interp(self.h, yr, xr, left=np.nan, right=np.nan)
            res = x_model - x_obs_interp
            return res[np.isfinite(res)]

        # NEW: Y-fit: compare y_model(x) to y_obs(x) on overlap
        def _yfit_residuals(params_log):
            A, B, C, D = np.exp(params_log[:4])
            hr = params_log[4]

            # model curve (x(h), y(h))
            xo = ((hr + self.CM) / A)**1.5 - (hr / C)**1.5 + (B / (A**1.5)) * (hr + self.CM)**3 - (D / (C**1.5)) * hr**3
            mask1 = self.h <= (hr + self.CM)
            h1 = self.h[mask1]
            x1 = (h1 / A)**1.5 + B / (A**1.5) * h1**3
            mask2 = ~mask1
            h2 = self.h[mask2] - self.CM
            x2 = (h2 / C)**1.5 + D / (C**1.5) * h2**3 + xo
            x_model_rel = np.concatenate([x1, x2])     # relative to shoreline
            y_model = self.h + self.HTL

            # observed (x, y) relative to shoreline and vertical window
            m = np.isfinite(self.x_obs) & np.isfinite(self.y_obs)
            if not np.any(m):
                return np.array([1e6])
            xo = self.x_obs[m]
            yo = self.y_obs[m]

            # build y(x) by interpolating along model x; need monotonic x
            idxm = np.argsort(x_model_rel)
            xm = x_model_rel[idxm]
            ym = y_model[idxm]
            # overlap only
            xmin, xmax = np.nanmin(xm), np.nanmax(xm)
            mm = (xo >= xmin) & (xo <= xmax)
            if not np.any(mm):
                return np.array([1e6])
            y_pred = np.interp(xo[mm], xm, ym)
            res = y_pred - yo[mm]
            return res[np.isfinite(res)]

        # parameter vector: log(A,B,C,D), hr
        x0 = np.array([np.log(max(self.A,1e-4)), np.log(max(self.B,1e-6)),
                       np.log(max(self.C,1e-4)), np.log(max(self.D,1e-6)),
                       float(self.hr)])
        lb = [np.log(1e-5), np.log(1e-8), np.log(1e-5), np.log(1e-8), 0.0]
        ub = [np.log(1e+1), np.log(1e+2), np.log(1e+1), np.log(1e+2), float(self.h.max())]

        # pick residual function
        fun = _xfit_residuals if self.fit_mode == "x" else _yfit_residuals

        res = least_squares(
            fun, x0, bounds=(lb, ub),
            loss='huber', f_scale=0.1,
            xtol=1e-8, ftol=1e-8, max_nfev=4000
        )

        self.A, self.B, self.C, self.D = np.exp(res.x[:4])
        self.hr = float(res.x[4])
        self.def_hvec()
        return self.run()
