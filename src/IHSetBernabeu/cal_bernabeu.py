import numpy as np
import pandas as pd
from .bernabeu import Bernabeu
from IHSetUtils import wMOORE
from scipy.optimize import minimize


class cal_Bernabeu(object):
    """
    cal_Bernabeu
    
    Configuration to calibrate and run the Bernabeu profile.
    
    This class reads input datasets, calculates its parameters.
    """
    def __init__(self, CM, Hs50, D50, Tp50, doc, hr, HTL = 0):
        self.CM = CM
        self.Hs50 = Hs50
        self.D50 = D50
        self.Tp50 = Tp50
        self.doc = doc
        self.HTL = HTL
        self.hr = hr  
        self.x_obs = None   # observed cross-shore distance (m)
        self.y_obs = None   # observed profile elevation (m, positive upwards)
        self.data = False   # flag to check if data is loaded

        # Calculate parameters
        self.params()
        self.def_hvec()

    def params(self):

        ws = wMOORE(self.D50 / 1000)  # Convert D50 from mm to m
        gamma = self.Hs50 / (ws * self.Tp50)

        self.Ar = 0.21 - 0.02 * gamma
        self.B = 0.89 * np.exp(-1.24 * gamma)
        self.C = 0.06 + 0.04 * gamma
        self.D = 0.22 * np.exp(-0.83 * gamma)

    def def_xo(self):
        """ Calculate the offset for the profile based on hr and CM. """

        self.xo = ((self.hr + self.CM) / self.Ar)**(3/2) - (self.hr / self.C)**(3/2) + self.B / (self.Ar**(3/2)) * (self.hr + self.CM)**3 - self.D / (self.C**(3/2)) * self.hr**3

    def run(self):
        """
        Run the Bernabeu profile with the current parameters.
        """
        self.def_xo()

        x, x1, x2, y2 = Bernabeu(self.Ar, self.B, self.C, self.D, self.CM, self.h, self.xo)

        self.x1 = x1
        self.x2 = x2
        self.y2 = y2
        
        return (x, self.h + self.HTL)  # Return x and y in absolute coordinates (relative to HTL)

    def def_hvec(self):
        self.h = np.arange(0.1, self.CM + self.doc, 0.001)

    def from_D50(self, D50):
        """
        Calculate the Bernabeu profile parameters from D50.
        """
        self.D50 = D50
        self.params()
        
        return self.run()
    
    def from_Hs50(self, Hs50):
        """
        Calculate the Bernabeu profile parameters from Hs50.
        """
        self.Hs50 = Hs50
        self.params()
        
        return self.run()
    
    def from_Tp50(self, Tp50):
        """
        Calculate the Bernabeu profile parameters from Tp50.
        """
        self.Tp50 = Tp50
        self.params()
        
        return self.run()
    
    def change_hr(self, hr):
        """
        Change the height of the reference level (hr) and recalculate the Bernabeu profile.
        """
        self.hr = hr
        self.params()
        
        return self.run()
    
    def change_CM(self, CM):
        """
        Change the tidal range (CM) and recalculate the Bernabeu profile.
        """
        self.CM = CM
        self.params()
        self.def_hvec()
        
        return self.run()
    
    def change_doc(self, doc):
        """
        Change the depth of closure (doc) and recalculate the Bernabeu profile.
        """
        self.doc = doc
        self.def_hvec()
        
        return self.run()
    
    def change_HTL(self, HTL):
        """
        Change the height of the tidal limit (HTL) and recalculate the Bernabeu profile.
        """
        self.HTL = HTL
        self.def_hvec()
        
        return self.run()
    
    def change_A(self, A):
        """
        Change the parameter A and recalculate the Bernabeu profile.
        """
        self.Ar = A

        return self.run()
    
    def change_B(self, B):
        """
        Change the parameter B and recalculate the Bernabeu profile.
        """
        self.B = B
        
        return self.run()
    
    def change_C(self, C):
        """
        Change the parameter C and recalculate the Bernabeu profile.
        """
        self.C = C
        
        return self.run()
    
    def change_D(self, D):
        """
        Change the parameter D and recalculate the Bernabeu profile.
        """
        self.D = D
        
        return self.run()
    
    def add_data(self, path):
        
        df = pd.read_csv(path)
        self.x_raw = df["X"].values
        self.y_raw = df["Y"].values

        # positive depth down
        if np.mean(self.y_raw) < 0:
            self.y_raw = -self.y_raw

        # cut between SL and DoC
        m = (self.y_raw >= self.HTL) & (self.y_raw <= self.doc)
        if not np.any(m):
            raise ValueError("CSV does not contain values between SL and DoC.")
        self.x_obs = self.x_raw[m]
        self.x_obs = self.x_obs - min(self.x_obs)  # Normalize x_obs to start from 0
        self.y_obs = self.y_raw[m]

        self.y_obs_rel = self.y_obs - self.HTL

        self.data = True

        return self.calibrate()
    
    def calibrate(self):
        """
        Calibrate the Bernabeu profile with the observed data.
        Calibrates the parameters A, B, C, D and hr
        """
        if not self.data:
            raise ValueError("No data loaded. Use add_data() to load data.")
        
        x_obs_interp = np.interp(self.h, self.y_obs_rel, self.x_obs)
        
        def func(params):
            A, B, C, D, hr = np.exp(params)
            # Ensure parameters are positive
            # A = np.exp(A)
            # B = np.exp(B)
            # C = np.exp(C)
            # D = np.exp(D)
            xo = ((hr + self.CM) / A)**(3/2) - (hr / C)**(3/2) + B / (A**(3/2)) * (hr + self.CM)**3 - D / (C**(3/2)) * hr**3
            x = Bernabeu(A, B, C, D, self.CM, self.h, xo)[0]
            return np.sum((x - x_obs_interp)**2)
        
        initial_guess = [np.log(self.Ar), np.log(self.B), np.log(self.C), np.log(self.D), np.log(self.hr)]
        result = minimize(func, initial_guess, method='L-BFGS-B')

        A, B, C, D, hr = result.x
        self.Ar = np.exp(A)
        self.B = np.exp(B)
        self.C = np.exp(C)
        self.D = np.exp(D)
        self.hr = np.exp(hr)

        self.def_hvec()

        return self.run()
    


        

