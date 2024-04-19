import numpy as np
from scipy.interpolate import interp1d
from shapely.geometry import LineString
import xarray as xr
import pandas as pd
from IHSetUtils import wMOORE, Hs12Calc, depthOfClosure

class cal_Bernabeu(object):
    """
    cal_Bernabeu
    
    Configuration to calibrate and run the Bernabeu profile.
    
    This class reads input datasets, calculates its parameters.
    """
    def __init__(self, path_prof, path_wav, Switch_Obs, Switch_Cal_DoC, **kwargs):
        self.path_prof = path_prof
        self.path_wav = path_wav
        prof = pd.read_csv(path_prof)

        self.Hs50 = kwargs['Hs50']
        self.Tp50 = kwargs['Tp50']
        self.D50 = kwargs['D50']
        self.HTL = -kwargs['HTL']    
        self.LTL = -kwargs['LTL']
        self.CM = abs(self.HTL - self.LTL)
        self.xm = np.linspace(kwargs['Xm'][0], kwargs['Xm'][1], 1000).reshape(-1, 1)
        
        self.Switch_Obs = Switch_Obs              # Do you have profile data? (0: no; 1: yes)
        if Switch_Obs == 1:
            self.xp = prof.iloc[:, 0]
            self.zp = prof.iloc[:, 1]
            self.zp = abs(self.zp)
            xp_inx = self.xp[(self.zp >= self.HTL)]
            self.xp = self.xp - min(xp_inx)
            
        self.Switch_Cal_DoC = Switch_Cal_DoC
        if Switch_Cal_DoC == 1:                   # Calculate Depth of Closure if you have wave data [0: no; 1: yes]
            wav = xr.open_dataset(path_wav)
            Hs = wav['Hs'].values
            Hs = Hs.reshape(-1, 1)
            Tp = wav['Tp'].values
            Tp = Tp.reshape(-1, 1)
            
            H12,T12 = Hs12Calc(Hs,Tp)
            self.DoC = depthOfClosure(H12,T12)
            # self.DoC = self.DoC[0]
                          
    def params(self):        
        ws = wMOORE(self.D50)
        gamma = self.Hs50 / (ws * self.Tp50)
    
        self.Ar = 0.21 - 0.02 * gamma
        self.B = 0.89 * np.exp(-1.24 * gamma)
        self.C = 0.06 + 0.04 * gamma
        self.D = 0.22 * np.exp(-0.83 * gamma)    

        self.a = self.Ar**(-1.5)
        self.b = self.B / self.Ar**(1.5)
        self.c = self.C**(-1.5)
        self.d = self.D / self.C**(1.5)
        
        self.ha = 3 * self.Hs50
        self.hr = 1.1 * self.Hs50
        
        return self       

def Bernabeu(self):
    Xr = ((self.hr + self.CM) / self.Ar)**(1.5) + self.B / self.Ar**(1.5) * (self.hr + self.CM)**3
    Xo = Xr - (self.hr / self.C)**(1.5) - self.D / self.C**(1.5) * self.hr**3
    Xa = Xo + self.c * (self.ha**(1.5)) + self.d * self.ha**3

    hini = np.arange(0, 10.1, 0.1)
    xini = self.a * hini**1.5 + self.b * hini**3

    rotura = 0
    hrot = rotura - hini
    xrot = xini + rotura

    xdos = self.c * hini**(1.5) + self.d * hini**3

    haso = rotura - hini - self.CM
    xaso = xdos + Xo + rotura

    line_coords1 = list(zip(xaso, haso))
    polygon1 = LineString(line_coords1)
    
    line_coords2 = list(zip(xrot, hrot))
    polygon2 = LineString(line_coords2)
    intersection = polygon1.intersection(polygon2)
    iX = np.array(intersection.xy).T[0][0]
    iZ = np.array(intersection.xy).T[0][1]

    haso = haso[iX < xaso]
    xaso = xaso[iX < xaso]

    hrot = hrot[iX > xrot]
    xrot = xrot[iX > xrot]

    X = np.concatenate([xrot, xaso])
    hm = np.concatenate([hrot, haso])

    self.zm = -interp1d(X, hm, kind='linear', fill_value='extrapolate')(self.xm) + self.HTL 
    if self.Switch_Cal_DoC == 1:
        self.xm_DoC = np.mean(self.xm[(self.zm <= self.DoC + 0.05) & (self.zm >= self.DoC - 0.05)])
        self.zm_DoC = np.mean(self.zm[(self.zm <= self.DoC + 0.05) & (self.zm >= self.DoC - 0.05)])

    return self  
