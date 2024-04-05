import numpy as np
from scipy.interpolate import interp1d
from shapely.geometry import LineString
import xarray as xr
from IHSetUtils import wMOORE

class cal_Bernabeu(object):
    """
    cal_Bernabeu
    
    Configuration to calibrate and run the Bernabeu profile.
    
    This class reads input datasets, performs its calibration.
    """
    def __init__(self, path):
        self.path = path
        
        # cfg = xr.open_dataset(path+'config.nc')
        ens = xr.open_dataset(path+'ens.nc')
        wav = xr.open_dataset(path+'wav.nc')
                
        self.D50 = ens['D50'].values
        self.dp = ens['d'].values
        self.zp = ens['z'].values
        self.CM = ens['CM_95'].values
        self.Hs = wav['Hs50'].values
        self.Tp = wav['Tp50'].values
                
    def calibrate(self):
        self.zp = self.zp - self.zp[0]
        self.dd = self.dp - self.dp[0]

        # Profile with equidistant points
        dp = np.linspace(0, self.dp[-1], 500).reshape(-1, 1)
        
        interp_func = interp1d(self.dd, self.zp, kind="linear", fill_value="extrapolate")
        zp = interp_func(dp)
        zp = zp[1:]
        dp = dp[1:]
        
        ws = wMOORE(self.D50)
        gamma = self.Hs / (ws * self.Tp)
    
        self.Ar = 0.21 - 0.02 * gamma
        self.B = 0.89 * np.exp(-1.24 * gamma)
        self.C = 0.06 + 0.04 * gamma
        self.D = 0.22 * np.exp(-0.83 * gamma)    

        self.a = self.Ar**(-1.5)
        self.b = self.B / self.Ar**(1.5)
        self.c = self.C**(-1.5)
        self.d = self.D / self.C**(1.5)
        
        self.ha = 3 * self.Hs
        self.hr = 1.1 * self.Hs
        
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

    self.hmi = interp1d(X, hm, kind='linear', fill_value='extrapolate')(self.dp)  

    # err = RMSEq(self.zp, self.hmi)

    return self   