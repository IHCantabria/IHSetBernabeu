import numpy as np
from scipy.interpolate import interp1d
from shapely.geometry import LineString

def caida_grano(D50):
    ws = np.nan
    if D50 < 0.1:
        ws = 1.1e6 * (D50 * 0.001) ** 2
    elif 0.1 <= D50 <= 1:
        ws = 273 * (D50 * 0.001) ** 1.1
    elif D50 > 1:
        ws = 4.36 * D50**0.5
    return ws


def RMSEq(Y, Y2t):
    return np.sqrt(np.mean((Y - Y2t) ** 2, axis=0))


def Bernabeu(dp, zp, CM, D50, Hs, Tp):
    z = zp - zp[0]
    d = dp - dp[0]
    
    # Profile with equidistant points
    dp = np.linspace(0, dp[-1], 500).reshape(-1, 1)  # 500 points
    interp_func = interp1d(d, z, kind='linear', fill_value='extrapolate')
    zp = interp_func(dp)
    zp = zp[1:]
    dp = dp[1:]
    
    rotura = 0
    ws = caida_grano(D50)
    gamma = Hs / (ws * Tp)
    
    Ar = 0.21 - 0.02 * gamma
    B = 0.89 * np.exp(-1.24 * gamma)
    C = 0.06 + 0.04 * gamma
    D = 0.22 * np.exp(-0.83 * gamma)    

    Ks = [Ar, B, C, D]

    a = Ar**(-1.5)
    b = B / Ar**(1.5)
    c = C**(-1.5)
    d = D / C**(1.5)

    ha = 3 * Hs
    hr = 1.1 * Hs

    Xr = ((hr + CM) / Ar)**(1.5) + B / Ar**(1.5) * (hr + CM)**3
    Xo = Xr - (hr / C)**(1.5) - D / C**(1.5) * hr**3
    Xa = Xo + c * (ha**(1.5)) + d * ha**3

    hini = np.arange(0, 10.1, 0.1)
    xini = a * hini**1.5 + b * hini**3

    hrot = rotura - hini
    xrot = xini + rotura

    xdos = c * hini**(1.5) + d * hini**3

    haso = rotura - hini - CM
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

    hmi = interp1d(X, hm, kind='linear', fill_value='extrapolate')(dp)  

    err = RMSEq(zp, hmi)

    Para = {
        'model': 'Bernabeu (Dos tramos)',
        'formulation': [
            r'$x_r = \frac{h}{A_r}^{1.5} + \frac{B}{A_r^{1.5}}h^3$',
            r'$x_a = x - x_0 = \left(\frac{h-M}{C}\right)^{1.5} + \frac{D}{C^{1.5}}(h-M)^3$'
        ],
        'name_coeffs': ['Ar', 'B', 'C', 'D'],
        'coeffs': Ks,
        'RMSE': err
    }
    
    model = {'D': np.concatenate([[0], dp.flatten()]), 'Z': np.concatenate([[0], hmi.flatten()])}

    return Para, model          