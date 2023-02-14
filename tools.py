from scipy.interpolate import InterpolatedUnivariateSpline as interp
from scipy.stats import binned_statistic as b1d
from scipy.integrate import quad
import numpy as np
import pandas as pd

h = 0.6777
OmegaL0 = 0.693
OmegaM0 = 0.307
cm_Mpc = 3.085678e24
cm_km = 1e5
s_Gyr = 3.155e16
H0 = 100*h *cm_km/cm_Mpc *s_Gyr

integrand = lambda z: ((1+z)*np.sqrt(OmegaL0 + OmegaM0*(1+z)**3))**-1

def distance(r1, r0, Lbox):
    dims = np.ones(3)*Lbox
    delta = np.abs(r1 - r0)
    delta = np.where(delta > 0.5 * dims, delta - dims, delta)
    return np.sqrt((delta ** 2).sum(axis=-1))

def LBT(a):
    z = 1/a - 1
    return (1/H0)*quad(integrand, 0, z)[0]
    
def ScaleFactor(ain, tstep):
    zarr = np.concatenate((np.arange(0,1,.001),np.arange(1,5,0.01)))
    lenz = len(zarr)
    lbtarr = np.zeros(lenz, dtype=float)
    for i,z in enumerate(zarr): lbtarr[i] = (1/H0)*quad(integrand, 0, z)[0]
    LBT_in = LBT(ain)
    sol = interp(zarr, lbtarr - (LBT_in + tstep/1e3)).roots()
    return(1/(1+sol))

def RotMatrix(a,b):
    """
    Rodrigues' 3D rotation formula
    a: vector to be rotated
    b: target vector
    """
    amod = np.linalg.norm(a)
    bmod = np.linalg.norm(b)
    k = np.cross(a,b)
    k = k/np.linalg.norm(k)
    theta = np.arccos(np.dot(a,b)/(amod*bmod))
    
    K = np.array([[0, -k[2], k[1]],
                 [k[2], 0, -k[0]],
                 [-k[1], k[0], 0]])
    
    R = np.identity(3) + np.sin(theta)*K + (1 - np.cos(theta))*np.matmul(K,K)
    return R

def getpercentiles(x, y, bins, lo=16, hi=84):
        NaN = np.isnan(y)
        x = x[~NaN]
        y = y[~NaN]
        if len(x) == 0: return
        med = b1d(x, y, 'median', bins=bins).statistic
        lo  = b1d(x, y, statistic=lambda y: np.percentile(y, lo), bins=bins).statistic
        hi  = b1d(x, y, statistic=lambda y: np.percentile(y, hi), bins=bins).statistic
        return np.array([med,lo,hi]).T
