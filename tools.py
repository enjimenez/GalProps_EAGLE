import numpy as np
from scipy.integrate import quad

h = 0.6777
OmegaL0 = 0.693
OmegaM0 = 0.307
cm_Mpc = 3.085678e24
cm_km = 1e5
s_Gyr = 3.155e16
H0 = 100*h *cm_km/cm_Mpc *s_Gyr

def distance(r1, r0, Lbox):
    dims = np.ones(3)*Lbox
    delta = np.abs(r1 - r0)
    delta = np.where(delta > 0.5 * dims, delta - dims, delta)
    return np.sqrt((delta ** 2).sum(axis=-1))

def DeltaR(r, r0, Lboxkpc): 
    deltaR = r - r0
    Lx, Lxabs = deltaR[0], abs(deltaR[0])
    Ly, Lyabs = deltaR[1], abs(deltaR[1])
    Lz, Lzabs = deltaR[2], abs(deltaR[2])
    if Lxabs >= Lboxkpc/2.:
        if Lx > 0: deltaR[0] = Lxabs - Lboxkpc
        else: deltaR[0] = Lboxkpc - Lxabs
    if Lyabs >= Lboxkpc/2.:
        if Ly > 0: deltaR[1] = Lyabs - Lboxkpc
        else: deltaR[1] = Lboxkpc - Lyabs
    if Lzabs >= Lboxkpc/2.:
        if Lz > 0: deltaR[2] = Lzabs - Lboxkpc
        else: deltaR[2] = Lboxkpc - Lzabs
    return deltaR


