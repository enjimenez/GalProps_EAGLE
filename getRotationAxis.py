import h5py as h5
import numpy as np
from tools import distance

def getL(itype, data, Tree):
    # Angular momentum in [1e10 Msun Mpc km/s]
    keys = ['h','a', 'snap', 'fend', 'Lbox', 'NP', 'Base', 'BasePart']
    h,a,snap,fend,Lbox,NP,Base,BasePart = [data['params'].get(key) for key in keys]
    if itype == 0: itype = 'Gas'
    if itype == 4: itype = 'Star'
    
    gns  = data['gn']
    sgns = data['sgn']
    Pos_cop = data['Pos_cop']
    V_com   = data['Vcom']    #TODO should I use the COP vel instead? NOP
    
    LenGal = len(Pos_cop)
    Aps = [5,7,10,20,30,70,100]
    LenAps = len(Aps)
    Larr = np.zeros((LenAps, LenGal, 3), dtype=float)
    
    with h5.File(BasePart + '%s_%s_%iMpc_%i_%s.hdf5' %(snap, fend, Lbox, NP, itype), 'r') as f:
        PosPart      = f['PartData/Pos%s' %itype][()]/h
        VelPart      = f['PartData/Vel%s' %itype][()] * np.sqrt(a) # Transfor to peculiar
        MassPart     = f['PartData/Mass%s' %itype][()]/h    
        gnPart       = f['PartData/GrpNum_%s' %itype][()]
        sgnPart      = f['PartData/SubNum_%s' %itype][()]
        
    for i,idx in enumerate(Tree.query_ball_point(Pos_cop, (101/1000)/a)):
        gnGal = gnPart[idx]
        sgnGal = sgnPart[idx]
        mask = (gnGal == gns[i]) & (sgnGal == sgns[i])
        LenPart = len(np.where(mask)[0])
        if LenPart == 0: continue
        
        COP = Pos_cop[i]
        P = PosPart[idx][mask]
        M = MassPart[idx][mask]
        V = VelPart[idx][mask]
        
        P = np.mod(P - COP  + 0.5*Lbox , Lbox) + COP -0.5*Lbox
        P -= COP
        V -= V_com[i]
    
        r = np.sqrt(np.sum(np.power(P,2), axis=1)) * a * 1e3
        rsort = np.argsort(r)
    
        p = M[:,np.newaxis] * V
        L = np.cross(P, p, axis=1)
        L = L[rsort]
        r = r[rsort]
        Lcum = np.cumsum(L, axis=0)
        
        
        for j, Ap in enumerate(Aps):
            argMin = np.argmin(abs(r - Ap))
            Ltemp = Lcum[argMin]
            Larr[j,i,] = Ltemp/np.linalg.norm(Ltemp)
        
    return Larr
