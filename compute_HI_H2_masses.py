import numpy as np
import h5py as h5
from galcalc1 import HI_H2_masses as get_fH2
from Snapshots import get_zstr

def get_params(z):
    # UV Background from HM12
    red        = np.array([0,1,2,3,4,5])
    UVB_all    = np.array([2.27e-14, 3.42e-13, 8.98e-13, 8.74e-13, 6.14e-13, 4.57e-13])
    n0_all     = 10**np.array([-2.56, -2.29, -2.06, -2.13, -2.23, -2.35])
    alpha1_all = np.array([-1.86, -2.94, -2.22, -1.99, -2.05, -2.63])
    alpha2_all = np.array([-0.51, -0.90, -1.09, -0.88, -0.75, -0.57])
    beta_all   = np.array([2.83, 1.21, 1.75, 1.72, 1.93, 1.77])
    f_all      = np.array([0.01, 0.03, 0.03, 0.04, 0.02, 0.01])
    
    UVB    = np.interp(z, red, UVB_all)
    n0     = np.interp(z, red, n0_all)
    alpha1 = np.interp(z, red, alpha1_all)
    alpha2 = np.interp(z, red, alpha2_all)
    beta   = np.interp(z, red, beta_all)
    f      = np.interp(z, red, f_all)
    
    return UVB, n0, alpha1, alpha2, beta, f

def compute_fneutral(snap, NP, sBase, a=1, h=0.6777, Lbox=100):
    mH = 1.6733e-24 # Hydrogen Mass [g]
    if snap == 28: z = 0.0
    if snap == 25: z = 0.271
    if snap == 23: z = 0.503
    if snap == 19: z = 1.004
    if snap == 15: z = 2.012
    Lambda_UVB, n0, a1, a2, beta, f = get_params(z)

    fend = get_zstr(snap)
    fh = h5.File(sBase + '0%i_%s_%iMpc_%i_Gas.hdf5'
                    %(snap, fend, Lbox, NP), 'r')
    
    T           = fh['PartData/TempGas'].value
    UnitDensity = fh['Constants/UnitDensity'].value
    Density     = fh['PartData/Density'].value * h**2 * a**-3 * UnitDensity
    nH          = Density/mH
    del Density
    fh.close()

    alphaA     = 1.269e-13  *  ((315614./T)**1.503)/(1.0 + ((315614./T)/0.522)**0.47)**1.923 
    LambdaT    = 1.17e-10 * ( (np.sqrt(T) * np.exp(-157809./T))/(1.0 + np.sqrt(T/1.0e5)) )
    LambdaPhot = Lambda_UVB *( (1.0-f)*(1.0 + (nH/n0)**beta)**a1  + f*(1.0 + (nH/n0))**a2 )
    
    A = alphaA + LambdaT
    B = 2.* alphaA + LambdaPhot/nH + LambdaT
    del LambdaT, LambdaPhot
    
    term = B**2 - 4*A*alphaA
    sqrt_term = np.array([np.sqrt(B[i]**2 - 4*A[i]*alphaA[i]) if term[i] > 0.0 else 0.0 for i in range(len(T))])
    del term
    
    etas = (B - sqrt_term)/(2.*A)
    etas[etas <= 0] = 1e-30
    
    return etas

def HI_H2_masses(snap, NP, sBase, a=1, h=0.6777, Lbox=100):
        # Compute the HI and H2 masses in each gas particles
        if snap == 28: z = 0.0
        if snap == 25: z = 0.271
        if snap == 23: z = 0.503
        if snap == 19: z = 1.004
        if snap == 15: z = 2.012
        fend = get_zstr(snap)
        
        fh = h5.File(sBase + '0%i_%s_%iMpc_%i_Gas.hdf5' %(snap, fend, Lbox, NP), 'r')
    
        MassGas     = fh['PartData/MassGas'].value/h * 1.e10 # [Msun]
        TempGas     = fh['PartData/TempGas'].value
        SFR         = fh['PartData/SFR'].value     # Msun/yr
        Density     = fh['PartData/Density'].value 
        GasSZ       = fh['PartData/GasSZ'].value
            
        Density  = Density * h**2 * a**-3  * 1.0e10 /1.0e18 # [Msun/pc^3]
        fh.close()
        
        fs = h5.File(sBase + 'PartData/0%i_%s_%iMpc_%i_PartData.hdf5'  %(snap, fend, Lbox, NP), 'r')
        fneutral = fs['fneutral'].value
        fs.close()
        
        #fneutral = compute_fneutral(snap, NP, sBase, a=a, h=h, Lbox=Lbox)
        massHI, massH2 = get_fH2(MassGas, SFR, GasSZ, Density, TempGas, fneutral, z, method=2, UVB='HM12') 
        
        # EAGLE mas units
        massHI /= 1.e10   
        massH2 /= 1.e10 
        
        return massHI, massH2
    
    

    
    