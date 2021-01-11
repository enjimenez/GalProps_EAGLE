import numpy as np
import h5py as h5
from tools import distance
from numpy.linalg import norm 

def profile(e_arr, mass_arr, dist_arr, rmax):
    LenPart = len(e_arr)
    cumsum_egas = np.cumsum(e_arr)
    cumsum_mass = np.cumsum(mass_arr)
    sigma_val = np.sqrt(cumsum_egas/cumsum_mass)
    
    Radii = np.arange(1, rmax+1, 1).astype(int)
    Sigma_tmp = np.zeros(rmax, dtype=float)
    Num_tmp   = np.zeros(rmax, dtype=int)      
    
    st = 0
    while Radii[st] < dist_arr[0]: st+=1 
    
    j = 0
    for k, R in enumerate(Radii[st:]):
        while dist_arr[j] < R:
            j += 1
            if j == LenPart: break
        Sigma_tmp[st:][k] = sigma_val[j-1]
        Num_tmp[st:][k] = j
        if j == LenPart: break
            
    if R != rmax: 
        Sigma_tmp[R-1:] = np.ones(rmax-(R-1)) * sigma_val[-1]
        Num_tmp[R-1:] = np.ones(rmax-(R-1)) * Num_tmp[R-1]
        
    return Sigma_tmp, Num_tmp

def Gas(data, Tree, rmax=50, SF=True, All=False, feedback=False, nofeed=False, return_noth=False):
    keys = ['h','a', 'snap', 'fend', 'Lbox','L', 'NP', 'Base', 'BasePart']
    h,a,snap,fend,Lbox,L,NP,Base,BasePart = [data['params'].get(key) for key in keys]
    
    Pos_cop      = data['Pos_cop'] #[cMpc]
    Vcom         = data['Vcom']
    SpinStar     = data['SpinStar']
    SpinSF       = data['SpinSF']  
    gns          = data['gn']
    sgns         = data['sgn']
    
    LenGal       = len(Pos_cop)
    sigma        = np.zeros((LenGal,rmax), dtype=np.float64) 
    NumPart      = np.zeros((LenGal,rmax), dtype=int)
    if return_noth: sigma_nth = np.zeros((LenGal,rmax), dtype=np.float64)
    
    # === Get Gas Data === #
    fh = h5.File(BasePart + '%s_%s_%iMpc_%i_Gas.hdf5' %(snap, fend, L, NP), 'r')
    
    SFR = fh['PartData/SFR'][()]
    mask = SFR > 0 if SF else np.ones(len(SFR)).astype(bool)
    del SFR
    
    PosPart      = fh['PartData/PosGas'][()][mask]/h  #[cMpc]
    MassPart     = fh['PartData/MassGas'][mask]/h 
    VPart        = fh['PartData/VelGas'][()][mask] * np.sqrt(a)
    Density      = fh['PartData/Density'][mask] 
    Entropy      = fh['PartData/Entropy'][mask]
    gnGas        = fh['PartData/GrpNum_Gas'][mask]
    sgnGas       = fh['PartData/SubNum_Gas'][mask]
    UnitPressure = fh['Constants/UnitPressure'][()]
    UnitDensity  = fh['Constants/UnitDensity'][()]
    gamma        = fh['Constants/gamma'][()]

    if feedback or nofeed: 
        # Use them 'All' option ONLY
        Tmax         = fh['PartData/MaximumTemperature'][()]/(10**7.0)
        Tvir         = fh['PartData/HostHalo_TVir_Mass'][()]/(10**7.0)
        amax         = fh['PartData/AExpMaximumTemperature'][()] 
    
    Density  = Density * (h**2.0) * (a**(-3.0))  * UnitDensity
    Entropy  = Entropy * (h**(2.-2.*gamma)) * UnitPressure * UnitDensity**(-1.0*gamma)
    Pressure = Entropy*Density**gamma
    del Entropy
    fh.close()
    
    #######################################################################         

    for i in range(LenGal):
        if SF:
            Tree.search(center=Pos_cop[i], radius=(rmax/1000. + 0.005)/a) #[rad] = cMpc
            idx = Tree.getIndices()
            if idx is None: continue
            gns_gal  = gnGas[idx]
            sgns_gal = sgnGas[idx]
            masksgn = (gns_gal == gns[i]) & (sgns_gal == sgns[i]) 
                
            Pressure_gal = Pressure[idx][masksgn]
            Density_gal  = Density[idx][masksgn]
            MassPart_gal = MassPart[idx][masksgn]
            PosPart_gal  = PosPart[idx][masksgn]
            VPart_gal    = VPart[idx][masksgn]
            LenPart      = len(MassPart_gal)
            if LenPart == 0: continue
        
        else:
            Tree.search(center=Pos_cop[i], radius=(rmax/1000. + 0.005)/a)
            idx = Tree.getIndices()
            if idx is None: continue
            gns_gal  = gnGas[idx]
            sgns_gal = sgnGas[idx]
            masksgn = (gns_gal == gns[i]) & (sgns_gal == sgns[i]) 
            
            
            LenPart = len(np.where(masksgn)[0])
            mask = np.ones(LenPart).astype(bool)
                
            if feedback:
                if int(snap) == 28: acut = 0.998
                if int(snap) == 19: acut = 0.497
                if int(snap) == 15: acut = 0.330
                
                Tmax_tmp = Tmax[idx][masksgn] 
                amax_tmp = amax[idx][masksgn]
                Tvir_tmp = Tvir[idx][masksgn]
                
                Tmask1 = (Tmax_tmp >= 1.0) & (Tvir_tmp < 1.0)
                Tmask2 = (Tmax_tmp > Tvir_tmp) & (Tvir_tmp > 1.0)
                Tmaskf = np.logical_or(Tmask1, Tmask2)
                
                mask = mask & Tmaskf & (amax_tmp > acut)
                LenPart = len(np.where(mask)[0])
            
            if nofeed:
                mask = mask & (Tmax[idx][masksgn] < 1) 
                LenPart = len(np.where(mask)[0])
            
            if LenPart == 0: continue    
        
            Pressure_gal = Pressure[idx][masksgn][mask]
            Density_gal  = Density[idx][masksgn][mask]
            MassPart_gal = MassPart[idx][masksgn][mask]
            PosPart_gal  = PosPart[idx][masksgn][mask]
            VPart_gal    = VPart[idx][masksgn][mask]
       
        centre = Pos_cop[i]
        r = distance(PosPart_gal, centre, Lbox) *a * 1e3 # [pkpc]
        tt = np.argsort(r)
        r = r[tt]
        if r[0] > rmax: continue
        
        delta_V = VPart_gal[tt] - Vcom[i]
        MassPart_sorted = MassPart_gal[tt]
        e_gas_wthermal = np.zeros(LenPart, dtype=float)
        e_gas_wothermal = np.zeros(LenPart, dtype=float)
        sigmaP  = (np.sqrt(Pressure_gal[tt]/Density_gal[tt])) * 1.0e-5  # [km/s] 
        L_SpinSF = norm(SpinSF[i])
        L_SpinStar = norm(SpinStar[i])
        
        for k in range(LenPart):
            if L_SpinSF != 0: cos_theta = np.dot(delta_V[k], SpinSF[i])/(norm(delta_V[k])*L_SpinSF)
            else: cos_theta = np.dot(delta_V[k], SpinStar[i])/(norm(delta_V[k])*L_SpinStar)
            e_gas_wthermal[k] = MassPart_sorted[k] * ((norm(delta_V[k])*cos_theta)**2.0 + (1./3)*sigmaP[k]**2.0)
            e_gas_wothermal[k] = MassPart_sorted[k] * ((norm(delta_V[k])*cos_theta)**2.0)
            
        sigma[i], NumPart[i]  = profile(e_gas_wthermal, MassPart_sorted, r, rmax)
        if return_noth: sigma_nth[i],tmp = profile(e_gas_wothermal, MassPart_sorted, r, rmax)
    
    if return_noth: return sigma, sigma_nth, NumPart
    else: return sigma, NumPart
        
def Stars(data, Tree, rmax=50):
    keys = ['h','a', 'snap', 'fend', 'Lbox','L', 'NP', 'Base', 'BasePart']
    h,a,snap,fend,Lbox,L,NP,Base,BasePart = [data['params'].get(key) for key in keys]
    print(Lbox, L)
    
    Pos_cop      = data['Pos_cop']
    Vcom         = data['Vcom']
    SpinStar     = data['SpinStar']   #[pkpc km/s]
    gns          = data['gn']
    sgns         = data['sgn']
    
    LenGal       = len(Pos_cop)
    sigma        = np.zeros((LenGal,rmax), dtype=np.float64) 
    NumPart      = np.zeros((LenGal,rmax), dtype=int)
    
    # === Get Star Data ===
    fh = h5.File(BasePart + '%s_%s_%iMpc_%i_Star.hdf5' %(snap, fend, L, NP), 'r')
    MassPart    = fh['PartData/MassStar'][()]/h
    PosPart     = fh['PartData/PosStar'][()]/h
    VPart       = fh['PartData/VelStar'][()] * np.sqrt(a)
    sgnStar     = fh['PartData/SubNum_Star'][()]
    fh.close()
    #########################
    
    for i in range(LenGal):
        Tree.search(center=Pos_cop[i],radius=(rmax/1000.+0.005)/a)
        idx = Tree.getIndices()
        if idx is None: continue
        sgns_gal = sgnStar[idx]
        masksgn = sgns_gal == sgns[i]
        LenPart = len(sgns_gal[masksgn])
        if len(sgns_gal[masksgn]) == 0: continue
        
        MassPart_gal = MassPart[idx][masksgn]
        PosPart_gal  = PosPart[idx][masksgn]
        VPart_gal    = VPart[idx][masksgn]
        LenPart      = len(MassPart_gal)
        
        centre = Pos_cop[i]
        r = distance(PosPart_gal, centre, Lbox) * a * 1e3
        tt = np.argsort(r)
        r = r[tt]
        if r[0] > rmax: continue
        
        delta_V = VPart_gal[tt] - Vcom[i]
        MassPart_sorted = MassPart_gal[tt]
        e_gas = np.zeros(LenPart, dtype=float)
        
        for k in range(LenPart): 
            cos_theta = np.dot(delta_V[k], SpinStar[i])/(norm(delta_V[k])*norm(SpinStar[i]))
            e_gas[k] = MassPart_sorted[k] * ((norm(delta_V[k])*cos_theta)**2.0)

        sigma[i], NumPart[i]  = profile(e_gas, MassPart_sorted, r, rmax=rmax)

    return sigma, NumPart

def DM(data, Tree, rmax=50):
    keys = ['h','a', 'snap', 'fend', 'Lbox','L', 'NP', 'Base', 'BasePart']
    h,a,snap,fend,Lbox,L,NP,Base,BasePart = [data['params'].get(key) for key in keys]

    Pos_cop      = data['Pos_cop']
    Vcom         = data['Vcom']
    SpinStar     = data['SpinStar']
    gns          = data['gn']
    sgns         = data['sgn']
    
    LenGal       = len(Pos_cop)
    sigma        = np.zeros((LenGal,rmax), dtype=np.float64) 
    NumPart      = np.zeros((LenGal,rmax), dtype=int)
    
    # === Get DM data === #
    fh = h5.File(BasePart + '%s_%s_%iMpc_%i_DM.hdf5' %(snap, fend, L, NP), 'r')

    PosPart    = fh['PartData/PosDM'][()]/h 
    VPart      = fh['PartData/VelDM'][()] * np.sqrt(a)
    sgnDM      = fh['PartData/SubNum_DM'][()]
    MassPart   = fh['Header/PartMassDM'][()]/h 
    fh.close()
    #######################
    
    for i in range(LenGal):
        Tree.search(center=Pos_cop[i],radius=(rmax/1000.+0.005)/a)
        idx = Tree.getIndices()
        if idx is None: continue
        sgns_gal = sgnDM[idx]
        masksgn = sgns_gal == sgns[i]
        LenPart = len(sgns_gal[masksgn])
        if len(sgns_gal[masksgn]) == 0: continue
        
        PosPart_gal  = PosPart[idx][masksgn]
        VPart_gal    = VPart[idx][masksgn]
        LenPart      = len(PosPart_gal)
        MassPart_gal = np.ones(LenPart)*MassPart
        
        centre = Pos_cop[i]
        r = distance(PosPart_gal, centre, Lbox) * a * 1e3 
        tt = np.argsort(r)
        r = r[tt]
        if r[0] > rmax: continue
        
        delta_V = VPart_gal[tt] - Vcom[i]
        MassPart_sorted = MassPart_gal[tt]
        e_gas = np.zeros(LenPart, dtype=float)
    
        for k in range(LenPart): 
            cos_theta = np.dot(delta_V[k], SpinStar[i])/(norm(delta_V[k])*norm(SpinStar[i]))
            e_gas[k] = MassPart_sorted[k] * ((norm(delta_V[k])*cos_theta)**2.0)
        
        sigma[i], NumPart[i] = profile(e_gas, MassPart_sorted, r, rmax)
                
    return sigma, NumPart

