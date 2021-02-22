import numpy as np
import h5py as h5
from tools import distance
from numpy.linalg import norm 

def Gas(data, Tree, rmax=50, SF=True, All=False, feedback=False, nofeed=False, return_noth=False):
    keys = ['h','a', 'snap', 'fend', 'Lbox','L', 'NP', 'Base', 'BasePart', 'BaseGal']
    h,a,snap,fend,Lbox,L,NP,Base,BasePart,BaseGal = [data['params'].get(key) for key in keys]
    
    Pos_cop      = data['Pos_cop'] #[cMpc]
    Vcom         = data['Vcom']
    SpinStar     = data['SpinStar']
    SpinSF       = data['SpinSF']  
    gns          = data['gn']
    sgns         = data['sgn']
    
    with h5.File(BaseGal + '%s_%s_%iMpc_%i_galaxies.hdf5' %(snap, fend, L, NP), 'r') as f:
         RotAxis5 = f['GasData/RotAxes/L_5kpc'][()]
         RotAxis7 = f['GasData/RotAxes/L_7kpc'][()]
         RotAxis10 = f['GasData/RotAxes/L_10kpc'][()]
         RotAxis20 = f['GasData/RotAxes/L_20kpc'][()]
         RotAxis30 = f['GasData/RotAxes/L_30kpc'][()]
         RotAxis70 = f['GasData/RotAxes/L_70kpc'][()]
         RotAxis100 = f['GasData/RotAxes/L_100kpc'][()]
        
    LenGal       = len(Pos_cop)
    NRotAxis = 7
    sigma = np.zeros((NRotAxis, LenGal, rmax), dtype=np.float32)
    NumPart      = np.zeros((LenGal,rmax), dtype=int)
    if return_noth: sigma_nth = np.zeros((LenGal,rmax), dtype=np.float64)
    
    # === Get Gas Data === #
    with h5.File(BasePart + '%s_%s_%iMpc_%i_Gas.hdf5' %(snap, fend, L, NP), 'r') as fh:
        #SFR = fh['PartData/SFR'][()]
        #mask = SFR > 0 if SF else np.ones(len(SFR)).astype(bool)
        #del SFR
        
        PosPart      = fh['PartData/PosGas'][()]/h  #[cMpc]
        MassPart     = fh['PartData/MassGas'][()]/h 
        VPart        = fh['PartData/VelGas'][()] * np.sqrt(a)
        Density      = fh['PartData/Density'][()] 
        Entropy      = fh['PartData/Entropy'][()]
        gnGas        = fh['PartData/GrpNum_Gas'][()]
        sgnGas       = fh['PartData/SubNum_Gas'][()]
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
    
    #######################################################################         
    for i, idx in enumerate(Tree.query_ball_point(Pos_cop, (rmax/1000. + 0.005)/a)):
        if SF:
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
        sigmaP  = (np.sqrt(Pressure_gal[tt]/Density_gal[tt])) * 1.0e-5  # [km/s] 
        
        L_5kpc = RotAxis5[i]
        L_7kpc = RotAxis7[i]
        L_10kpc = RotAxis10[i]
        L_20kpc = RotAxis20[i]
        L_30kpc = RotAxis30[i]
        L_70kpc = RotAxis70[i]
        L_100kpc = RotAxis100[i]
        #L_SpinSF = norm(SpinSF[i])    # only for snapshots
        #L_SpinStar = SpinStar[i]
        
        #if (L_30kpc == 0) & (L_70kpc == 0) & (L_100kpc == 0): print(i)
        
        for j, L in enumerate([L_5kpc, L_7kpc, L_10kpc, L_20kpc, L_30kpc, L_70kpc, L_100kpc]):
            normL = norm(L)
            normV = norm(delta_V, axis=1)
            cos_theta = np.dot(delta_V, L)/(normV*normL)
            e_gas_wthermal = MassPart_sorted * ((normV*cos_theta)**2 + (1/3.)*sigmaP**2)
            #e_gas_wothermal = MassPart_sorted * (normV*cos_theta)**2 
            
            if j == 0:
                Aps = np.arange(1,rmax+1).astype(int)
                argmin  = np.zeros(len(Aps), dtype=int)
                Npart = np.arange(1,LenPart+1).astype(int)
                for k, ap in enumerate(Aps):
                    p = np.argmin(abs(r - ap))  # TODO More efficient way?
                    argmin[k] = p if r[p] <= ap else p-1
            
            sigma_wth  = np.sqrt(np.cumsum(e_gas_wthermal)/np.cumsum(MassPart_sorted))
            #sigma_woth = np.cumsum(e_gas_wothermal)/np. cumsum(MassPart_sorted)
            
            for k in range(len(Aps)):
                sigma[j,i,k] = sigma_wth[argmin[k]]
                if j == 0: NumPart[i,k] = Npart[argmin[k]]
    
    return sigma, NumPart
    #if return_noth: return sigma, sigma_nth, NumPart
    #else: return sigma, NumPart
        
def Stars(data, Tree, rmax=50):
    keys = ['h','a', 'snap', 'fend', 'Lbox','L', 'NP', 'Base', 'BasePart']
    h,a,snap,fend,Lbox,L,NP,Base,BasePart = [data['params'].get(key) for key in keys]
    
    Pos_cop      = data['Pos_cop']
    Vcom         = data['Vcom']
    SpinStar     = data['SpinStar']   #[pkpc km/s]
    gns          = data['gn']
    sgns         = data['sgn']
    
    LenGal       = len(Pos_cop)
    sigma        = np.zeros((LenGal,rmax), dtype=np.float64) 
    NumPart      = np.zeros((LenGal,rmax), dtype=int)
    
    # === Get Star Data ===
    with h5.File(BasePart + '%s_%s_%iMpc_%i_Star.hdf5' %(snap, fend, L, NP), 'r') as fh:
        MassPart    = fh['PartData/MassStar'][()]/h
        PosPart     = fh['PartData/PosStar'][()]/h
        VPart       = fh['PartData/VelStar'][()] * np.sqrt(a)
        gnStar      = fh['PartData/GrpNum_Star'][()]
        sgnStar     = fh['PartData/SubNum_Star'][()]
    
    for i, idx in enumerate(Tree.query_ball_point(Pos_cop, (rmax/1000. + 0.005)/a)):
        if idx is None: continue
        gns_gal  = gnStar[idx]
        sgns_gal = sgnStar[idx]
        masksgn = (gns_gal == gns[i]) & (sgns_gal == sgns[i])
        LenPart = len(sgns_gal[masksgn])
        if LenPart == 0: continue
        
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
        normL = norm(SpinStar[i])
        normV = norm(delta_V, axis=1)
        cos_theta = np.dot(delta_V, SpinStar[i])/(normV*normL)
        e_gas = MassPart_sorted * (normV*cos_theta)**2
            
        Aps = np.arange(1,rmax+1).astype(int)
        argmin  = np.zeros(len(Aps), dtype=int)
        Npart = np.arange(1,LenPart+1).astype(int)
        for k, ap in enumerate(Aps):
            p = np.argmin(abs(r - ap)) 
            argmin[k] = p if r[p] <= ap else p-1
            
        sigma_wth  = np.sqrt(np.cumsum(e_gas)/np.cumsum(MassPart_sorted))

        for k in range(len(Aps)):
            sigma[i,k] = sigma_wth[argmin[k]]
            NumPart[i,k] = Npart[argmin[k]]

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
    with h5.File(BasePart + '%s_%s_%iMpc_%i_DM.hdf5' %(snap, fend, L, NP), 'r') as fh:
        PosPart    = fh['PartData/PosDM'][()]/h 
        VPart      = fh['PartData/VelDM'][()] * np.sqrt(a)
        sgnDM      = fh['PartData/SubNum_DM'][()]
        MassPart   = fh['Header/PartMassDM'][()]/h 
    #######################
    for i, idx in enumerate(Tree.query_ball_point(Pos_cop, (rmax/1000. + 0.005)/a)):
        if idx is None: continue
        sgns_gal = sgnDM[idx]
        masksgn = sgns_gal == sgns[i]
        LenPart = len(sgns_gal[masksgn])
        if len(sgns_gal[masksgn]) == 0: continue
        
        PosPart_gal  = PosPart[idx][masksgn]
        VPart_gal    = VPart[idx][masksgn]
        LenPart      = len(PosPart_gal)
        
        centre = Pos_cop[i]
        r = distance(PosPart_gal, centre, Lbox) * a * 1e3 
        tt = np.argsort(r)
        r = r[tt]
        if r[0] > rmax: continue
        
        delta_V = VPart_gal[tt] - Vcom[i]
        normL     = norm(SpinStar[i])
        normV     = norm(delta_V, axis=1)
        cos_theta = np.dot(delta_V, SpinStar[i])/(normV*normL)
        e_gas    = MassPart *(normV*cos_theta)**2
                                       
        Aps = np.arange(1,rmax+1).astype(int)
        argmin  = np.zeros(len(Aps), dtype=int)
        Npart = np.arange(1,LenPart+1).astype(int)
        for k, ap in enumerate(Aps):
            p = np.argmin(abs(r - ap)) 
            argmin[k] = p if r[p] <= ap else p-1
            
        sigma_wth  = np.sqrt(np.cumsum(e_gas)/np.cumsum(np.ones(LenPart)*MassPart))
        for k in range(len(Aps)):
            sigma[i,k] = sigma_wth[argmin[k]]
            NumPart[i,k] = Npart[argmin[k]]
                
    return sigma, NumPart

