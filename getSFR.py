import numpy as np
import h5py as h5
from scipy.integrate import quad

h = 0.6777
OmegaL0 = 0.693
OmegaM0 = 0.307
cm_Mpc = 3.085678e24
cm_km = 1e5
s_Gyr = 3.155e16
H0 = 100*h *cm_km/cm_Mpc *s_Gyr

# [tstep] = Gyr
def getSFR(data, Tree, tstep=0.1, ap=100):
    keys = ['h','a', 'snap', 'fend', 'Lbox','L', 'NP', 'Base', 'BasePart']
    h,a,snap,fend,Lbox,L,NP,Base,BasePart = [data['params'].get(key) for key in keys]
    
    Pos_cop      = data['Pos_cop'] #[cMpc]
    gns          = data['gn']
    sgns         = data['sgn']
    LenGal       = len(Pos_cop)
    SFR_av        = np.zeros(LenGal, dtype=np.float32) 
    
    # === Get Star Data === #
    fh = h5.File(BasePart + '%s_%s_%iMpc_%i_Star.hdf5' %(snap, fend, L, NP), 'r')
    star_aform    = fh['PartData/StellarFormationTime'][()]
    gnStar        = fh['PartData/GrpNum_Star'][()]
    sgnStar       = fh['PartData/SubNum_Star'][()]
    MassStar      = fh['PartData/MassStar'][()]/h * 1e10 # [Msun]
    
    fh.close()
    
    integrand = lambda z: ((1+z)*np.sqrt(OmegaL0 + OmegaM0*(1+z)**3))**-1
    red = 1/a - 1
    lbt_snap = (1/H0)*quad(integrand, 0, red)[0]
    
    for i in range(LenGal):
        Tree.search(center=Pos_cop[i], radius=(ap/1000.)/a)  #[ap] = cMpc
        idx = Tree.getIndices()
        if idx is None: continue
        gns_gal  = gnStar[idx]
        sgns_gal = sgnStar[idx]
        masksgn = (gns_gal == gns[i]) & (sgns_gal == sgns[i]) 
            
        star_aform_gal = star_aform[idx][masksgn]
        MassStar_gal   = MassStar[idx][masksgn]
        LenPart = len(star_aform_gal)
        if LenPart == 0: continue
        
        # Convert to LBT... 
        lbt = np.zeros(LenPart)
        z_arr = 1/star_aform_gal - 1
        for j, z in enumerate(z_arr): lbt[j] = quad(integrand, 0, z)[0]
        
        lbt *= H0**-1 # [lbt] = Gyr
        ages = (lbt - lbt_snap) * 1e3 # Myr
        
        mask = ages < 30
        SFR_av[i] = np.sum(MassStar_gal[mask])/(3e7) #[Msun/yr]
    
    return SFR_av
        
        
        
        

        



    
