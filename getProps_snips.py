from scipy.interpolate import UnivariateSpline 
from scipy.spatial import cKDTree as KDTree
from tools import ScaleFactor, RotMatrix
import multiprocessing as mp
from Snipshots import get_zstr
import pandas as pd
import numpy as np
import h5py as h5
import pickle
import time
import sys
import Periodicity
from pyread_eagle import *

h = 0.6777
Base      = '/fred/oz009/ejimenez/'
BaseGal   = Base + 'Galaxies/L0100N1504/snipshots/' 
BasePart  = Base + 'L0100N1504/snipshots/'
BasePart2 =  '/fred/oz009/clagos/EAGLE/L0100N1504/data/'

#For each node divide the simulation in 6x6x6 subvolumes and read the particle data.
#It means a lower number of reads hence less IOPS...

##### AUXILIARY FUNCTIONS ##########

def get_cylindrical(pos):
    rho      = np.sqrt(np.square(pos[:,0]) + np.square(pos[:,1]))
    varphi   = np.arctan2(pos[:,1], pos[:,0])
    z        = pos[:,2]    
    PosCyl = np.array([rho, varphi, z]).T
    return(PosCyl) 

def getRmatrix(PosGas, PosStar, VelGas, VelStar, MassGas, MassStar, isCold, aStar, Mrad, Vcom_dict, agal,gn):
    # Compute the angular momentum of cold gas + young stars 
    # Postions and velocities are in the frame of the galax
    
    r_arr = Vcom_dict['r_arr']
    splVx = UnivariateSpline(r_arr, Vcom_dict['Vx'], s=0, k=1)
    splVy = UnivariateSpline(r_arr, Vcom_dict['Vy'], s=0, k=1)
    splVz = UnivariateSpline(r_arr, Vcom_dict['Vz'], s=0, k=1)
    
    Ncold = len(np.where(isCold)[0])
    maskA = aStar > ScaleFactor(agal, 100)  
    
    if Ncold >= 100:
        rmax = Mrad['cold_gas']['r50']
        Vcom_rot = np.array([splVx(rmax), splVy(rmax), splVz(rmax)])
        
        PosCold  = PosGas[isCold]
        MassCold = MassGas[isCold]
        VelCold  = VelGas[isCold] - Vcom_rot
        
        PosYoung  = PosStar[maskA]
        MassYoung = MassStar[maskA]
        VelYoung  = VelStar[maskA] - Vcom_rot
        
        rCold  = np.linalg.norm(PosCold, axis=1) 
        rYoung = np.linalg.norm(PosYoung, axis=1)
        
        rComb    = np.concatenate((rCold, rYoung))
        PosComb  = np.concatenate((PosCold, PosYoung))[rComb < rmax]
        MassComb = np.concatenate((MassCold, MassYoung))[rComb < rmax]
        VelComb  = np.concatenate((VelCold, VelYoung))[rComb < rmax]
        LenComb = len(PosComb)
        
    else:
        rmax     = Mrad['stars']['r50']
        Vcom_rot = np.array([splVx(rmax), splVy(rmax), splVz(rmax)])
        
        rComb    = np.linalg.norm(PosStar, axis=1) 
        PosComb  = PosStar[rComb < rmax]
        MassComb = MassStar[rComb < rmax]
        VelComb  = VelStar[rComb < rmax] - Vcom_rot
        LenComb  = len(PosComb)
        
    j1 = np.sum(MassComb[:, np.newaxis] * np.cross(PosComb, VelComb), axis=0)/np.sum(MassComb)
    R = RotMatrix(j1, np.array([0,0,1]))
    
    pos_new = np.zeros((LenComb,3))
    for i, p in enumerate(PosComb): pos_new[i] = R.dot(p) 

    maskz  = abs(pos_new[:,2]) < 3
    LenCyl = len(MassComb[maskz])
    
    if LenCyl >= 50:
        vel_new = np.zeros((LenComb,3))
        for i, v in enumerate(VelComb): vel_new[i] = R.dot(v) 
        
        j2 = np.sum(MassComb[maskz, np.newaxis] * np.cross(pos_new[maskz], vel_new[maskz]), axis=0)/np.sum(MassComb[maskz])
       
        Rf = RotMatrix(j2, np.array([0,0,1])) @ RotMatrix(j1, np.array([0,0,1]))
        if np.any(np.isnan(Rf)): return (R, j1, j2)
        else: return (Rf, j1, j2)
    
    else: 
        j2 = np.array([-1,-1,-1])
        return (R, j1, j2)
    
def recentre(PosGas, PosStar, COP, Lbox):
    PosGas  = np.mod(PosGas - COP  + 0.5*Lbox, Lbox) + COP -0.5*Lbox
    PosStar = np.mod(PosStar - COP  + 0.5*Lbox, Lbox) + COP -0.5*Lbox
    PosGas  -= COP
    PosStar -= COP
    return(PosGas, PosStar)

def getR50(PosCold, MassCold):
    rsph = np.linalg.norm(PosCold, axis=1)
    tt = np.argsort(rsph)
    rc = rsph[tt]
    mc = np.cumsum(MassCold[tt])
    R50 = rc[np.argmin(abs(mc - 0.5*mc[-1]))] 
    return(R50)

def nbsearch(pos, nb, tree):
    d, idx = tree.query(pos, k=nb)
    hsml = d[:,nb-1]
    return hsml

def getCube(sindex, Ns):
    i = int(sindex%(Ns*Ns)/Ns)
    j = (sindex%(Ns*Ns))%Ns
    k = int(sindex/(Ns*Ns))
    return (i,j,k)
    
###### PROP FUNCTIONS ##########

def gethz(pos, mass, r50):
    rho = np.sqrt(pos[:,0]**2  + pos[:,1]**2)     
    z   = np.abs(pos[:,2])
    
    mask = rho < 3*r50
    z    = z[mask]
    mass = mass[mask]
    LenP = len(mass)
    
    tt   = np.argsort(z)
    z    = z[tt]
    mass = mass[tt]
    Mcum = np.cumsum(mass)
    Mtot = np.sum(mass)

    # Mcum and z monotonic functions, hence can simply use the inverse
    Mspl = UnivariateSpline(Mcum, z, s=0, k=1)
    M50  = 0.50*Mtot
    M75  = 0.75*Mtot
    M90  = 0.90*Mtot
    z50 = Mspl(M50) if LenP >= 50 else 0 
    z75 = Mspl(M75) if LenP >= 50 else 0
    z90 = Mspl(M90) if LenP >= 50 else 0
    return {'z50': z50 , 'z75': z75, 'z90': z90}

def getMassRad(PosCold, PosStar, MassCold, MassStar):
    r         = np.linalg.norm(PosStar, axis=1) 
    isort     = np.argsort(r) 
    rsort     = r[isort]
    Mscum     = np.cumsum(MassStar[isort])

    spl_stars = UnivariateSpline(Mscum, rsort, s=0, k=1)
    Mtot      = np.sum(MassStar)
    
    r25_stars = spl_stars(0.25*Mtot)
    r50_stars = spl_stars(0.50*Mtot) 
    r90_stars = spl_stars(0.90*Mtot) 
    rstars = {'r25': r25_stars, 'r50': r50_stars, 'r90': r90_stars}
    
    if len(PosCold) >= 100:
        r        = np.linalg.norm(PosCold, axis=1) 
        isort    = np.argsort(r) 
        rsort    = r[isort]
        Mccum    = np.cumsum(MassCold[isort])
        
        spl_cold = UnivariateSpline(Mccum, rsort, s=0, k=1)
        Mtot     = np.sum(MassCold)
        
        r25_cold = spl_cold(0.25*Mtot)
        r50_cold = spl_cold(0.50*Mtot) 
        r90_cold = spl_cold(0.90*Mtot) 
        rcold = {'r25': r25_cold, 'r50': r50_cold, 'r90': r90_cold}
    else : rcold = -1
    
    MassRad = {'stars': rstars, 'cold_gas': rcold}
    return MassRad
            
def getkappaRot(pos, mass, vel, r50):
    rho  = np.sqrt(pos[:,0]**2 + pos[:,1]**2)
    rsph = np.linalg.norm(pos, axis=1)
    mask = (rsph < 2*r50) 
    
    rho  = rho[mask]
    pos  = pos[mask]
    vel  = vel[mask] 
    mass = mass[mask]
    LenP = len(rho)

    K = 0.5 * np.sum(mass *  np.sum(np.square(vel), axis=1))
    
    # Rotation axis definition
    Ltot = np.sum(mass[:, np.newaxis] * np.cross(pos, vel), axis=0)
    Lhat = Ltot/np.linalg.norm(Ltot)
    
    L_i = mass[:, np.newaxis] * np.cross(pos, vel)
    Lz_i = np.dot(L_i, np.array([0,0,1]))
    isCo = Lz_i > 0
    
    Krot = 0.5 * np.sum(mass * (Lz_i/(mass*rho))**2)/K if LenP >= 50 else 0
    Kco  = 0.5 * np.sum(mass[isCo]* (Lz_i[isCo]/(mass[isCo]*rho[isCo]))**2)/K if LenP >= 50 else 0
    
    return {'Krot': Krot, 'Kco': Kco}

def getRadii(pos, mass, z90):    
    rdict = {'R25':0, 'R50': 0, 'R90': 0} 
    
    if z90 == 0: return rdict
    
    hz = z90 if (z90 > 0.5) & (z90 < 3) else (0.5 if z90 < 0.5 else 3)
    
    rho = np.sqrt(pos[:,0]**2  + pos[:,1]**2)     
    maskCyl = abs(pos[:,2]) < hz
    LenP    = len(rho[maskCyl])
    
    if LenP < 50: return rdict
        
    rho  = rho[maskCyl]
    mass = mass[maskCyl]
    
    tt   = np.argsort(rho)
    rho  = rho[tt]
    mass = mass[tt]
    
    Mtot = np.sum(mass)
    M25 = 0.25 * Mtot
    M50 = 0.50 * Mtot
    M90 = 0.90 * Mtot
    
    Mcum = np.cumsum(mass)
    Mspl = UnivariateSpline(Mcum, rho, s=0, k=1)
    
    rdict['R25'] = Mspl(M25)
    rdict['R50'] = Mspl(M50)
    rdict['R90'] = Mspl(M90)
    
    return rdict
    
# Still need to add spherical apertures
def getVelDisp(pos, vel, mass, R50, z90, sigmaP=None)
    radii  = np.arange(1,101)
    sigma = np.zeros(100)
    Npart = np.zeros(100, dtype=np.int32)
    
    if (z90 == 0) | (R50 == 0): return {'profile': sigma, 'Npart': Npart, 'sigma_3R50': 0, 'Npart_3R50': 0} 
    
    hz = z90 if (z90 > 0.5) & (z90 < 3) else (0.5 if z90 < 0.5 else 3)
    rho = np.sqrt(pos[:,0]**2  + pos[:,1]**2)     
    maskCyl = abs(pos[:,2]) < hz

    coldg = True if isinstance(sigmaP, np.ndarray) else False
  
    rho   = rho[maskCyl]
    vel   = vel[maskCyl]
    mass  = mass[maskCyl]
    
    isort = np.argsort(rho)
    rho   = rho[isort]
    vel   = vel[isort]
    mass  = mass[isort]
        
    if coldg:
        sigmaP = sigmaP[maskCyl]
        sigmaP = sigmaP[isort]
    
    LenP  = len(rho)
    Ncum  = np.arange(LenP+1)
    rho   = np.append([0], rho)
    
    if coldg:
        sigma_cum = np.append([0], np.sqrt(np.cumsum(mass*(vel[:,2]**2 + sigmaP**2))/np.cumsum(mass)))
    else:
        sigma_cum = np.append([0], np.sqrt(np.cumsum(mass*vel[:,2]**2)/np.cumsum(mass)))
        
    
    spl_sigma  = UnivariateSpline(rho, sigma_cum, s=0, k=1)
    spl_Npart  = UnivariateSpline(rho, Ncum, s=0, k=1)
    
    
    rho_max    = rho[-1]
    idx        = np.where(radii < rho_max)[0]    
    sigma[idx] = spl_sigma(radii[idx])
    Npart[idx] = np.floor(spl_Npart(radii[idx]))
    
    idxOut        = np.where(radii > rho_max)[0] 
    LenOut        = len(idxOut)
    sigma[idxOut] = np.ones(LenOut)*spl_sigma(rho_max)
    Npart[idxOut] = np.ones(LenOut)*int(spl_Npart(rho_max))
    
    Rev = 3*R50 
    sigma_3R50 = spl_sigma(Rev) if Rev < rho_max else spl_sigma(rho_max)
    Npart_3R50 = int(spl_Npart(Rev)) if Rev < rho_max else int(spl_Npart(rho_max))
    
    return {'profile': sigma, 'Npart': Npart, 'sigma_3R50': sigma_3R50, 'Npart_3R50': Npart_3R50} 
    
def getMasses(pos, mass, R50, z90):
    Mtot  = np.sum(mass)
    if (R50 == 0) | (z90 == 0): return {'Mtot': Mtot*1e10, 'Mdisk': 0}
    
    rho     = np.sqrt(pos[:,0]**2  + pos[:,1]**2)
    hz      = z90 if (z90 > 0.5) & (z90 < 3) else (0.5 if z90 < 0.5 else 3) 
    maskCyl = (rho < 3*R50) & (abs(pos[:,2]) < hz)

    LenP = len(mass[maskCyl])
    Mdisk = np.sum(mass[maskCyl]) if LenP >= 50 else 0
    
    return {'Mtot': Mtot*1e10, 'Mdisk': Mdisk*1e10}

def get_GSmisl(PosCold, VelCold, MassCold, PosStar, VelStar, MassStar, r50s):
    rgas  = np.linalg.norm(PosCold, axis=1)
    rstar = np.linalg.norm(PosStar, axis=1)

    maskg = rgas < r50s
    masks = rstar < r50s
    
    jgas  = np.sum(MassCold[maskg, np.newaxis] * np.cross(PosCold[maskg], VelCold[maskg]), axis=0)/np.sum(MassCold)
    jstar = np.sum(MassStar[masks, np.newaxis] * np.cross(PosStar[masks], VelStar[masks]), axis=0)/np.sum(MassStar)
    
    costheta = np.dot(jgas, jstar)/(np.linalg.norm(jgas) * np.linalg.norm(jstar))
    return costheta
    
##### MAIN FUNCTIONS #####

def LoadData(snip):
    global a, Lbox, LenGal, step, LL, Lsub
    global gns, COP, Rs
    global ix_arr, iy_arr, iz_arr, sindex_arr
    global eagle_data, UnitDensity, UnitPressure, gamma
    global Vx, Vy, Vz, Vcom_Rs, r_arr
 
    with h5.File(BaseGal + '%s_%s_100Mpc_1504_galaxies.hdf5' %(str(snip), get_zstr(snip)), 'r') as f:
        a         = f['Header/a'][()]
        Lbox      = f['Header/BoxSize'][()] # comoving size in EAGLE units
        gns       = f['SubHaloData/MainHaloID'][()]
        sgns      = f['SubHaloData/SubHaloID'][()]
        COP       = f['SubHaloData/Pos_cop'][()] # EAGLE units
        LogMstell = f['StarData/TotalMstell'][()]
    
        mask      = (LogMstell > 9.0) & (sgns == 0) 
        gns       = gns[mask]
        COP       = COP[mask]

        R200       = f['FOF/R200'][()] # EAGLE units
        MainHaloID = f['FOF/MainHaloID'][()]
        
        xy, xidx, yidx = np.intersect1d(gns, MainHaloID, return_indices=True)
        gns = gns[xidx]
        COP = COP[xidx]
        Rs = 0.2*R200[xidx]

    # Velocities in physical
    with h5.File(BasePart + 'Vcom/%s_%s_Vcom.hdf5' %(str(snip), get_zstr(snip)), 'r') as f:
        mask    = f['sgns'][()] == 0
        gns_v   = f['gns'][mask]
        Vx      = f['Vcom_x'][()][mask]
        Vy      = f['Vcom_y'][()][mask]
        Vz      = f['Vcom_z'][()][mask]
        Vcom_Rs = f['Vcom_Rs'][()][mask]
        r_arr   = f['r_arr'][()]
        
    xy, xidx, yidx = np.intersect1d(gns, gns_v, return_indices=True)
    gns     = gns[xidx]
    COP     = COP[xidx]
    Rs      = Rs[xidx]
    Vx      = Vx[yidx]
    Vy      = Vy[yidx]
    Vz      = Vz[yidx]
    Vcom_Rs = Vcom_Rs[yidx]
                
    LL     = Lbox/h # h-factor corrected comoving size
    Lsub   = Lbox/Ns # size of subvolume in EAGLE units
        
    ix_arr = (COP[:,0]/Lsub).astype(int)
    iy_arr = (COP[:,1]/Lsub).astype(int)
    iz_arr = (COP[:,2]/Lsub).astype(int)
    sindex_arr = Ns*ix_arr + iy_arr + Ns*Ns*iz_arr
    
    BaseDir = BasePart2 + 'particledata_snip_%s_%s/' %(str(snip), get_zstr(snip)) 
    fn = BaseDir + "eagle_subfind_snip_particles_%s_%s.0.hdf5" %(str(snip), get_zstr(snip))
    eagle_data = EagleSnapshot(fn)
    
    with h5.File(fn, 'r') as f:
        UnitPressure = f['Units'].attrs['UnitPressure_in_cgs']
        UnitDensity  = f['Units'].attrs['UnitDensity_in_cgs']
        gamma        = f['Constants'].attrs['GAMMA']
        
    LenGal = len(gns)
    step   = int(LenGal/10)
    
def getData(sindex):
    idx = np.where(sindex_arr == sindex)[0] # Select all galaxies in this region
    if len(idx) == 0: return -1

    gns_box = gns[idx]
    COP_box = COP[idx]/h #h-factor corrrected COP
    Rs_box  = Rs[idx]/h  #h-factor corrrected Rs=0.2R200  
    Vx_box  = Vx[idx]
    Vy_box  = Vy[idx]
    Vz_box  = Vz[idx]
    Vcom_Rs_box = Vcom_Rs[idx]
    
    
    i,j,k = getCube(sindex, Ns)
    xmin, xmax = i*Lsub, (i+1)*Lsub
    ymin, ymax = j*Lsub, (j+1)*Lsub
    zmin, zmax = k*Lsub, (k+1)*Lsub
    MainBox = np.array([xmin, xmax, ymin, ymax, zmin, zmax])
    
    boxes  = Periodicity.extent(1, MainBox, Lbox, sindex, Ns)
    for region in boxes: eagle_data.select_region(*region)
    
    # Gas Data
    gnsGas  = eagle_data.read_dataset(0, 'GroupNumber')
    sgnsGas = eagle_data.read_dataset(0, 'SubGroupNumber')    
    PosGas  = eagle_data.read_dataset(0, 'Coordinates')/h 
    VelGas  = eagle_data.read_dataset(0, 'Velocity') * np.sqrt(a)
    TempGas = eagle_data.read_dataset(0, 'Temperature')/1e4
    MassGas = eagle_data.read_dataset(0, 'Mass')/h
    DenGas  = eagle_data.read_dataset(0, 'Density')
    Entropy = eagle_data.read_dataset(0, 'Entropy')
    
    DenGas  = DenGas * (h**2.0) * (a**(-3.0))  * UnitDensity
    Entropy  = Entropy * (h**(2.-2.*gamma)) * UnitPressure * UnitDensity**(-1.0*gamma)
    Pressure = Entropy*DenGas**gamma
    sigmaP = np.sqrt(Pressure/DenGas) * 1e-5 * (1/np.sqrt(3)) # [km/s]
    del Entropy, Pressure
    isCold = (TempGas < 1.0) | (np.log10(DenGas) > -24)
    
    #Star Data
    gnsStar  = eagle_data.read_dataset(4, 'GroupNumber')
    sgnsStar = eagle_data.read_dataset(4, 'SubGroupNumber')
    PosStar  = eagle_data.read_dataset(4, 'Coordinates')/h 
    VelStar  = eagle_data.read_dataset(4, 'Velocity') * np.sqrt(a)
    MassStar = eagle_data.read_dataset(4, 'Mass')/h
    aStar    = eagle_data.read_dataset(4, 'StellarFormationTime')
    
    eagle_data.clear_selection()
    

    GasData  = {'gnsGas': gnsGas, 'sgnsGas': sgnsGas, 'PosGas': PosGas, 'MassGas': MassGas, 'isCold': isCold, 
              'sigmaP': sigmaP, 'VelGas': VelGas}
    
    StarData = {'gnsStar': gnsStar, 'sgnsStar': sgnsStar, 'PosStar': PosStar, 'VelStar': VelStar, 
                'MassStar': MassStar, 'aStar': aStar}
    
    GalData  = {'gns_box': gns_box, 'COP_box': COP_box, 'Rs_box': Rs_box, 
               'Vx_box': Vx_box, 'Vy_box': Vy_box, 'Vz_box': Vz_box, 'Vcom_Rs_box': Vcom_Rs_box}

    return {'GasData': GasData, 'StarData': StarData, 'GalData': GalData}
    
def getProps(sindex):
    data = getData(sindex)
    if data == -1: return -1

    gnsGas   = data['GasData']['gnsGas']
    sgnsGas  = data['GasData']['sgnsGas']
    PosGas   = data['GasData']['PosGas']
    MassGas  = data['GasData']['MassGas']
    isCold   = data['GasData']['isCold']
    sigmaP   = data['GasData']['sigmaP']
    VelGas   = data['GasData']['VelGas']
    
    gnsStar  = data['StarData']['gnsStar']
    sgnsStar = data['StarData']['sgnsStar']
    PosStar  = data['StarData']['PosStar']
    VelStar  = data['StarData']['VelStar']
    MassStar = data['StarData']['MassStar']
    aStar    = data['StarData']['aStar']
    
    gns_box = data['GalData']['gns_box']
    COP_box = data['GalData']['COP_box']
    Rs_box  = data['GalData']['Rs_box']
    Vx_box  = data['GalData']['Vx_box']
    Vy_box  = data['GalData']['Vy_box']
    Vz_box  = data['GalData']['Vz_box']
    Vcom_Rs_box  = data['GalData']['Vcom_Rs_box']
    
    LenGal_box = len(gns_box)
    del data
    
    TreeGas  = KDTree(PosGas, leafsize=10, boxsize=LL)
    TreeStar = KDTree(PosStar, leafsize=10, boxsize=LL)
    
    gnDict   = {}
    print(LenGal_box)
    
    for i, gn, COP, Rs in zip(range(LenGal_box), gns_box, COP_box, Rs_box):
        idx          = TreeGas.query_ball_point(COP, Rs)
        maskGas      = (gnsGas[idx] == gn) & (sgnsGas[idx] == 0)
        PosGas_gal   = PosGas[idx][maskGas]
        MassGas_gal  = MassGas[idx][maskGas]
        VelGas_gal   = VelGas[idx][maskGas]
        sigmaP_gal   = sigmaP[idx][maskGas]
        isCold_gal   = isCold[idx][maskGas]
        
        idx          = TreeStar.query_ball_point(COP, Rs)
        maskStar     = (gnsStar[idx] == gn) & (sgnsStar[idx] == 0)
        PosStar_gal  = PosStar[idx][maskStar]
        VelStar_gal  = VelStar[idx][maskStar]
        MassStar_gal = MassStar[idx][maskStar]
        aStar_gal    = aStar[idx][maskStar]
        
        LenStar = len(PosStar_gal)
        LenGas  = len(PosGas_gal)
        LenCold = len(PosGas_gal[isCold_gal])        
        
        # Reframe coordinates 
        PosGas_gal, PosStar_gal = recentre(PosGas_gal, PosStar_gal, COP, LL)        
        PosGas_gal  *= (a*1e3)
        PosStar_gal *= (a*1e3)
        
        # Reframe velocities
        Vcom        = Vcom_Rs_box[i]
        VelGas_gal  -= Vcom
        VelStar_gal -= Vcom
        Vx_arr      = Vx_box[i] - Vcom[0]
        Vy_arr      = Vy_box[i] - Vcom[1]
        Vz_arr      = Vz_box[i] - Vcom[2]
        Vcom_dict   = {'r_arr': r_arr, 'Vx': Vx_arr, 'Vy': Vy_arr, 'Vz': Vz_arr}
        
        #Get r50 for stars and cold gas
        MassRad = getMassRad(PosGas_gal[isCold_gal], PosStar_gal, MassGas_gal[isCold_gal], MassStar_gal)
        
        ## Align the galaxy with the z axis
        R,j1,j2 = getRmatrix(PosGas_gal, PosStar_gal, VelGas_gal, VelStar_gal, MassGas_gal, MassStar_gal, isCold_gal, aStar_gal, MassRad, Vcom_dict, a,gn)

        # Rotate coordinates and velocities
        for k, p, v in zip(range(LenGas), PosGas_gal, VelGas_gal):  
            PosGas_gal[k] = R.dot(p) 
            VelGas_gal[k] = R.dot(v) 
                
        for k, p, v in zip(range(LenStar), PosStar_gal, VelStar_gal):  
            PosStar_gal[k] = R.dot(p) 
            VelStar_gal[k] = R.dot(v) 
            
        GalProp = {'Vcom': Vcom, 'j1': j1, 'j2': j2}  
        
        ##### Compute stellar and cold gas mass properties  #####
        r50_star    = MassRad['stars']['r50']
        hz_stars    = gethz(PosStar_gal, MassStar_gal, r50_star)
        Kco_stars   = getkappaRot(PosStar_gal, MassStar_gal, VelStar_gal, r50_star)
        Radii_stars = getRadii(PosStar_gal, MassStar_gal, hz_stars['z90'])
        sigma_stars = getVelDisp(PosStar_gal, VelStar_gal, MassStar_gal, Radii_stars['R50'], hz_stars['z90'])
        Mass_stars  = getMasses(PosStar_gal, MassStar_gal, Radii_stars['R50'], hz_stars['z90'])
        
        StellarProp = {'rsph': MassRad['stars'], 'hz': hz_stars, 'Krot': Kco_stars,
                                    'Radii': Radii_stars, 'sigma': sigma_stars, 'Mass': Mass_stars, 
                                    'LenStar': LenStar}
        
        if LenCold >= 100:
            PosCold    = PosGas_gal[isCold_gal]
            VelCold    = VelGas_gal[isCold_gal]
            MassCold   = MassGas_gal[isCold_gal]
            sigmaPCold = sigmaP_gal[isCold_gal]
            
            r50_cold   = MassRad['cold_gas']['r50']
            hz_cold    = gethz(PosCold, MassCold, r50_cold)
            Kco_cold   = getkappaRot(PosCold, MassCold, VelCold, r50_cold)
            Radii_cold = getRadii(PosCold, MassCold, hz_cold['z90'])
            sigma_cold = getVelDisp(PosCold, VelCold, MassCold, Radii_cold['R50'], hz_cold['z90'],sigmaP=sigmaPCold)
            Mass_cold  = getMasses(PosCold, MassCold, Radii_cold['R50'], hz_cold['z90'])
            
            ColdGasProp = {'rsph': MassRad['cold_gas'], 'hz': hz_cold, 'Krot': Kco_cold,
                                    'Radii': Radii_cold, 'sigma': sigma_cold, 'Mass': Mass_cold, 
                                    'LenCold': LenCold}
        
        else: ColdGasProp = {'LenCold': LenCold}
    
        gnDict['gn_%i' %gn] = {'GalProp': GalProp, 'StellarProp': StellarProp, 'ColdGasProp': ColdGasProp}
    
    print('sindex:', sindex, '-- Finished: ',round(time.time()-start_time,2), 'sec')
    
    return {'Props': gnDict, 'LenGal_box': LenGal_box, 'gns_box': gns_box}

if __name__ == '__main__':    
    ncpu = 32
    Ns   = 6
    
    #snip = int(sys.argv[1])
    snips = np.concatenate((np.arange(202,313,2), np.arange(316,405,2)))[::-1]
    #snips = [362,358]

    for snip in snips:
        start_time = time.time()
        print('snip:', snip)
        
        LoadData(snip)    
        Nboxes = int(Ns**3)

        with mp.Pool(ncpu-1) as pool: out = pool.map(getProps, range(Nboxes), chunksize=1)
        
        gns          = np.zeros(LenGal)
        j1           = np.zeros((LenGal,3))
        j2           = np.zeros((LenGal,3))
        Vcom         = np.zeros((LenGal,3))
        
        r_stars      = np.zeros((LenGal,3))
        hz_stars     = np.zeros((LenGal,3))
        Kco_stars    = np.zeros((LenGal,2))
        R_stars      = np.zeros((LenGal,3))
        s_stars      = np.zeros((LenGal,100))
        N_stars      = np.zeros((LenGal,100))
        s_stars_3R50 = np.zeros(LenGal)
        N_stars_3R50 = np.zeros(LenGal)
        M_stars      = np.zeros((LenGal,2))
        Ntot_stars   = np.zeros(LenGal, dtype=np.int32)
        
        r_cold       = np.zeros((LenGal,3))
        hz_cold      = np.zeros((LenGal,3))
        Kco_cold     = np.zeros((LenGal,2))
        R_cold       = np.zeros((LenGal,3))
        s_cold       = np.zeros((LenGal,100))
        N_cold       = np.zeros((LenGal,100))
        s_cold_3R50  = np.zeros(LenGal)
        N_cold_3R50  = np.zeros(LenGal)
        M_cold       = np.zeros((LenGal,2))
        Ntot_cold    = np.zeros(LenGal, dtype=np.int32)
        
        j = 0
        for idict in out:
            if idict == -1: continue
            gns_box = idict['gns_box']
            LenGal_box = idict['LenGal_box']
            
            for i, gn in zip(np.arange(j,j+LenGal_box), gns_box):
                gns[i]          = gn
                gn_props        = idict['Props']['gn_%i' %gn]
                
                j1[i]           = gn_props['GalProp']['j1']
                j2[i]           = gn_props['GalProp']['j2']
                Vcom[i]         = gn_props['GalProp']['Vcom']
                
                stell_prop      = gn_props['StellarProp']
                r_stars[i]      = list(stell_prop['rsph'].values())
                hz_stars[i]     = list(stell_prop['hz'].values())
                Kco_stars[i]    = list(stell_prop['Krot'].values())
                R_stars[i]      = list(stell_prop['Radii'].values())
                s_stars[i]      = stell_prop['sigma']['profile']
                N_stars[i]      = stell_prop['sigma']['Npart']
                s_stars_3R50[i] = stell_prop['sigma']['sigma_3R50']
                N_stars_3R50[i] = stell_prop['sigma']['Npart_3R50']
                M_stars[i]      = list(stell_prop['Mass'].values())
                Ntot_stars[i]   = stell_prop['LenStar']
                
                cold_prop       = gn_props['ColdGasProp']
                LenCold         = cold_prop['LenCold']
                Ntot_cold[i]    = LenCold
                
                if LenCold >= 100:
                    r_cold[i]      = list(cold_prop['rsph'].values())
                    hz_cold[i]     = list(cold_prop['hz'].values())
                    Kco_cold[i]    = list(cold_prop['Krot'].values())
                    R_cold[i]      = list(cold_prop['Radii'].values())
                    s_cold[i]      = cold_prop['sigma']['profile']
                    N_cold[i]      = cold_prop['sigma']['Npart']
                    s_cold_3R50[i] = cold_prop['sigma']['sigma_3R50']
                    N_cold_3R50[i] = cold_prop['sigma']['Npart_3R50']
                    M_cold[i]      = list(cold_prop['Mass'].values())

            j += LenGal_box    
            
        isort = np.argsort(gns)
        
        with h5.File(BaseGal + 'GalProps/%s_%s_100Mpc_1504_Props.hdf5' %(str(snip), get_zstr(snip)), 'w') as f:
            f.create_dataset('gns',              data = gns[isort].astype(int))
            f.create_dataset('Nstars',           data = Ntot_stars[isort].astype(int))
            f.create_dataset('Ncold',            data = Ntot_cold[isort].astype(int))
            f.create_dataset('j1_rot',           data = j1[isort].astype(np.float32))
            f.create_dataset('j2_rot',           data = j2[isort].astype(np.float32))
            f.create_dataset('Vcom_Rs',          data = Vcom[isort].astype(np.float32))
            
            grp1 = f.create_group('StellarProps')
            grp2 = f.create_group('ColdGasProps')
        
            grp1.create_dataset('r25',           data = r_stars[:,0][isort].astype(np.float32))
            grp1.create_dataset('r50',           data = r_stars[:,1][isort].astype(np.float32))
            grp1.create_dataset('r90',           data = r_stars[:,2][isort].astype(np.float32))
            grp1.create_dataset('z50',           data = hz_stars[:,0][isort].astype(np.float32))
            grp1.create_dataset('z75',           data = hz_stars[:,1][isort].astype(np.float32))
            grp1.create_dataset('z90',           data = hz_stars[:,2][isort].astype(np.float32))
            grp1.create_dataset('Krot',          data = Kco_stars[:,0][isort].astype(np.float32))
            grp1.create_dataset('Kco',           data = Kco_stars[:,1][isort].astype(np.float32))
            grp1.create_dataset('R25',           data = R_stars[:,0][isort].astype(np.float32))
            grp1.create_dataset('R50',           data = R_stars[:,1][isort].astype(np.float32))
            grp1.create_dataset('R90',           data = R_stars[:,2][isort].astype(np.float32))
            grp1.create_dataset('sigma_prof',    data = s_stars[isort].astype(np.float32))
            grp1.create_dataset('N_prof',        data = N_stars[isort].astype(np.int32))
            grp1.create_dataset('sigma_3R50',    data = s_stars_3R50[isort].astype(np.float32))
            grp1.create_dataset('N_3R50',        data = N_stars_3R50[isort].astype(np.int32))
            grp1.create_dataset('LogMstell',     data = np.log10(M_stars[:,0][isort].astype(np.float32) + 1e-10))
            grp1.create_dataset('LogMstellDisk', data = np.log10(M_stars[:,1][isort].astype(np.float32) + 1e-10))
            
            
            grp2.create_dataset('r25',           data = r_cold[:,0][isort].astype(np.float32))
            grp2.create_dataset('r50',           data = r_cold[:,1][isort].astype(np.float32))
            grp2.create_dataset('r90',           data = r_cold[:,2][isort].astype(np.float32))
            grp2.create_dataset('z50',           data = hz_cold[:,0][isort].astype(np.float32))
            grp2.create_dataset('z75',           data = hz_cold[:,1][isort].astype(np.float32))
            grp2.create_dataset('z90',           data = hz_cold[:,2][isort].astype(np.float32))
            grp2.create_dataset('Krot',          data = Kco_cold[:,0][isort].astype(np.float32))
            grp2.create_dataset('Kco',           data = Kco_cold[:,1][isort].astype(np.float32))
            grp2.create_dataset('R25',           data = R_cold[:,0][isort].astype(np.float32))
            grp2.create_dataset('R50',           data = R_cold[:,1][isort].astype(np.float32))
            grp2.create_dataset('R90',           data = R_cold[:,2][isort].astype(np.float32))
            grp2.create_dataset('sigma_prof',    data = s_cold[isort].astype(np.float32))
            grp2.create_dataset('N_prof',        data = N_cold[isort].astype(np.int32))
            grp2.create_dataset('sigma_3R50',    data = s_cold_3R50[isort].astype(np.float32))
            grp2.create_dataset('N_3R50',        data = N_cold_3R50[isort].astype(np.int32))
            grp2.create_dataset('LogMcold',     data = np.log10(M_cold[:,0][isort].astype(np.float32) + 1e-10))
            grp2.create_dataset('LogMcoldDisk', data = np.log10(M_cold[:,1][isort].astype(np.float32) + 1e-10))
                        
        print('Done', round(time.time()-start_time,2))
            

    
