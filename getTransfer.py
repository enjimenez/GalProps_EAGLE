import numpy as np
import h5py as h5
import matplotlib.pyplot as plt
from tools import distance
from scipy.stats import kde
from Snapshots import get_zstr
from numpy.linalg import norm 
import sys 
from scipy.integrate import quad
from matplotlib import rc
rc('text', usetex=True) 
rc('font', family='serif')
rc('axes', linewidth=1.5)


h = 0.6777
OmegaL0 = 0.693
OmegaM0 = 0.307
cm_Mpc = 3.085678e24
cm_km = 1e5
s_Gyr = 3.155e16
H0 = 100*h *cm_km/cm_Mpc *s_Gyr


Base = '/home/esteban/Documents/EAGLE/'

def GasTransfer(snap, L, NP, gn_new=17, sgn_new=0):
    fend_new = get_zstr(snap)
    box = 'L%sN%s' %(str(L).zfill(4), str(NP).zfill(4))
    BasePart = Base + 'processed_data/'
    BaseGal  =  Base + 'Galaxies/%s/REFERENCE/' %box
    
    snap_str_new = str(snap).zfill(3)
    
    ##################################   Read galaxies (Descendants) ##############################
    with h5.File(BaseGal + '%s_%s_%iMpc_%i_galaxies.hdf5' %(snap_str_new, fend_new, L, NP), 'r') as f:
        anew         = f['Header/a'][()]               # Scale factor              # h
        Lbox         = f['Header/BoxSize'][()]/h     # L [cMpc]
        gns          = f['SubHaloData/MainHaloID'][()]
        sgns         = f['SubHaloData/SubHaloID'][()]
        idx          = np.where((gns == gn_new) & (sgns == sgn_new))[0][0]
        GalID_new    = f['IDs/GalaxyID'][idx]
        TopID_new    = f['IDs/TopLeafID'][idx]
        LastID_new   = f['IDs/LastProgID'][idx]
        #mass_new    = np.log10(f['StarData/TotalMstell'][idx]/h) + 10
        COP_new      = f['SubHaloData/Pos_cop'][idx]/h
        Vcom_new     = f['SubHaloData/V_com'][idx]
        SpinStar_new = f['StarData/StarSpin'][()][idx]/h * 1e3 # [pkpc km/s]
        SpinSF_new   = f['GasData/SF/Spin'][()][idx]/h * 1e3
    
    fend_old = get_zstr(snap-1)
    snap_str_old =  str(snap-1).zfill(3)
    
    ############################### Read galaxies (Progenitors) ###################################
    with h5.File(BaseGal + '%s_%s_%iMpc_%i_galaxies.hdf5' %(snap_str_old, fend_old, L, NP), 'r') as f:
        aold         = f['Header/a'][()]               
        GalID_old    = f['IDs/GalaxyID'][()]
        idx          = np.where((GalID_old > GalID_new) & (GalID_old <= TopID_new))[0][0]
        gn_old       = f['SubHaloData/MainHaloID'][idx]
        sgn_old      = f['SubHaloData/SubHaloID'][idx]
        COP_old      = f['SubHaloData/Pos_cop'][idx]/h 
        #mass_old    = np.log10(f['StarData/TotalMstell'][idx]/h) + 10
        Vcom_old     = f['SubHaloData/V_com'][idx]
        SpinStar_old = f['StarData/StarSpin'][()][idx]/h * 1e3 # [pkpc km/s]
        SpinSF_old   = f['GasData/SF/Spin'][()][idx]/h * 1e3
        del GalID_old
    
    #print(sgn_old)
    #"""
    # Continue if the progenitor is the main progenitor was a central
    if sgn_old == 0:
        ###################################### Gas Data of progenitor #############################################
        with h5.File(BasePart + '%s_%s_%iMpc_%i_Gas.hdf5' %(snap_str_old, fend_old, L, NP), 'r') as f:
            gnsGas  = f['PartData/GrpNum_Gas'][()]
            sgnsGas = f['PartData/SubNum_Gas'][()]
            mask        = (gnsGas == gn_old) & (sgnsGas == sgn_old)
            del gnsGas, sgnsGas
            
            PosPart      = f['PartData/PosGas'][()][mask]/h  #[cMpc]
            VPart        = f['PartData/VelGas'][()][mask] * np.sqrt(aold)
            #MassPart     = f['PartData/MassGas'][mask]/h
            Density      = f['PartData/Density'][mask] 
            Entropy      = f['PartData/Entropy'][mask]
            PID_old      = f['PartData/ParticleIDs'][mask]
            UnitPressure = f['Constants/UnitPressure'][()]
            UnitDensity  = f['Constants/UnitDensity'][()]
            gamma        = f['Constants/gamma'][()]
            Density  = Density * (h**2.0) * (aold**(-3.0))  * UnitDensity
            Entropy  = Entropy * (h**(2.-2.*gamma)) * UnitPressure * UnitDensity**(-1.0*gamma)
            Pressure = Entropy*Density**gamma
            del Entropy
    
        r_gas_old = distance(PosPart, COP_old, Lbox) * aold * 1e3 # [pkpc]
        mask = r_gas_old < 5
   
        r_gas_old = r_gas_old[mask]
        VPart = VPart[mask]
        Density = Density[mask]
        Pressure = Pressure[mask]
        PID_old  = PID_old[mask]
        del PosPart
        LenGasOld = len(PID_old)
        
        LenPart = len(r_gas_old)
        delta_V = VPart - Vcom_old
        sigma_tot_old = np.zeros(LenPart, dtype=float)
        sigmaP  = np.sqrt(Pressure/Density) * 1e-5  # [km/s] 
        
        L_SpinSF = norm(SpinSF_old)
        L_SpinStar = norm(SpinStar_old)
        useSF = True if L_SpinSF != 0 else False
        
        if useSF:
            for k in range(LenPart): 
                cos_theta = np.dot(delta_V[k], SpinSF_old)/(norm(delta_V[k])*L_SpinSF)
                sigma_tot_old[k] = np.sqrt((norm(delta_V[k])*cos_theta)**2.0 + (1./3)*sigmaP[k]**2.0)
                #e_gas_old[k] = MassPart_sorted[k] * ((norm(delta_V[k])*cos_theta)**2.0 + (1./3)*sigmaP[k]**2.0)
    
        else:
            for k in range(LenPart):
                cos_theta = np.dot(delta_V[k], SpinStar_old)/(norm(delta_V[k])*L_SpinStar)
                sigma_tot_old[k] = np.sqrt((norm(delta_V[k])*cos_theta)**2.0 + (1./3)*sigmaP[k]**2.0)
    
        ################################# Gas data of descendant ##################################################
        with h5.File(BasePart + '%s_%s_%iMpc_%i_Gas.hdf5' %(snap_str_new, fend_new, L, NP), 'r') as f:
            gnsGas  = f['PartData/GrpNum_Gas'][()]
            sgnsGas = f['PartData/SubNum_Gas'][()]
            
            mask        = (gnsGas == gn_new) & (sgnsGas == sgn_new)
            del gnsGas, sgnsGas
            
            PosPart_gas  = f['PartData/PosGas'][()][mask]/h  #[cMpc]
            PID_gas_new  = f['PartData/ParticleIDs'][mask]
            Tmax         = f['PartData/MaximumTemperature'][mask]/10**7.5
            amax         = f['PartData/AExpMaximumTemperature'][mask] 
            VPart        = f['PartData/VelGas'][()][mask] * np.sqrt(anew)
            Density      = f['PartData/Density'][mask] 
            Entropy      = f['PartData/Entropy'][mask]
            Density  = Density * (h**2.0) * (anew**(-3.0))  * UnitDensity
            Entropy  = Entropy * (h**(2.-2.*gamma)) * UnitPressure * UnitDensity**(-1.0*gamma)
            Pressure = Entropy*Density**gamma
            del Entropy
        
        
        ############################ STAR DATA OF DESCENDANT #########################################################
        with h5.File(BasePart + '%s_%s_%iMpc_%i_Star.hdf5' %(snap_str_new, fend_new, L, NP), 'r') as f:
            gnsStar  = f['PartData/GrpNum_Star'][()]
            sgnsStar = f['PartData/SubNum_Star'][()]
            mask        = (gnsStar == gn_new) & (sgnsStar == sgn_new)
            del gnsStar, sgnsStar
            PosPart_star      = f['PartData/PosStar'][()][mask]/h  #[cMpc]
            PID_star_new      = f['PartData/ParticleIDs'][mask]
            
        ##########################################################################################
        r_gas_new = distance(PosPart_gas, COP_new, Lbox) * anew * 1e3 # [pkpc]
        r_star_new = distance(PosPart_star, COP_new, Lbox) * anew * 1e3 # [pkpc]
        
        LenPart = len(r_gas_new)
        delta_V = VPart - Vcom_new
        sigma_tot_new = np.zeros(LenPart, dtype=float)
        sigmaP  = np.sqrt(Pressure/Density) * 1e-5  # [km/s] 
        L_SpinSF = norm(SpinSF_new)
        L_SpinStar = norm(SpinStar_new)
        useSF = True if L_SpinSF != 0 else False
        
        if useSF:
            for k in range(LenPart): 
                cos_theta = np.dot(delta_V[k], SpinSF_new)/(norm(delta_V[k])*L_SpinSF)
                sigma_tot_new[k] = np.sqrt((norm(delta_V[k])*cos_theta)**2.0 + (1./3)*sigmaP[k]**2.0)
    
        else:
            for k in range(LenPart):
                cos_theta = np.dot(delta_V[k], SpinStar_new)/(norm(delta_V[k])*L_SpinStar)
                sigma_tot_new[k] = np.sqrt((norm(delta_V[k])*cos_theta)**2.0 + (1./3)*sigmaP[k]**2.0)
        
        maskgas = r_gas_new <= 5
        maskstar = r_star_new <= 5
        maskgas_eject = (r_gas_new > 5) & (r_gas_new < 60) 
        sigma_tot_new = sigma_tot_new[maskgas]
        
        del PosPart_gas, PosPart_star
        
        PID_gas_new_in    = PID_gas_new[maskgas]
        PID_gas_new_eject = PID_gas_new[maskgas_eject]
        PID_star_new_in   = PID_star_new[maskstar]
        
        #Tmax_eject = Tmax[maskgas_eject]
        #amax_eject = amax[maskgas_eject]
        #sigma_tot_new = sigma_tot_new[maskgas_eject]
        
        LenGas = len(PID_gas_new_in)
        LenStar = len(PID_star_new_in)
        
        PID_new = np.concatenate((PID_gas_new_in, PID_star_new_in))   # ALL gas and stars within a 5kpc aperture
        IsStar  = np.concatenate((np.zeros(LenGas), np.ones(LenStar)))
        
        xy, x_idx, y_idx = np.intersect1d(PID_old, PID_new, assume_unique=True, return_indices=True)
        xy_eject, x_idx_eject, y_idx_eject = np.intersect1d(PID_old, PID_gas_new_eject, assume_unique=True, return_indices=True)
        #del PID_gas_new_in, PID_star_new_in, PID_gas_new
    
        #integrand = lambda z: ((1+z)*np.sqrt(OmegaL0 + OmegaM0*(1+z)**3))**-1
        #znew = 1/anew - 1
        #zold = 1/aold - 1
        #lbt_snaps = (1/H0)*quad(integrand, znew, zold)[0]
        
        #z_amax = 1/amax[y_idx_all] - 1
        #lbt_amax = np.zeros(len(z_amax))
        #for j, zpart in enumerate(z_amax): lbt_amax[j] = quad(integrand, znew, zpart)[0]
        #lbt_amax *= H0**-1 # [lbt] = Gyr 
        #idx = lbt_amax < lbt_snaps
        
        f_ejected =  1-len(PID_old[x_idx])/LenGasOld  
        f_converted = len(np.where(IsStar[y_idx] == 1)[0])/LenGasOld
        f_stillgas = len(np.where(IsStar[y_idx] == 0)[0])/LenGasOld
        
        
        fig = plt.figure(figsize=[5,5])
        idx = np.where(IsStar[y_idx] == 1)[0]
                                               
        plt.hist(sigma_tot_old, bins=np.arange(0,301,10),color='r', histtype='step', lw=2)
        plt.hist(sigma_tot_old[x_idx][idx],bins=np.arange(0,301,10), histtype='step',color='c',ls='--', lw=1.5, label='Converted into stars')
        plt.hist(sigma_tot_old[x_idx_eject],bins=np.arange(0,301,10), histtype='step',color='m',ls='--',lw=1.5, label='Gas ejected out to 5kpc')
        plt.hist(sigma_tot_new,bins=np.arange(0,301,10), color='b', histtype='step',lw=2)
        plt.yscale('log')
        plt.xlabel(r'$\rm \sigma_{gas}\ [km/s]$', fontsize=16)
        plt.ylabel(r'$\rm N$', fontsize=16)
        plt.tick_params(labelsize=16,top=True, right=True, direction='in',length=10)
        plt.tick_params(length=5,top=True, right=True, which='minor', direction='in')
        plt.minorticks_on()
        plt.legend(loc='best',fontsize=14)
        plt.text(180,10**2,r'$\rm f_{stillgas} = %.2f$' %(f_stillgas), fontsize=14, color='r')
        plt.text(180,10**1.8,r'$\rm f_{stars} = %.2f$' %(f_converted), fontsize=14, color='c')
        plt.text(180,10**1.6, r'$\rm f_{ejected} = %.2f$' %(f_ejected), fontsize=14, color='m')
        
        #x = r_gas_new[y_idx]/r_gas_old[x_idx]
        #y = sigma_tot_new[y_idx]/sigma_tot_old[x_idx]       
        #xmin,xmax = 0,3
        #ymin,ymax = 0,3
        #nbins = 50
        #k = kde.gaussian_kde(np.array([x, y]))
        #xi, yi = np.mgrid[xmin:xmax:nbins*1j, ymin:ymax:nbins*1j]
        #zi = k(np.vstack([xi.flatten(), yi.flatten()]))
        #plt.pcolormesh(xi, yi, zi.reshape(xi.shape),shading='gouraud', cmap=plt.cm.viridis)
        #plt.axhline(y=1, color='k', lw=1, ls='--')
        #plt.axvline(x=1, color='k', lw=1, ls='--')
        #plt.xlabel(r'$\rm r(z=2)/r(z=2.24)$', fontsize=16)
        #plt.ylabel(r'$\rm \sigma_{gas}(z=2)/\sigma_{gas}(z=2.24)$', fontsize=16)
        #plt.tick_params(labelsize=16,top=True, right=True, direction='in',length=10)
        #plt.tick_params(length=5,top=True, right=True, which='minor', direction='in')
        #plt.minorticks_on()
        #plt.xlim(0,3)
        #plt.ylim(0,3)
        #plt.gca().set_aspect('equal', adjustable='box')
        #plt.yticks([0,1,2,3])
        #fig.savefig('Gas_position_sigma_variation_satellite.png', format='png', dpi=200, bbox_inches='tight')
        fig.savefig('gas_variations_2snaps.png', format='png', dpi=200, bbox_inches='tight')
        plt.show()
        
        
    else: print('Main Progenitor is not a Central')
    #"""   
    
if __name__ == '__main__':
    snap = int(sys.argv[1])
    L = int(sys.argv[2])
    NP  = int(sys.argv[3])
    GasTransfer(snap, L, NP)
    
    
    
    
