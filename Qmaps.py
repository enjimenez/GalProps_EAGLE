from scipy.interpolate import UnivariateSpline 
from sphviewer.tools import QuickView
from scipy.interpolate import interp1d 
from scipy.spatial import cKDTree as KDTree
import matplotlib.pyplot as plt
from astropy.constants import G
from matplotlib import rc
import astropy.units as u
import pandas as pd
import numpy as np
import h5py as h5

import scicm
np.seterr(divide='ignore', invalid='ignore')
rc('text', usetex=True) 
rc('axes', linewidth=1.2)
rc('font', family='serif')

Base = '/mnt/su3ctm/ejimenez/'
DataDir = '/mnt/su3ctm/ludlow/Equiparition/ICs/'

Mdm = 0.001000304
Mstar = 0.0002000705
Grav = G.to(u.km**2 * u.kpc * u.Msun**-1 * u.s**-2).value *1e10

def nbsearch(pos, nb, tree):
    d, idx = tree.query(pos, k=nb)
    hsml = d[:,nb-1]
    return hsml

def get_cylindrical(pos, vel):
    rho      = np.sqrt(np.square(pos[:,0]) + np.square(pos[:,1]))
    varphi   = np.arctan2(pos[:,1], pos[:,0])
    z        = pos[:,2]    
    v_rho    = vel[:,0] * np.cos(varphi) + vel[:,1] * np.sin(varphi)
    v_varphi = -vel[:,0]* np.sin(varphi) + vel[:,1] * np.cos(varphi)
    v_z      = vel[:,2]
    
    PosCyl = np.array([rho, varphi, z]).T
    VelCyl = np.array([v_rho, v_varphi, v_z]).T
    return(PosCyl, VelCyl) 

def getData(fdisk):
    global PosDM, VelDM
    global PosStar, VelStar
    
    fds = np.array([0.01,0.02,0.03])
    ffs = ['0p01', '0p02', '0p03']
    ID = np.where(fds == fdisk)[0][0]
    ff = ffs[ID]
    
    with h5.File(DataDir + 'mu_5_fdisk_%s/fdisk_%s_lgMdm_7p0_V200-200kmps/snap_010.hdf5' %(ff,ff), 'r') as f:  
        PosDM    = f['PartType1/Coordinates'][()]
        VelDM    = f['PartType1/Velocities'][()]
        PosStar  = f['PartType2/Coordinates'][()]
        VelStar  = f['PartType2/Velocities'][()]
        
    return (PosDM, VelDM, PosStar, VelStar)

def get_splines(PosDM, PosStar):
    LenDM = len(PosDM)
    LenStar = len(PosStar)
    pall = np.concatenate((PosDM, PosStar))
    mall = np.concatenate((np.ones(LenDM)*Mdm, np.ones(LenStar)*Mstar))
    rall = np.linalg.norm(pall, axis=1)
    
    isort = np.argsort(rall)
    rall  = rall[isort]
    mcum  = np.cumsum(mall[isort])
    
    Vc = np.sqrt(Grav*mcum/rall)
    rf = np.concatenate(([0], rall))
    vf = np.concatenate(([0], Vc))
    Vc = interp1d(rf, vf, assume_sorted=True)
    
    # Makes a smoother version of Vc and beta
    xx = np.arange(0, rf[-1], 0.2) 
    yy = Vc(xx)
    Vc = UnivariateSpline(xx, yy, s=0, k=1)
    grad = Vc.derivative()
    return (Vc, grad)

def kappa_map(lpix=0.07):
    """
    Epicyclic frequency (kappa map)
    lpix: length of the squared-pixel 
    """
    nbin  = int(2*Rmax/lpix)
    Vc_spl, grad_spl = get_splines(PosDM, PosStar)
    
    ##### Epicylic frequency (kappa) map ######
    xbins = np.linspace(-Rmax, Rmax, nbin+1)
    ybins = np.linspace(-Rmax, Rmax, nbin+1)
    rm =  (xbins[1:] + xbins[:-1])/2.
    X,Y = np.meshgrid(rm, rm)
    R = np.float32(np.sqrt(X**2 + Y**2))
    kappa = np.sqrt(2 * Vc_spl(R)/R * (grad_spl(R) + Vc_spl(R)/R))
    return kappa
    
    
def Q_map(lpix=0.07, nb=50):
    TreeStar = KDTree(PosStar)
    hsmlStar = nbsearch(PosStar, nb+1, TreeStar) # Smoothing lengths
    
    # Impose maximum hsml 
    hsmlMax = Rmax*np.sqrt(2)
    hsmlStar = np.where(hsmlStar > hsmlMax, hsmlMax, hsmlStar)
    
    PosStarCyl, VelStarCyl = get_cylindrical(PosStar, VelStar)
    argr  = np.zeros(len(PosStarCyl))
    
    for i, idx in enumerate(TreeStar.query_ball_point(PosStar, r=hsmlStar)):
        LenP = len(idx)
        
        # Discard particles with few neighbours
        if LenP < 25: continue 
        vr = VelStarCyl[idx,0]
        argr[i] = np.sum(vr**2)/LenP - (np.sum(vr)/LenP)**2
        
    sigma_r = np.where(argr > 0, np.sqrt(argr), 0)
    mask    = (PosStarCyl[:,0] < Rmax)
 
    MassStar = np.ones(len(PosStar))*Mstar 
    
    # Pixelated maps 
    nbin  = int(2*Rmax/lpix)

    Sigma_map = QuickView(PosStar[mask], MassStar[mask], hsmlStar[mask], r='infinity', plot=False, x=0, y=0, z=0, extent=[-Rmax,Rmax,-Rmax,Rmax], xsize=nbin, ysize=nbin, logscale=False).get_image()
    
    # mass-weighted sigma_r map
    sigma_map = QuickView(PosStar[mask], MassStar[mask]*sigma_r[mask], hsmlStar[mask], r='infinity', plot=False, x=0, y=0, z=0, extent=[-Rmax,Rmax,-Rmax,Rmax], xsize=nbin, ysize=nbin, logscale=False).get_image()
    
    sigma_map = sigma_map/Sigma_map 
    sigma_map = np.where(sigma_map > 0, sigma_map, np.NaN)
    
    kappa = kappa_map()
    
    Q = (kappa * sigma_map)/(3.36*Grav*Sigma_map)    
    
    Sigma_map = np.where(Sigma_map*1e10 > 1e5, np.log10(Sigma_map*1e10), -10)

    return {'sigma': sigma_map, 'Q': Q, 'Sigma': Sigma_map}
    
def plotMaps(data):
    sigma = data['sigma']
    Q     = data['Q']
    Sigma = data['Sigma']
    
    fig, (ax1,ax2,ax3) = plt.subplots(1,3, figsize=(12,4))

    labels = [r'\boldmath$\rm \log_{10}\ \Sigma_{\star}\ [M_{\odot} kpc^{-2}]$',
    r'\boldmath $\sigma_r\ \rm [km\ s^{-1}]$',
    r'\boldmath $\rm \log_{10}Q_{\star}$']

    extent = [-Rmax, Rmax,-Rmax, Rmax]
    cc1 = ax1.imshow(Sigma, vmin=5, vmax=9, origin='lower', cmap='inferno', extent=extent)
    cc2 = ax2.imshow(sigma, vmin=0, origin='lower', cmap='plasma', extent=extent)
    cc3 = ax3.imshow(np.log10(Q), origin='lower', vmin=-1, vmax=1, cmap='scicm.BkR', extent=extent)
    
    for ax in (ax1,ax2,ax3):
        ax.set_xlim(-Rmax,Rmax)
        ax.set_ylim(-Rmax,Rmax)
        ax.set_aspect('equal')
        ax.tick_params(labelsize=12,top=True,right=True,width=1.2, direction='in',length=5)
        ax.tick_params(length=2,top=True, right=True,which='minor', direction='in')
        ax.minorticks_on()
        
    fig.subplots_adjust(wspace=0.1, hspace=0.1, top=0.95)        
       
    for ax, label, cc in zip((ax1,ax2,ax3), labels, [cc1,cc2,cc3]):
        p = ax.get_position().get_points().flatten()
        cbar = fig.add_axes([p[0], 0.97, p[2]-p[0], 0.02])    
        cb = fig.colorbar(cc, cax=cbar, orientation='horizontal')
        cb.ax.tick_params(labelsize=11)
        cb.ax.set_xlabel(label, fontsize=12, labelpad=8)
        cbar.xaxis.set_label_position('top')
        cbar.xaxis.set_ticks_position('top')
    
    fig.subplots_adjust(hspace=0.1)
    fig.savefig(Base + 'test_map.jpeg', dpi=300, format='jpeg', bbox_inches='tight')
    plt.close(fig)

if __name__ == '__main__':
    Rmax = 20
    
    getData(0.01)    
    data = Q_map()
    plotMaps(data)
    
    
    
    
        
    
    
    
    
    







