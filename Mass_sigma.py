from Snapshots import get_zstr
import pandas as pd
import numpy as np
import h5py as h5
import sys

# What happen with the EagleVariations???
Base = '/home/esteban/Documents/EAGLE/'

def Gas(snap, Lbox, NP, phy='REF', ap=10, Nmin=50, All=True, sf=True, nth=True, feed=True, nfeed=True, Mh=True):
    
    if phy == 'sSNII':  phy = 'StrongSNII'
    if phy == 'wSNII':  phy = 'WeakSNII'
    if phy == 'nofeed': phy = 'NoFeedback'
    if phy == 'noAGN':  phy = 'NOAGN'
    if phy == 'eos1':   phy = 'EOS1p000'
    if phy == 'eos5/3': phy = 'EOS1p666'
    if phy == 'REF':    phy = 'REFERENCE'
    
    fend = get_zstr(snap)
    snap = str(snap).zfill(3)
    box = 'L%sN%s' %(str(Lbox).zfill(4), str(NP).zfill(4))
    BaseGal = Base + 'Galaxies/%s/%s/' %(box,phy)
    output = Base + 'Data/%s/%s/' %(box,phy)
    
    LogMmin = 10. if Mh else 9.
    LogMmax = 14. if Mh else 12.4
    dLogM   = 0.2
    NBIN =  int((LogMmax-LogMmin)/dLogM)
    LogM = np.linspace(LogMmin, LogMmax, NBIN, endpoint=False) + dLogM/2.
    
    with h5.File(BaseGal + '%s_%s_%iMpc_%i_galaxies.hdf5' %(snap, fend, Lbox, NP), 'r') as f:
        a, h = f['Header/a'][()], f['Header/h'][()] 
        if Mh: LogMass = np.log10(f['SubHaloData/SubHaloMass']/h) + 10.
        else: LogMass = np.log10(f['StarData/TotalMstell']/h) + 10.
        
        VelDispGas       = f['Kinematics/Gas/VelDisp'][:,ap-1]
        VelDispGas_nth   = f['Kinematics/Gas/VelDisp_nth'][:,ap-1]
        VelDispGas_feed  = f['Kinematics/Gas/VelDisp_Feed'][:,ap-1]
        VelDispGas_nfeed = f['Kinematics/Gas/VelDisp_noFeed'][:,ap-1]
        VelDispSF        = f['Kinematics/Gas/SF/VelDisp'][:,ap-1]
        VelDispSF_nth    = f['Kinematics/Gas/SF/VelDisp_nth'][:,ap-1]
        
        NumPartGas       = f['Kinematics/Gas/NumPart'][:,ap-1]
        NumPartGas_feed  = f['Kinematics/Gas/NumPart_Feed'][:,ap-1]
        NumPartGas_nfeed = f['Kinematics/Gas/NumPart_noFeed'][:,ap-1]
        NumPartSF        = f['Kinematics/Gas/SF/NumPart'][:,ap-1]
        
    labels = ((LogMass - LogMmin)/(LogMmax-LogMmin) * NBIN).astype(int)
    strs = ['Gas', 'Gas_nothermal', 'Gas_Feed', 'Gas_noFeed', 'SF', 'SF_nothermal' ]
    
    for VelDisp, NumPart, usemode, mode  in zip([VelDispGas, VelDispGas_nth, VelDispGas_feed, VelDispGas_nfeed, VelDispSF, VelDispSF_nth],
                                      [NumPartGas, NumPartGas, NumPartGas_feed, NumPartGas_nfeed, NumPartSF, NumPartSF],
                                      [All, nth, feed, nfeed, sf, nth], strs):
        if usemode:
            mask = NumPart > Nmin
            sigma_med = np.zeros(NBIN, dtype=float)
            lo_err = np.zeros(NBIN, dtype=float)
            hi_err = np.zeros(NBIN, dtype=float)
            
            for i in range(NBIN):
                idx = np.where(labels[mask] == i)[0]
                LenArr = len(idx)
                if LenArr > 5:
                    srt_sigma = np.sort(VelDisp[mask][idx])
                    sigma_med[i] = np.median(srt_sigma)
                    lo_err[i] = srt_sigma[int(0.16*LenArr)]
                    hi_err[i] = srt_sigma[int(0.84*LenArr)]
                    
            data = {'LogM': LogM, 'sigma': sigma_med, 'lo_err': lo_err, 'hi_err': hi_err}

            x = 'LogMsub' if Mh else 'LogMstar'
            df = pd.DataFrame.from_dict(data)
            df.to_csv(output + 'LogM_VelDisp/%s_%s_VelDisp_%s_%ikpc.csv' %(snap,x,mode,ap), index=False)  
                
if __name__ == '__main__':
    snap = int(sys.argv[1])
    Lbox = int(sys.argv[2])
    NP   = int(sys.argv[3])
    #phy    = sys.argv[4]
    Gas(snap, Lbox, NP)
    Gas(snap, Lbox, NP, Mh=False)
    
            
