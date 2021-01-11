import h5py as h5
from Snapshots import get_zstr

def read_header (snap, NP, Lbox, phy):
    dirf = '/mnt/su3ctm/ejimenez/'
    fend = get_zstr(snap)
    
    if phy == 'sSNII': phy = 'StrongSNII'
    if phy == 'wSNII': phy = 'WeakSNII'
    if phy == 'nofeed': phy = 'NoFeedback'
    if phy == 'noAGN': phy = 'NOAGN'
    if phy == 'REF': phy = 'REFERENCE'
    if phy == 'eos1':   phy = 'EOS1p000'
    if phy == 'eos5/3': phy = 'EOS1p666'
    
    box = 
    
    fs = h5.File(dirf + 'Galaxies/%s_%s_%iMpc_%i_galaxies.hdf5' %(snap, fend, Lbox, NP), 'r')
    a        = fs['Header/a'][()]         # Scale factor
    h        = fs['Header/h'][()]  # h
    boxsize  = fs['Header/BoxSize'][()]      # L [cMpc/h]
    fs.close()
    return a, h, boxsize
