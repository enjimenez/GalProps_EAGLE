import h5py as h5
from Snapshots import get_zstr

def read_header (snap, NP, Lbox):
    dirf = '/home/esteban/Documents/EAGLE/'
    fend = get_zstr(snap)
    
    fs = h5.File(dirf + 'Galaxies/0%i_%s_%iMpc_%i_galaxies.hdf5' %(snap, fend, Lbox, NP), 'r')
    a        = fs['Header/a'].value         # Scale factor
    h        = fs['Header/h'].value  # h
    boxsize  = fs['Header/BoxSize'].value      # L [cMpc/h]
    fs.close()
    return a, h, boxsize