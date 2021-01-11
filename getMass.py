import h5py as h5
import numpy as np

def Aperture(itype, data, Tree, ap=30, sf=False):
        keys = ['h','a', 'snap', 'fend', 'Lbox', 'NP', 'Base', 'BasePart']
        h,a,snap,fend,Lbox,NP,Base,BasePart = [data['params'].get(key) for key in keys]
        if itype == 0: itype = 'Gas'
        if itype == 1: itype = 'DM'
        if itype == 4: itype = 'Star'
        
        gns  = data['gn']
        sgns = data['sgn']
        Pos_cop = data['Pos_cop']
        
        fh = h5.File(BasePart + '%s_%s_%iMpc_%i_%s.hdf5' %(snap, fend, Lbox, NP, itype), 'r')
        MassPart     = fh['PartData/Mass%s' %itype][()]/h    
        gnPart       = fh['PartData/GrpNum_%s' %itype][()]
        sgnPart      = fh['PartData/SubNum_%s' %itype][()]
        if sf: 
            mask = fh['PartData/SFR'][()] > 0
            MassPart = MassPart[mask]
            gnPart = gnPart[mask]
            sgnPart = sgnPart[mask]
        fh.close()
        
        LenGal = len(Pos_cop)   
        ApMass = np.zeros(LenGal, dtype=float)
        
        for i in range(LenGal):
            Tree.search(center=Pos_cop[i], radius=(ap/1000.)/a)
            idx = Tree.getIndices()
            if idx is None: continue
            gns_gal  = gnPart[idx]
            sgns_gal = sgnPart[idx]
            mask = (gns_gal == gns[i]) & (sgns_gal == sgns[i])
            LenPart = len(sgns_gal[mask])
            if LenPart == 0: continue
        
            ApMass[i] = np.sum(MassPart[idx][mask])
        return ApMass
