from Snapshots import get_zstr
from read_header import read_header
import numpy as np
import astropy.units as u
from astropy.constants import G
from numpy.linalg import norm 
import h5py as h5
from compute_HI_H2_masses import HI_H2_masses
from prody.kdtree.kdtree import KDTree
from scipy.interpolate import interp1d
import time
import sys
#Base = '/mnt/su3ctm/ejimenez/'
Base  = '/home/esteban/Documents/EAGLE/'


def distance(r1, r0, Lboxkpc):
        Lx = abs(r1[0]-r0[0])
        Ly = abs(r1[1]-r0[1])
        Lz = abs(r1[2]-r0[2])
        if Lx >= Lboxkpc/2.: Lx = Lboxkpc - Lx
        if Ly >= Lboxkpc/2.: Ly = Lboxkpc - Ly
        if Lz >= Lboxkpc/2.: Lz = Lboxkpc - Lz
        return np.sqrt(Lx**2 + Ly**2 + Lz**2)
    
def DeltaR(r, r0, Lboxkpc): 
    deltaR = r - r0
    Lx, Lxabs = deltaR[0], abs(deltaR[0])
    Ly, Lyabs = deltaR[1], abs(deltaR[1])
    Lz, Lzabs = deltaR[2], abs(deltaR[2])
    if Lxabs >= Lboxkpc/2.:
        if Lx > 0: deltaR[0] = Lxabs - Lboxkpc
        else: deltaR[0] = Lboxkpc - Lxabs
    if Lyabs >= Lboxkpc/2.:
        if Ly > 0: deltaR[1] = Lyabs - Lboxkpc
        else: deltaR[1] = Lboxkpc - Lyabs
    if Lzabs >= Lboxkpc/2.:
        if Lz > 0: deltaR[2] = Lzabs - Lboxkpc
        else: deltaR[2] = Lboxkpc - Lzabs
    return deltaR

class GalCat:
    def __init__(self, snap, NP=1504, Lbox=100, REF=True, gas=False, stars=False, dm=False):
        self.snap = snap
        self.fend = get_zstr(snap)
        self.NP = NP
        self.a, self.h, self.boxsize = read_header(snap, NP, Lbox)
        self.Lbox = Lbox
        self.Lboxkpc = Lbox * 1000
        if REF: 
            self.pp = ''
            phy = 'REFERENCE'
        else: 
            self.pp = 'RECAL_'
            phy = 'RECALIBRATED'
        
        self.data = self.get_data() 
        #self.BasePart = Base + 'L%sN%s/%s/' %(str(Lbox).zfill(4), str(NP).zfill(4), phy)
        self.BasePart = Base + 'processed_data/'
        
        BS  = self.boxsize
        # Inititalize the Trees
        if gas: 
            fh = h5.File(self.BasePart + '0%i_%s_%iMpc_%i_Gas.hdf5'
                            %(self.snap, self.fend, self.Lbox, self.NP), 'r')
            PosGas  = fh['PartData/PosGas'].value
            fh.close()
            self.TreeGas  = KDTree(unitcell=np.asarray([BS,BS,BS]), coords = PosGas, bucketsize=10)
            del PosGas
        if stars: 
            fh = h5.File(self.BasePart + '0%i_%s_%iMpc_%i_Star.hdf5'
                            %(self.snap, self.fend, self.Lbox, self.NP), 'r')
            PosStar  = fh['PartData/PosStar'].value
            fh.close()
            self.TreeStar  = KDTree(unitcell=np.asarray([BS,BS,BS]), coords = PosStar, bucketsize=10) 
            del PosStar
        if dm: 
            fh = h5.File(self.BasePart + '0%i_%s_%iMpc_%i_DM.hdf5'
                            %(self.snap, self.fend, self.Lbox, self.NP), 'r')
            PosDM  = fh['PartData/PosDM'].value
            fh.close()
            self.TreeDM  = KDTree(unitcell=np.asarray([BS,BS,BS]), coords = PosDM, bucketsize=10) 
            del PosDM
        
    def get_data (self):
        fs = h5.File(Base + 'Galaxies/0%i_%s_%iMpc_%i_%sgalaxies.hdf5' 
                %(self.snap, self.fend, self.Lbox, self.NP, self.pp), 'r')
    
        TotalMstell = fs['StarData/TotalMstell'].value/self.h
        LogMstell   = np.log10(TotalMstell +1e-10 ) + 10
        del TotalMstell
        Pos_cop     = fs['SubHaloData/Pos_cop'].value/self.h * self.a * 1000 #pkpc
        Vcom        = fs['SubHaloData/V_com'].value * self.a**(1/2.)
        hmr         = fs['StarData/HalfMassRad'].value/self.h * self.a * 1000 
        try: 
            Lstars = fs['Kinematics/Stars/L_70kpc'].value 
            fs.close()
            return {'Pos_cop': Pos_cop, 'LogMstell': LogMstell, 'Vcom':Vcom, 'hmr': hmr, 'Lstars': Lstars}
        except: 
            print("No Lstars in 'Galaxies' File...")
            fs.close()
            return {'Pos_cop': Pos_cop, 'LogMstell': LogMstell, 'Vcom':Vcom, 'hmr': hmr}
    
    def add_Gas_mass(self):
        COPc = self.data['Pos_cop'] * self.h * self.a**-1 * 1.e-3        
        fh = h5.File(self.BasePart + '0%i_%s_%iMpc_%i_Gas.hdf5'
                        %(self.snap, self.fend, self.Lbox, self.NP), 'r')
        MassGas     = fh['PartData/MassGas'].value/self.h    
        fh.close()
        
        LenGal = len(COPc)
        MassTotal  = np.zeros(LenGal, dtype=float)
        for i in range(LenGal):
            self.TreeGas.search(center=COPc[i],radius=0.03)
            idx = self.TreeGas.getIndices()
            MassTotal[i] = np.sum(MassGas[idx])
            
        fs = h5.File(Base + 'Galaxies/0%i_%s_%iMpc_%i_%sgalaxies.hdf5' 
                     %(self.snap, self.fend, self.Lbox, self.NP, self.pp), 'a')
        
        fs.create_dataset('GasData/MassGas_30kpc', data = MassTotal) 
        fs.close()

    def add_HI_H2_masses(self):
            # Add the HI and H2 Masses for each galaxy
            # uses the pre-computed masses in gas particles   
            COPc = self.data['Pos_cop'] * self.h * self.a**-1 * 1.e-3
            massHI, massH2 = HI_H2_masses(self.snap, self.NP, self.BasePart, a=self.a, h=self.h, Lbox=self.Lbox)
            
            LenGal = len(COPc)
            MassHI_30kpc = np.zeros(LenGal)
            MassH2_30kpc = np.zeros(LenGal)
            
            for i in range(LenGal):
                self.TreeGas.search(center=COPc[i],radius=0.03)
                idx = self.TreeGas.getIndices()
                MassHI_gal, MassH2_gal = massHI[idx], massH2[idx]
                if len(idx) < 10: continue
                MassHI_30kpc[i] = np.sum(MassHI_gal)
                MassH2_30kpc[i] = np.sum(MassH2_gal)
            
            fs = h5.File(Base + 'Galaxies/0%i_%s_%iMpc_%i_%sgalaxies.hdf5' 
                            %(self.snap, self.fend, self.Lbox, self.NP, self.pp), 'a')

            fs.create_dataset('GasData/MassHI_30kpc', data = MassHI_30kpc)  
            fs.create_dataset('GasData/MassH2_30kpc', data = MassH2_30kpc) 
            fs.close()  
            
    def add_Lstars(self):
        Pos_cop    = self.data['Pos_cop']
        COPc       = Pos_cop * self.h * self.a**-1 * 1.e-3
        Vcom       = self.data['Vcom']
        BSkpc      = self.Lboxkpc
    
        fh = h5.File(self.BasePart + '0%i_%s_%iMpc_%i_Star.hdf5'
                        %(self.snap, self.fend, self.Lbox, self.NP), 'r')
        
        MassStar    = fh['PartData/MassStar'].value/self.h
        PosStar     = fh['PartData/PosStar'].value/self.h * self.a *  1000
        Vstar       = fh['PartData/VelStar'].value * self.a**(1/2.)
        fh.close()
    
        LenGal = len(Pos_cop)
        L    = np.zeros((LenGal, 3), dtype=float)
        
        for i in range(LenGal):
            self.TreeStar.search(center=COPc[i],radius=0.07)
            idx = self.TreeStar.getIndices()
            MassStar_gal = MassStar[idx]
            PosStar_gal  = PosStar[idx]
            Vstar_gal    = Vstar[idx] 
            
            centre = Pos_cop[i]
            LenStars = len(idx)
            Lstars = np.zeros((LenStars, 3), dtype=float)
            rfinal = np.array([DeltaR(p, centre, BSkpc) for p in PosStar_gal]) 
            vfinal = Vstar_gal - Vcom[i]
            cross = np.cross(rfinal, vfinal)
            for k in range(LenStars): Lstars[k] = MassStar_gal[k] * cross[k]
            L[i] = np.sum(Lstars, axis=0)
            
        fs = h5.File(Base + 'Galaxies/0%i_%s_%iMpc_%i_%sgalaxies.hdf5' 
                     %(self.snap, self.fend, self.Lbox, self.NP,self.pp), 'a')
        fs.create_dataset('/Kinematics/Stars/L_70kpc', data = L) 
        fs.close()
    
    def add_sigma(self, itype, SF=True, NSF=False, neutral=False):
        if itype == 0:
            if SF: 
                fs = h5.File(Base + 'Galaxies/0%i_%s_%iMpc_%i_%sgalaxies.hdf5' 
                             %(self.snap, self.fend, self.Lbox, self.NP, self.pp), 'a')
                
                sigmas = self.compute_sigma(itype, SF=True)
                try:
                    fs.create_dataset('Kinematics/Gas/SF/VelDisp', data = sigmas)  
                except:
                    dat1 = fs['Kinematics/Gas/SF/VelDisp']
                    dat1[...] = sigmas
                
            if NSF:
                fs = h5.File(Base + 'Galaxies/0%i_%s_%iMpc_%i_%sgalaxies.hdf5' 
                             %(self.snap, self.fend, self.Lbox, self.NP,self.pp), 'a')
                
                sigmas = self.compute_sigma(itype, SF=False, NSF=True)
                fs.create_dataset('Kinematics/Gas/NSF/VelDisp', data = sigmas) 
                
            if neutral:
                fs = h5.File(Base + 'Galaxies/0%i_%s_%iMpc_%i_%sgalaxies.hdf5' 
                             %(self.snap, self.fend, self.Lbox, self.NP,self.pp), 'a')
                
                sigmas = self.compute_sigma(itype, SF=False, neutral=True)
                fs.create_dataset('Kinematics/Gas/neutral/VelDisp', data = sigmas) 
                
        if itype == 4:
            fs = h5.File(Base + 'Galaxies/0%i_%s_%iMpc_%i_%sgalaxies.hdf5' 
                             %(self.snap, self.fend, self.Lbox, self.NP,self.pp), 'a')
                
            sigmas = self.compute_sigma(itype)
            try: fs.create_dataset('Kinematics/Stars/VelDisp', data = sigmas) 
            except:
                dat1 = fs['Kinematics/Stars/VelDisp']
                dat1[...] = sigmas
        fs.close()
    
    def add_SigmaGas(self):
        # Sigmas below cold be computing in the same function...
        Pos_cop   = self.data['Pos_cop']
        COPc       = Pos_cop * self.h * self.a**-1 * 1.e-3
        BSkpc      = self.Lboxkpc
        
        fgas = h5.File(self.BasePart + '0%i_%s_%iMpc_%i_Gas.hdf5'
                       %(self.snap, self.fend, self.Lbox, self.NP), 'r')
        
        MassGas        = fgas['PartData/MassGas'].value/self.h * 1.e10          #[Msun] 
        PosGas         = fgas['PartData/PosGas'].value/self.h * self.a * 1000   #[pkpc]    
        fgas.close()
        
        LenGal = len(Pos_cop)
        Sigma  = np.zeros((LenGal, 30), dtype=float)
        
        for i in range(LenGal):
            self.TreeGas.search(center=COPc[i],radius=0.031)
            idx = self.TreeGas.getIndices()
            
            PosGas_gal, MassGas_gal   = PosGas[idx], MassGas[idx]
            if len(idx) < 10: continue
            
            centre = Pos_cop[i]
            r = np.array([distance(p, centre, BSkpc) for p in PosGas_gal])
            mask = np.argsort(r)
            r = r[mask]
            cmass = np.cumsum(MassGas_gal[mask])
            pi = 3.141593
            SigmaCum = cmass/(pi*r*r)
            
            maxr, minr = r.max(), r.min()
            rmax = 30 if maxr > 30 else int(maxr)
            rmin = 1  if minr < 1  else int(minr+1)
            
            rads = np.arange(rmin,rmax+1)
            Sigma[i][rmin-1:rmax] = np.interp(rads, r, SigmaCum)
                
        fs = h5.File(Base + 'Galaxies/0%i_%s_%iMpc_%i_%sgalaxies.hdf5' 
                        %(self.snap, self.fend, self.Lbox, self.NP,self.pp), 'a')
        
        fs.create_dataset('GasData/SigmaGas', data = Sigma)  
        fs.close()
    
    def add_SigmaSFR(self):
        Pos_cop   = self.data['Pos_cop']
        COPc       = Pos_cop * self.h * self.a**-1 * 1.e-3
        BSkpc      = self.Lboxkpc

        fgas = h5.File(self.BasePart + '0%i_%s_%iMpc_%i_Gas.hdf5'
                       %(self.snap, self.fend, self.Lbox, self.NP), 'r')
        
        SFR            = fgas['PartData/SFR'].value   
        PosGas         = fgas['PartData/PosGas'].value/self.h * self.a * 1000   #[pkpc]    
        fgas.close()
    
        LenGal = len(Pos_cop)
        Sigma  = np.zeros((LenGal, 30), dtype=float)
        
        for i in range(LenGal):
            self.TreeGas.search(center=COPc[i],radius=0.031)
            idx = self.TreeGas.getIndices()
            PosGas_gal, SFR_gal   = PosGas[idx], SFR[idx]
            if len(idx) < 10: continue
            
            centre = Pos_cop[i]
            r = np.array([distance(p, centre, BSkpc) for p in PosGas_gal])
            mask = np.argsort(r)
            r = r[mask]
            csfr = np.cumsum(SFR_gal[mask])
            pi = 3.141593
            SigmaCum = csfr/(pi*r*r)
            
            maxr, minr = r.max(), r.min()
            rmax = 30 if maxr > 30 else int(maxr)
            rmin = 1  if minr < 1  else int(minr+1)
            
            rads = np.arange(rmin,rmax+1)
            Sigma[i][rmin-1:rmax] = np.interp(rads, r, SigmaCum)
                
        fs = h5.File(Base + 'Galaxies/0%i_%s_%iMpc_%i_%sgalaxies.hdf5' 
                        %(self.snap, self.fend, self.Lbox, self.NP,self.pp), 'a')
        
        fs.create_dataset('GasData/SigmaSFR', data = Sigma)  
        fs.close()
    
    def add_SigmaStar(self):
        Pos_cop   = self.data['Pos_cop']     
        COPc       = Pos_cop * self.h * self.a**-1 * 1.e-3
        BSkpc      = self.Lboxkpc
        fs = h5.File(self.BasePart + '0%i_%s_%iMpc_%i_Star.hdf5'
                       %(self.snap, self.fend, self.Lbox, self.NP), 'r')
        
        MassStar        = fs['PartData/MassStar'].value/self.h * 1.e10          #[Msun] 
        PosStar         = fs['PartData/PosStar'].value/self.h * self.a * 1000   #[pkpc]    
        fs.close()
        
        LenGal = len(Pos_cop)
        Sigma  = np.zeros((LenGal, 30), dtype=float)
        
        for i in range(LenGal):
            self.TreeStar.search(center=COPc[i],radius=0.031)
            idx = self.TreeStar.getIndices()
            
            PosStar_gal, MassStar_gal   = PosStar[idx], MassStar[idx]
            if len(idx) < 10: continue
            
            centre = Pos_cop[i]
            r = np.array([distance(p, centre, BSkpc) for p in PosStar_gal])
            mask = np.argsort(r)
            r = r[mask]
            cmass = np.cumsum(MassStar[mask])
            pi = 3.141593
            SigmaCum = cmass/(pi*r*r)
                    
            maxr, minr = r.max(), r.min()
            rmax = 30 if maxr > 30 else int(maxr)
            rmin = 1  if minr < 1  else int(minr+1)
            
            rads = np.arange(rmin,rmax+1)
            Sigma[i][rmin-1:rmax] = np.interp(rads, r, SigmaCum)
                
        fs = h5.File(Base + 'Galaxies/0%i_%s_%iMpc_%i_%sgalaxies.hdf5' 
                        %(self.snap, self.fend, self.Lbox, self.NP,self.pp), 'a')
        
        fs.create_dataset('StarData/SigmaStar', data = Sigma)  
        fs.close()
    
    def add_Vcirc(self): 
        Pos_cop     = self.data['Pos_cop']
        COPc        = Pos_cop * self.h * self.a**-1 * 1.e-3
        BSkpc       = self.Lboxkpc
        fdm         = h5.File(self.BasePart + '0%i_%s_%iMpc_%i_DM.hdf5'
                              %(self.snap, self.fend, self.Lbox, self.NP), 'r')
        
        dm_mass       = fdm['Header/PartMassDM'].value/self.h * 1.e10
        PosDM         = fdm['PartData/PosDM'].value/self.h * self.a * 1000      #[pkpc] 
        MassDM        = np.ones(len(PosDM))*dm_mass              #[Msun] 
        fdm.close()
        
        fgas = h5.File(self.BasePart + '0%i_%s_%iMpc_%i_Gas.hdf5'
                        %(self.snap, self.fend, self.Lbox, self.NP), 'r')
        
        MassGas        = fgas['PartData/MassGas'].value/self.h * 1.e10          #[Msun] 
        PosGas         = fgas['PartData/PosGas'].value/self.h * self.a * 1000   #[pkpc]    
        fgas.close()
        
        fstar = h5.File(self.BasePart + '0%i_%s_%iMpc_%i_Star.hdf5'
                        %(self.snap, self.fend, self.Lbox, self.NP), 'r')
        
        MassStar       = fstar['PartData/MassStar'].value/self.h * 1.e10         #[Msun] 
        PosStar        = fstar['PartData/PosStar'].value/self.h * self.a * 1000  #[pkpc]   
        fstar.close()
        
        LenGal = len(Pos_cop)
        Vcirc  = np.zeros((LenGal, 150), dtype=float)
        kappa  = np.zeros((LenGal, 150), dtype=float)
        
        def rotation_curve(pos, mass, centre):
            r = np.array([distance(p,centre, BSkpc) for p in pos])
            mask = np.argsort(r)
            r = r[mask]
            cmass = np.cumsum(mass[mask])
            myG = G.to(u.km**2 * u.kpc * u.Msun**-1 * u.s**-2).value
            v = np.sqrt((myG*cmass) / r )
            LenPart = len(v)
            
            dvdr = np.zeros(LenPart, dtype=float)
            dvdr[0]=v[0]/r[0]
            for k in range(1, LenPart): dvdr[k] = (v[k]-v[k-1])/(r[k]-r[k-1])
            return r, v, dvdr
        
        for i in range(LenGal):
            
            self.TreeDM.search(center=COPc[i],radius=0.155) #TODO
            self.TreeGas.search(center=COPc[i],radius=0.155)
            self.TreeStar.search(center=COPc[i],radius=0.155)
            
            idxDM = self.TreeDM.getIndices()
            idxGas = self.TreeGas.getIndices()
            idxStar = self.TreeStar.getIndices()
        
            PosDM_gal, MassDM_gal     = PosDM[idxDM], MassDM[idxDM]
            PosGas_gal, MassGas_gal   = PosGas[idxGas], MassGas[idxGas]
            PosStar_gal, MassStar_gal = PosStar[idxStar], MassStar[idxStar]
            
            data_pos  = np.concatenate((PosDM_gal, PosGas_gal, PosStar_gal), axis=0)
            data_mass = np.concatenate((MassDM_gal, MassGas_gal, MassStar_gal))
    
            r, v, dvdr = rotation_curve(data_pos, data_mass, Pos_cop[i])
            rmask = r < 155
            r = r[rmask]
            v = v[rmask]
            dvdr = dvdr[rmask]
            
            maxr, minr = r.max(), r.min()
            rmax = 150 if maxr > 150 else int(maxr)
            rmin = 1  if minr < 1  else int(minr+1)
            
            R = np.arange(rmin,rmax+1)
            Vcirc[i][rmin-1:rmax] = np.interp(R, r, v)
            
            Vc  = Vcirc[i][rmin-1:rmax]
            dVdR = np.interp(R, r, dvdr) 
            
            kappa[i][rmin-1:rmax] = np.sqrt(2) * np.sqrt(Vc**2/R**2 + Vc*dVdR/R )
            
        fs = h5.File(Base + 'Galaxies/0%i_%s_%iMpc_%i_%sgalaxies.hdf5' 
                     %(self.snap, self.fend, self.Lbox, self.NP,self.pp), 'a')
        
        try: fs.create_dataset('Kinematics/Vcirc', data = Vcirc) 
        except:
            dat1 = fs['Kinematics/Vcirc']
            dat1[...] = Vcirc
        fs.create_dataset('Kinematics/kappa', data = kappa)  
        fs.close()
    
    def add_jstars(self):
        Pos_cop     = self.data['Pos_cop']
        COPc        = Pos_cop * self.h * self.a**-1 * 1.e-3
        hmr         = self.data['hmr']
        Vcom        = self.data['Vcom']
        LenGal   = len(Pos_cop)
        BSkpc      = self.Lboxkpc
        
        j_r50    = np.zeros(LenGal, dtype=np.float32)
        fh = h5.File(self.BasePart + '0%i_%s_%iMpc_%i_Star.hdf5'
                    %(self.snap, self.fend, self.Lbox, self.NP), 'r')
        
        MassStar    = fh['PartData/MassStar'].value/self.h
        PosStar     = fh['PartData/PosStar'].value/self.h * self.a * 1000
        Vstar       = fh['PartData/VelStar'].value * self.a**(1/2.)
        fh.close()
        
        for i in range(LenGal):
            self.TreeStar.search(center=COPc[i],radius=0.03) 
            idx = self.TreeStar.getIndices()
            
            MassStar_gal = MassStar[idx]
            PosStar_gal  = PosStar[idx]
            Vstar_gal    = Vstar[idx] 
            centre = Pos_cop[i]
            
            delta_V = Vstar_gal - Vcom[i]
            Len_Stars = len(PosStar_gal)
            r = np.zeros(Len_Stars, dtype=float)
            for k in range(Len_Stars): r[k] = distance(PosStar_gal[k], centre, BSkpc) 
            
            tt = np.argsort(r)
            r = r[tt]
            MassStar_sorted = MassStar_gal[tt]

            #Compute j
            deltaPos = np.zeros((Len_Stars, 3), dtype=np.float32)
            for k in range(Len_Stars): deltaPos[k] = DeltaR(PosStar_gal[k], centre, BSkpc)
            Ltemp = np.cross(deltaPos[tt], delta_V[tt]) # checked
            L_stars = np.zeros((Len_Stars,3), dtype=np.float32)  
            for k in range(Len_Stars): L_stars[k] = MassStar_sorted[k] * Ltemp[k]
            
            j_tmp = np.zeros((Len_Stars,3), dtype=np.float32)
            Lcum = np.cumsum(L_stars, axis=0)               
            Masscum = np.cumsum(MassStar_sorted)            
            for k in range(Len_Stars): j_tmp[k] = Lcum[k]/Masscum[k]
            
            jj = norm(j_tmp, axis=1)
        
            mask = (r > hmr[i]-5) & (r < hmr[i]+5)
            f = interp1d(r[mask], jj[mask])
            j_r50[i]  = f(hmr[i])
            
        fs = h5.File(Base + 'Galaxies/0%i_%s_%iMpc_%i_%sgalaxies.hdf5' 
                             %(self.snap, self.fend, self.Lbox, self.NP, self.pp), 'a')
        
        try:
            fs.create_dataset('Kinematics/Stars/jr50', data = j_r50)
        except:
            dat1 = fs['Kinematics/Stars/jr50']
            dat1[...] = j_r50
        fs.close()
    
    def compute_sigma(self, itype, NumP=10, SF=True, NSF=False, neutral=False):
        # should be in a separate function
        BSkpc      = self.Lboxkpc
        Pos_cop     = self.data['Pos_cop']
        COPc = self.data['Pos_cop'] * self.h * self.a**-1 * 1.e-3
        Vcom        = self.data['Vcom']
        L           = self.data['Lstars']
        LenGal   = len(Pos_cop)
        sigma = np.zeros((LenGal,30), dtype=np.float32) 
        
        if itype == 0:
            fh = h5.File(self.BasePart + '0%i_%s_%iMpc_%i_Gas.hdf5'
                        %(self.snap, self.fend, self.Lbox, self.NP), 'r')
        
            MassPart    = fh['PartData/MassGas'].value/self.h
            PosPart     = fh['PartData/PosGas'].value/self.h * self.a * 1000
            VPart       = fh['PartData/VelGas'].value * self.a**(1/2.)
            SFR         = fh['PartData/SFR'].value     # Msun/yr
            Density     = fh['PartData/Density'].value 
            Entropy     = fh['PartData/Entropy'].value
            #fneutral    = fh['PartData/fneutral'].value
            
            UnitPressure = fh['Constants/UnitPressure'].value
            UnitDensity  = fh['Constants/UnitDensity'].value
            gamma        = fh['Constants/gamma'].value
            
            Density  = Density * self.h**2 * self.a**-3  * UnitDensity
            Entropy  = Entropy * self.h**(2-2*gamma) * UnitPressure * UnitDensity**(-1*gamma)
            Pressure = Entropy*Density**gamma
            del Entropy
            fh.close()
            
        if itype == 4:
            fh = h5.File(self.BasePart + '0%i_%s_%iMpc_%i_Star.hdf5'
                    %(self.snap, self.fend, self.Lbox, self.NP), 'r')

            MassPart    = fh['PartData/MassStar'].value/self.h
            PosPart     = fh['PartData/PosStar'].value/self.h * self.a * 1000
            VPart       = fh['PartData/VelStar'].value * self.a**(1/2.)
            fh.close()
                
        for i in range(LenGal):
            if itype == 0:
                self.TreeGas.search(center=COPc[i],radius=0.035)
                idx = self.TreeGas.getIndices()
                
                if SF: 
                    SFR_gal = SFR[idx]
                    mask   = SFR_gal > 0
                    LenPart = len(SFR_gal[mask])
                    if LenPart < NumP: continue
                if NSF:
                    SFR_gal = SFR[idx]
                    mask   = SFR_gal == 0
                    LenPart = len(SFR_gal[mask])
                    if LenPart < NumP: continue
                #if neutral:
                    #fneutral_gal = fneutral[idx]
                    #mask   = fneutral_gal > 1e-10
                    #LenPart = len(fneutral_gal[mask])
                    #if LenPart < NumP: continue
                    
                Pressure_gal     = Pressure[idx][mask]
                Density_gal      = Density[idx][mask]
                MassPart_gal = MassPart[idx][mask]
                PosPart_gal  = PosPart[idx][mask]
                VPart_gal    = VPart[idx][mask] 
            
            if itype == 4:
                self.TreeStar.search(center=COPc[i],radius=0.035)
                idx = self.TreeStar.getIndices()
                
                MassPart_gal = MassPart[idx]
                PosPart_gal  = PosPart[idx]
                VPart_gal    = VPart[idx] 
                LenPart      = len(idx)
            
            centre = Pos_cop[i]
            vcom   = Vcom[i]
            
            # Compute sigma
            r = np.zeros(LenPart, dtype=float)
            for k in range(LenPart): r[k] = distance(PosPart_gal[k], centre, BSkpc) 
            tt = np.argsort(r)
            r = r[tt]
            
            delta_V = VPart_gal[tt] - vcom
            MassPart_sorted = MassPart_gal[tt]
            e_gas = np.zeros(LenPart, dtype=float)
            
            if itype == 0: 
                sigmaP  = (np.sqrt(Pressure_gal/Density_gal))[tt] * 1e-5  # km/s  
                for k in range(LenPart): e_gas[k] = MassPart_sorted[k] * ( (np.dot(delta_V[k], L[i])/float(norm(L[i])))**2 + (sigmaP[k]/3.)**2)
            
            if itype == 4:
                for k in range(LenPart): e_gas[k] = MassPart_sorted[k] * ( (np.dot(delta_V[k], L[i])/float(norm(L[i])))**2 )
                
            
            sigma_tmp = np.sqrt( np.cumsum(e_gas)/np.cumsum(MassPart_sorted) )
            
            minr, maxr = r.min(), r.max()
            rmax = 30 if maxr > 30 else int(maxr)
            rmin = 1  if minr < 1  else int(minr+1)
            
            rads = np.arange(rmin,rmax+1)
            sigma[i][rmin-1:rmax] = np.interp(rads, r, sigma_tmp)

        return sigma
    
    def add_ToomreParam(self):
        pi = 3.141593
        myG = G.to(u.kpc**3 * u.Msun**-1 * u.s**-2)
        km_per_kpc = 3.2408e-17
        
        fs = h5.File(Base + 'Galaxies/0%i_%s_%iMpc_%i_%sgalaxies.hdf5'
                     %(self.snap, self.fend, self.Lbox, self.NP,self.pp), 'r+')
        
        kappa       = fs['Kinematics/kappa'].value[:,:30] 
        VelDispGas  = fs['Kinematics/Gas/SF/VelDisp'].value
        VelDispStar = fs['Kinematics/Stars/VelDisp'].value
        SigmaGas    = fs['GasData/SigmaGas'].value
        SigmaStar  = fs['StarData/SigmaStar'].value
        
        Pos_cop  = self.data['Pos_cop']
        LenGal   = len(Pos_cop)
        Qgas     = np.zeros((LenGal,30), dtype=float)
        Qstar    = np.zeros((LenGal,30), dtype=float)
        Qnet     = np.zeros((LenGal,30), dtype=float)
        
        for i in range(LenGal):
            Qgas[i]  = kappa[i]*VelDispGas[i]/(pi*myG*SigmaGas[i]) * km_per_kpc**2
            Qstar[i] = kappa[i]*VelDispStar[i]/(pi*myG*SigmaStar[i]) * km_per_kpc**2
            
            W = 2*VelDispGas[i]*VelDispStar[i]/(VelDispGas[i]**2 + VelDispStar[i]**2)
            for k in range(30):
                qgas, qstar = Qgas[i][k], Qstar[i][k]
                if qstar >= qgas: Qnet[i][k] = (W[k]/qstar + qgas**-1)**-1
                if qgas >= qstar: Qnet[i][k] = (W[k]/qgas + qstar**-1)**-1
            
        fs.create_dataset('Kinematics/Qgas',  data = Qgas)
        fs.create_dataset('Kinematics/Qstar', data = Qstar)
        fs.create_dataset('Kinematics/Qnet',  data = Qnet)
        fs.close()
    
        #if Net:
            ##W = np.zeros((LenGal, 10), dtype=float)
            #VelDispGas    = fs['Kinematics/Gas/SF/VelDisp'].value 
            #VelDispStar   = fs['Kinematics/Stars/VelDisp'].value 
            #Qstars        = fs['Kinematics/Stars/Q'].value
            #Qgas          = fs['Kinematics/Gas/Q'].value
            #fs.close()
            
            #for i in range(LenGal): 
                #W = 2* VelDispGas[i]* VelDispStar[i]/( VelDispGas[i]**2 + VelDispStar[i]**2)
                
                #idx1 =  np.where( (Qstars[i] >= Qgas[i]) & (Qgas[i] != 0) & (Qstars[i] != 0))[0]
                #idx2 =  np.where( (Qgas[i] >= Qstars[i]) & (Qgas[i] != 0) & (Qstars[i] != 0))[0]
                #Q[i][idx1] = W[idx1]/Qstars[i][idx1] + Qgas[i][idx1]**-1 
                #Q[i][idx2] = W[idx2]/Qgas[i][idx2] + Qstars[i][idx2]**-1 
            
            #fs = h5.File(fn, 'a')
            #data = fs['Kinematics/Qnet']
            #data[...] = Q
            ##fs.create_dataset('Kinematics/Qnet', data = Q)  
            #fs.close()    
        #else:
            #if itype == 0:
                #SigmaGas = fs['GasData/SigmaGas'].value  * u.Msun * u.kpc**-2      #[Msun/kpc^2]
                #VelDisp  = fs['Kinematics/Gas/SF/VelDisp'].value                   #[km/s]
                #prod     = VelDisp * Vcirc
                #const = (myG * r * SigmaGas * pi).to(u.km**2 * u.s**-2)
                
                #for i in range(LenGal): 
                    #idx = np.where(const[i] != 0)[0]
                    #Q[i][idx] = np.sqrt(2) * prod[i][idx]/const[i][idx]
                
                #fs.close()
                #fs = h5.File(fn, 'a')
                ##data = fs['Kinematics/Gas/Q']
                ##data[...] = Q
                #fs.create_dataset('Kinematics/Gas/Q', data = Q)  
                #fs.close()

            #if itype == 4:
                #SigmaStar = fs['StarData/SigmaStar'].value  * u.Msun * u.kpc**-2      #[Msun/kpc^2]
                #VelDisp  = fs['Kinematics/Stars/VelDisp'].value                   #[km/s]
                #prod     = VelDisp * Vcirc
                #const = (myG * r * SigmaStar * pi).to(u.km**2 * u.s**-2)
                
                #for i in range(LenGal):
                    #idx = np.where(const[i] != 0)[0]
                    #Q[i][idx] = np.sqrt(2) * prod[i][idx]/const[i][idx]
                
                #fs.close()
                #fs = h5.File(fn, 'a')
                #fs.create_dataset('Kinematics/Stars/Q', data = Q)  
                #fs.close()
    
if __name__ == '__main__':
    snap = int(sys.argv[1])
    Lbox = int(sys.argv[2])
    NP   = int(sys.argv[3])
    p    = sys.argv[4]
    if p == "REF": REF=True
    if p == "REC": REF=False

    start_time = time.time()
    ctg = GalCat(snap, NP=NP, Lbox=Lbox,REF=REF, stars=True, gas=True, dm=True)
    print("--- %2s: Tree Loaded ---" % (time.time() - start_time))
    print("--- Adding Prop ---")
    #ctg.add_Lstars()
    ctg.add_sigma(0)
    ctg.add_sigma(4)
    ctg.add_jstars()
    ctg.add_Gas_mass()
    ctg.add_SigmaGas()
    ctg.add_SigmaSFR()
    ctg.add_SigmaStar()
    ctg.add_Vcirc()
    ctg.add_ToomreParam()
    print("--- %2s: Done ---" % (time.time() - start_time))

    

