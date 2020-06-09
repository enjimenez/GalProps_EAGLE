from Snapshots import get_zstr
from read_header import read_header
import numpy as np
import astropy.units as u
from astropy.constants import G
from numpy.linalg import norm 
import h5py as h5
from compute_HI_H2_masses import HI_H2_masses
from compute_HI_H2_masses import compute_fneutral
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
        gns         = fs['SubHaloData/MainHaloID'].value
        sgns        = fs['SubHaloData/SubHaloID'].value
        Pos_cop     = fs['SubHaloData/Pos_cop'].value/self.h * self.a * 1000  # [pkpc]
        Vcom        = fs['SubHaloData/V_com'].value * self.a**(1/2.)          # [km/s]
        hmr         = fs['StarData/HalfMassRad'].value/self.h * self.a * 1000 # [pkpc]
        Spin        = fs['StarData/StarSpin']/self.h * 1000 # [pkpc km/s]
        Mstell      = fs['StarData/TotalMstell']/self.h      # [1e10 Msun]
        Lstars = np.array( [ m*s for m,s in zip(Mstell, Spin) ] )   # [1e10 Msun pkpc km/s]
        fs.close()
        
        return {'gn': gns, 'sgn':sgns, 'Pos_cop': Pos_cop, 'LogMstell': LogMstell, 'Vcom':Vcom, 'hmr': hmr, 'Lstars': Lstars} 
            
    def add_Gas_mass(self):
        COPc = self.data['Pos_cop'] * self.h * self.a**-1 * 1.e-3        
        sgns = self.data['sgn']
        
        fh = h5.File(self.BasePart + '0%i_%s_%iMpc_%i_Gas.hdf5'
                        %(self.snap, self.fend, self.Lbox, self.NP), 'r')
        MassGas     = fh['PartData/MassGas'].value/self.h    
        sgnGas      = fh['PartData/SubNum_Gas'].value
        fh.close()
        
        LenGal = len(COPc)
        MassTotal  = np.zeros(LenGal, dtype=float)
        for i in range(LenGal):
            self.TreeGas.search(center=COPc[i],radius=0.031*self.h)
            idx = self.TreeGas.getIndices()
            if idx is None: continue
            sgns_gal = sgnGas[idx]
            mask = sgns_gal == sgns[i]
            LenPart = len(sgns_gal[mask])
            if LenPart == 0: continue
        
            MassTotal[i] = np.sum(MassGas[idx][mask])
            
        fs = h5.File(Base + 'Galaxies/0%i_%s_%iMpc_%i_%sgalaxies.hdf5' 
                     %(self.snap, self.fend, self.Lbox, self.NP, self.pp), 'a')
        
        try: fs.create_dataset('GasData/MassGas_30kpc', data = MassTotal) 
        except: fs['GasData/MassGas_30kpc'][...] = MassTotal
        fs.close()

    def add_fneutral(self):
        fneutral = compute_fneutral(self.snap, self.NP, self.BasePart, a=self.a, h=self.h, Lbox=self.Lbox)
        
        fs = h5.File(self.BasePart + 'PartData/0%i_%s_%iMpc_%i_%sPartData.hdf5' 
                        %(self.snap, self.fend, self.Lbox, self.NP, self.pp), 'w')
        
        fs.create_dataset('fneutral', data=fneutral)
        fs.close()
    
    def add_HI_H2_masses(self, addPart=False):
            # Add the HI and H2 Masses for each galaxy
            COPc = self.data['Pos_cop'] * self.h * self.a**-1 * 1.e-3
            sgns = self.data['sgn']
            fgas = h5.File(self.BasePart + '0%i_%s_%iMpc_%i_Gas.hdf5'
                       %(self.snap, self.fend, self.Lbox, self.NP), 'r')
            sgnGas = fgas['PartData/SubNum_Gas'].value
            fgas.close()
                
            massHI, massH2 = HI_H2_masses(self.snap, self.NP, self.BasePart, a=self.a, h=self.h, Lbox=self.Lbox)
            if addPart:
                fs = h5.File(self.BasePart + 'PartData/0%i_%s_%iMpc_%i_%sPartData.hdf5' 
                             %(self.snap, self.fend, self.Lbox, self.NP, self.pp), 'a')
                
                fs.create_dataset('MassHI', data=massHI)
                fs.create_dataset('MassH2', data=massH2)
                fs.close()
            
            LenGal = len(COPc)
            MassHI_30kpc = np.zeros(LenGal)
            MassH2_30kpc = np.zeros(LenGal)
            
            for i in range(LenGal):
                self.TreeGas.search(center=COPc[i],radius=0.031*self.h)
                idx = self.TreeGas.getIndices()
                if idx is None: continue
                sgns_gal = sgnGas[idx]
                mask = sgns_gal == sgns[i]
                LenPart = len(sgns_gal[mask])
                if LenPart == 0: continue
            
                MassHI_gal, MassH2_gal = massHI[idx][mask], massH2[idx][mask]
                MassHI_30kpc[i] = np.sum(MassHI_gal)
                MassH2_30kpc[i] = np.sum(MassH2_gal)
            
            fs = h5.File(Base + 'Galaxies/0%i_%s_%iMpc_%i_%sgalaxies.hdf5' 
                            %(self.snap, self.fend, self.Lbox, self.NP, self.pp), 'a')

            try: fs.create_dataset('GasData/MassHI_30kpc', data = MassHI_30kpc)  
            except: fs['GasData/MassHI_30kpc'][...] = MassHI_30kpc
        
            try: fs.create_dataset('GasData/MassH2_30kpc', data = MassH2_30kpc) 
            except: fs['GasData/MassH2_30kpc'][...] = MassH2_30kpc
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
                try: fs.create_dataset('Kinematics/Gas/SF/VelDisp', data = sigmas)  
                except: fs['Kinematics/Gas/SF/VelDisp'][...] = sigmas
                
            if NSF:
                fs = h5.File(Base + 'Galaxies/0%i_%s_%iMpc_%i_%sgalaxies.hdf5' 
                             %(self.snap, self.fend, self.Lbox, self.NP,self.pp), 'a')
                
                sigmas = self.compute_sigma(itype, SF=False, NSF=True)
                try: fs.create_dataset('Kinematics/Gas/NSF/VelDisp', data = sigmas)  
                except: fs['Kinematics/Gas/NSF/VelDisp'][...] = sigmas
                
            if neutral:
                fs = h5.File(Base + 'Galaxies/0%i_%s_%iMpc_%i_%sgalaxies.hdf5' 
                             %(self.snap, self.fend, self.Lbox, self.NP,self.pp), 'a')
                
                sigmas = self.compute_sigma(itype, SF=False, neutral=True)
                try: fs.create_dataset('Kinematics/Gas/neutral/VelDisp', data = sigmas)  
                except: fs['Kinematics/Gas/neutral/VelDisp'][...] = sigmas
                
        if itype == 4:
            fs = h5.File(Base + 'Galaxies/0%i_%s_%iMpc_%i_%sgalaxies.hdf5' 
                             %(self.snap, self.fend, self.Lbox, self.NP,self.pp), 'a')
                
            sigmas = self.compute_sigma(itype)
            try: fs.create_dataset('Kinematics/Stars/VelDisp', data = sigmas) 
            except: fs['Kinematics/Stars/VelDisp'][...] = sigmas
                
        fs.close()
    
    def add_SigmaGas(self):
        # Sigmas below cold be computing in the same function...
        Pos_cop   = self.data['Pos_cop']
        COPc       = Pos_cop * self.h * self.a**-1 * 1.e-3
        BSkpc      = self.Lboxkpc
        hmr        = self.data['hmr']
        sgns       = self.data['sgn']
        
        fgas = h5.File(self.BasePart + '0%i_%s_%iMpc_%i_Gas.hdf5'
                       %(self.snap, self.fend, self.Lbox, self.NP), 'r')
        
        MassGas        = fgas['PartData/MassGas'].value/self.h * 1.e10          #[Msun] 
        PosGas         = fgas['PartData/PosGas'].value/self.h * self.a * 1000   #[pkpc]    
        sgnGas         = fgas['PartData/SubNum_Gas'].value
        
        fgas.close()
        
        LenGal = len(Pos_cop)
        Sigma  = np.zeros((LenGal, 30), dtype=float)
        
        for i in range(LenGal):
            self.TreeGas.search(center=COPc[i],radius=0.031*self.h)
            idx = self.TreeGas.getIndices()
            if idx is None: continue
            sgns_gal = sgnGas[idx]
            mask = sgns_gal == sgns[i]
            LenPart = len(sgns_gal[mask])
            if LenPart == 0: continue
            
            PosGas_gal, MassGas_gal   = PosGas[idx][mask], MassGas[idx][mask]
        
            centre = Pos_cop[i]
            r = np.array([distance(p, centre, BSkpc) for p in PosGas_gal])
            mask = np.argsort(r)
            r = r[mask]
            if r[0] > 30.0: continue 
            
            cmass = np.cumsum(MassGas_gal[mask])
            pi = 3.14159265
            SigmaCum = cmass/(pi*r*r)
            
            R = np.arange(1, 30+1, 1).astype(float)
            Sigma_tmp = np.zeros(30, dtype=float)
            
            k = 0
            while R[k] < r[0]: k+=1 
        
            for j in range(1, LenPart):
                if k == 30: break
                if R[k] > r[j]: continue
                while R[k] < r[j]:
                    Sigma_tmp[k] = (SigmaCum[j] - SigmaCum[j-1])/(r[j]-r[j-1]) * (R[k] - r[j-1]) + SigmaCum[j-1]
                    k += 1
                    if k == 30: break
            
            if k != 30: Sigma_tmp[k:] = np.ones(30-k)*cmass[-1]/R[k:]
            Sigma[i] = Sigma_tmp
            
        fs = h5.File(Base + 'Galaxies/0%i_%s_%iMpc_%i_%sgalaxies.hdf5' 
                        %(self.snap, self.fend, self.Lbox, self.NP,self.pp), 'a')
        
        try: fs.create_dataset('GasData/SigmaGas', data = Sigma)  
        except: fs['GasData/SigmaGas'][...] = Sigma
    
        
        fs.close()
    
    def add_SigmaSFR(self):
        Pos_cop    = self.data['Pos_cop']
        hmr        = self.data['hmr']
        sgns       = self.data['sgn']
        COPc       = Pos_cop * self.h * self.a**-1 * 1.e-3
        BSkpc      = self.Lboxkpc

        fgas = h5.File(self.BasePart + '0%i_%s_%iMpc_%i_Gas.hdf5'
                       %(self.snap, self.fend, self.Lbox, self.NP), 'r')
        
        SFR            = fgas['PartData/SFR'].value   
        PosGas         = fgas['PartData/PosGas'].value/self.h * self.a * 1000   #[pkpc]    
        sgnGas         = fgas['PartData/SubNum_Gas'].value
        fgas.close()
        
        LenGal   = len(Pos_cop)
        Sigma    = np.zeros((LenGal, 30), dtype=float)
        
        for i in range(LenGal):
            self.TreeGas.search(center=COPc[i],radius=0.031*self.h)
            idx = self.TreeGas.getIndices()
            if idx is None: continue
            sgns_gal = sgnGas[idx]
            mask = sgns_gal == sgns[i]
            LenPart = len(sgns_gal[mask])
            if LenPart == 0: continue
            
            PosGas_gal, SFR_gal   = PosGas[idx][mask], SFR[idx][mask]
            centre = Pos_cop[i]
            r = np.array([distance(p, centre, BSkpc) for p in PosGas_gal])
            mask = np.argsort(r)
            r = r[mask]
            if r[0] > 30.0: continue
        
            csfr = np.cumsum(SFR_gal[mask])
            pi = 3.141593
            SigmaCum = csfr/(pi*r*r)
            
            R = np.arange(1, 30+1, 1).astype(float)
            Sigma_tmp = np.zeros(30, dtype=float)
            
            k = 0
            while R[k] < r[0]: k+=1 

            for j in range(1, LenPart):
                if k == 30: break
                if R[k] > r[j]: continue
                while R[k] < r[j]:
                    Sigma_tmp[k] = (SigmaCum[j] - SigmaCum[j-1])/(r[j]-r[j-1]) * (R[k] - r[j-1]) + SigmaCum[j-1]
                    k += 1
                    if k == 30: break
            
            if k != 30: Sigma_tmp[k:] = np.ones(30-k)*csfr[-1]/R[k:]
            Sigma[i] = Sigma_tmp
                
        fs = h5.File(Base + 'Galaxies/0%i_%s_%iMpc_%i_%sgalaxies.hdf5' 
                        %(self.snap, self.fend, self.Lbox, self.NP,self.pp), 'a')
        
        try: fs.create_dataset('GasData/SigmaSFR', data = Sigma)  
        except: fs['GasData/SigmaSFR'][...] = Sigma
    
        fs.close()
        
    def add_SigmaStar(self):
        Pos_cop   = self.data['Pos_cop']     
        COPc       = Pos_cop * self.h * self.a**-1 * 1.e-3
        BSkpc      = self.Lboxkpc
        hmr        = self.data['hmr']
        sgns       = self.data['sgn']
        
        fs = h5.File(self.BasePart + '0%i_%s_%iMpc_%i_Star.hdf5'
                       %(self.snap, self.fend, self.Lbox, self.NP), 'r')
        
        MassStar        = fs['PartData/MassStar'].value/self.h * 1.e10          #[Msun] 
        PosStar         = fs['PartData/PosStar'].value/self.h * self.a * 1000   #[pkpc]    
        sgnStar         = fs['PartData/SubNum_Star'].value
        fs.close()
        
        LenGal = len(Pos_cop)
        Sigma  = np.zeros((LenGal, 30), dtype=float)
        #Sigmar50 = np.zeros((LenGal, 100), dtype=float)
        
        for i in range(LenGal):
            self.TreeStar.search(center=COPc[i],radius=0.031*self.h)
            idx = self.TreeStar.getIndices()
            if idx is None: continue
            sgns_gal = sgnStar[idx]
            mask = sgns_gal == sgns[i]
            LenPart = len(sgns_gal[mask])
            if LenPart == 0: continue
        
            PosStar_gal, MassStar_gal   = PosStar[idx][mask], MassStar[idx][mask]            
            centre = Pos_cop[i]
            r = np.array([distance(p, centre, BSkpc) for p in PosStar_gal])
            mask = np.argsort(r)
            r = r[mask]
            if r[0] > 30.0: continue
            
            cmass = np.cumsum(MassStar[mask])
            pi = 3.141593
            SigmaCum = cmass/(pi*r*r)
                    
            R = np.arange(1, 30+1, 1).astype(float)
            Sigma_tmp = np.zeros(30, dtype=float)
            
            k = 0
            while R[k] < r[0]: k+=1 
        
            for j in range(1, LenPart):
                if k == 30: break
                if R[k] > r[j]: continue
                while R[k] < r[j]:
                    Sigma_tmp[k] = (SigmaCum[j] - SigmaCum[j-1])/(r[j]-r[j-1]) * (R[k] - r[j-1]) + SigmaCum[j-1]
                    k += 1
                    if k == 30: break
            
            if k != 30: Sigma_tmp[k:] = np.ones(30-k)*cmass[-1]/R[k:]
            Sigma[i] = Sigma_tmp
            
        fs = h5.File(Base + 'Galaxies/0%i_%s_%iMpc_%i_%sgalaxies.hdf5' 
                        %(self.snap, self.fend, self.Lbox, self.NP,self.pp), 'a')
        
        try: fs.create_dataset('StarData/SigmaStar', data = Sigma)  
        except: fs['StarData/SigmaStar'][...] = Sigma
    
        fs.close()
    
    def add_Vcirc(self): 
        km_per_kpc = 3.2408e-17
        Pos_cop     = self.data['Pos_cop']
        hmr         = self.data['hmr']
        COPc        = Pos_cop * self.h * self.a**-1 * 1.e-3
        sgns        = self.data['sgn']
        BSkpc       = self.Lboxkpc
        fdm         = h5.File(self.BasePart + '0%i_%s_%iMpc_%i_DM.hdf5'
                              %(self.snap, self.fend, self.Lbox, self.NP), 'r')
        
        dm_mass       = fdm['Header/PartMassDM'].value/self.h * 1.e10
        PosDM         = fdm['PartData/PosDM'].value/self.h * self.a * 1000      #[pkpc] 
        MassDM        = np.ones(len(PosDM))*dm_mass              #[Msun] 
        sgnDM         = fdm['PartData/SubNum_DM'].value
        fdm.close()
        
        fgas = h5.File(self.BasePart + '0%i_%s_%iMpc_%i_Gas.hdf5'
                        %(self.snap, self.fend, self.Lbox, self.NP), 'r')
        
        MassGas        = fgas['PartData/MassGas'].value/self.h * 1.e10          #[Msun] 
        PosGas         = fgas['PartData/PosGas'].value/self.h * self.a * 1000   #[pkpc]   
        sgnGas         = fgas['PartData/SubNum_Gas'].value
        fgas.close()
        
        fstar = h5.File(self.BasePart + '0%i_%s_%iMpc_%i_Star.hdf5'
                        %(self.snap, self.fend, self.Lbox, self.NP), 'r')
        
        MassStar       = fstar['PartData/MassStar'].value/self.h * 1.e10         #[Msun] 
        PosStar        = fstar['PartData/PosStar'].value/self.h * self.a * 1000  #[pkpc]   
        sgnStar        = fstar['PartData/SubNum_Star'].value
        fstar.close()
        
        LenGal = len(Pos_cop)
        Vcirc  = np.zeros((LenGal, 100), dtype=float)
        kappa  = np.zeros((LenGal, 100), dtype=float)
        
        def rotation_curve(pos, mass, centre):
            r = np.array([distance(p,centre, BSkpc) for p in pos])
            mask = np.argsort(r)
            r = r[mask]
            cmass = np.cumsum(mass[mask])
            myG = G.to(u.km**2 * u.kpc * u.Msun**-1 * u.s**-2).value
            v = np.sqrt((myG*cmass) / r )
            LenPart = len(v)
            
            #dvdr = np.zeros(LenPart, dtype=float)
            dlogvdlogr = np.zeros(LenPart, dtype=float)
            logv = np.log10(v)
            logr = np.log10(r)
            
            #dvdr[0]=v[0]/r[0]
            dlogvdlogr[0] = logv[0]/logr[0]
            
            #for k in range(1, LenPart): dvdr[k] = (v[k]-v[k-1])/(r[k]-r[k-1])
            for k in range(1, LenPart): dlogvdlogr[k] = (logv[k] - logv[k-1])/(logr[k]-logr[k-1])
            
            #dvdr *= km_per_kpc 
            return r, v, dlogvdlogr
        
        for i in range(LenGal):
            
            self.TreeDM.search(center=COPc[i],radius=0.11*self.h) #TODO
            self.TreeGas.search(center=COPc[i],radius=0.11*self.h)
            self.TreeStar.search(center=COPc[i],radius=0.11*self.h)
            
            idxDM = self.TreeDM.getIndices()
            idxGas = self.TreeGas.getIndices()
            idxStar = self.TreeStar.getIndices()
            
            maskDM   = sgnDM[idxDM] == sgns[i]
            maskGas  = sgnGas[idxGas] == sgns[i]
            maskStar = sgnStar[idxStar] == sgns[i]
            
            PosDM_gal, MassDM_gal     = PosDM[idxDM][maskDM], MassDM[idxDM][maskDM]
            PosGas_gal, MassGas_gal   = PosGas[idxGas][maskGas], MassGas[idxGas][maskGas]
            PosStar_gal, MassStar_gal = PosStar[idxStar][maskStar], MassStar[idxStar][maskStar]
            
            data_pos  = np.concatenate((PosDM_gal, PosGas_gal, PosStar_gal), axis=0)
            data_mass = np.concatenate((MassDM_gal, MassGas_gal, MassStar_gal))
    
            r, v, gradv = rotation_curve(data_pos, data_mass, Pos_cop[i])
            
            LenPart = len(r) 
            Rmin, deltaR = 0, 0.05
            gradL = []            
            while r[0] > Rmin + deltaR: Rmin += deltaR
            
            j = 0
            for k in range(LenPart):
                if r[k] > Rmin + deltaR:
                    while r[k] > Rmin + deltaR: Rmin += deltaR
                    N = len(gradL)
                    gradv[j:j+N] = np.ones(N)*np.median(np.array(gradL))
                    j += N
                    gradL = []
                
                gradL.append(gradv[k])
                if k == LenPart-1: 
                    N = len(gradL)
                    gradv[j:] = np.ones(N)*np.median(np.array(gradL))
            
            R = np.arange(1, 100+1, 1).astype(float)
            Vc = np.zeros(100, dtype=float)
            Grad_Vc = np.zeros(100, dtype=float)
            kappa_gal = np.zeros(100, dtype=float)
                        
            k = 0
            for j in range(1, LenPart):
                if k == 100: break
                if R[k] > r[j]: continue
                while R[k] < r[j]:
                    Vc[k]       = (v[j] - v[j-1])/(r[j]-r[j-1]) * (R[k] - r[j-1]) + v[j-1]
                    Grad_Vc[k]  = (gradv[j] - gradv[j-1])/(r[j]-r[j-1]) * (R[k] - r[j-1]) + gradv[j-1]
                    kappa_gal[k]    = np.sqrt(2) * Vc[k]/R[k] * np.sqrt(1 + Grad_Vc[k])
                    k += 1
                    if k == 100: break
            
            Vcirc[i] = Vc
            kappa[i] = kappa_gal
            
        fs = h5.File(Base + 'Galaxies/0%i_%s_%iMpc_%i_%sgalaxies.hdf5' 
                     %(self.snap, self.fend, self.Lbox, self.NP,self.pp), 'a')
        
        try: fs.create_dataset('Kinematics/Vcirc', data = Vcirc) 
        except: fs['Kinematics/Vcirc'][...] = Vcirc

        try: fs.create_dataset('Kinematics/kappa', data = kappa)  
        except:fs['Kinematics/kappa'][...] = kappa

        fs.close()
    
    def add_jstars(self):
        Pos_cop     = self.data['Pos_cop']
        COPc        = Pos_cop * self.h * self.a**-1 * 1.e-3
        hmr         = self.data['hmr']
        Vcom        = self.data['Vcom']
        sgns        = self.data['sgn']
        LenGal      = len(Pos_cop)
        BSkpc      = self.Lboxkpc
        
        jstars    = np.zeros((LenGal,50), dtype=np.float32)
        fh = h5.File(self.BasePart + '0%i_%s_%iMpc_%i_Star.hdf5'
                    %(self.snap, self.fend, self.Lbox, self.NP), 'r')
        
        MassStar    = fh['PartData/MassStar'].value/self.h
        PosStar     = fh['PartData/PosStar'].value/self.h * self.a * 1000
        Vstar       = fh['PartData/VelStar'].value * self.a**(1/2.)
        sgnStar     = fh['PartData/SubNum_Star'].value
        fh.close()
        
        for i in range(LenGal):
            self.TreeStar.search(center=COPc[i],radius=0.051*self.h) 
            idx = self.TreeStar.getIndices()
            if idx is None: continue
            sgns_gal = sgnStar[idx]
            mask = sgns_gal == sgns[i]
            LenPart = len(sgns_gal[mask])
            if LenPart == 0: continue
        
        
        
            MassStar_gal = MassStar[idx][mask]
            PosStar_gal  = PosStar[idx][mask]
            Vstar_gal    = Vstar[idx][mask]
            centre = Pos_cop[i]
            
            delta_V = Vstar_gal - Vcom[i]
            Len_Stars = len(PosStar_gal)
            r = np.zeros(Len_Stars, dtype=float)
            for k in range(Len_Stars): r[k] = distance(PosStar_gal[k], centre, BSkpc) 
            
            tt = np.argsort(r)
            r = r[tt]
            if r[0] > 50.0: continue
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
            
            R = np.arange(1, 50+1, 1).astype(float)
            j_tmp = np.zeros(50, dtype=float)
            
            k = 0
            while R[k] < r[0]: k+=1 
            
            # Interpolation (may be a bit noisy)
            for j in range(1, LenPart):
                if k == 50: break
                if R[k] > r[j]: continue
                while R[k] < r[j]:
                    j_tmp[k] = (jj[j] - jj[j-1])/(r[j]-r[j-1]) * (R[k] - r[j-1]) + jj[j-1]
                    k += 1
                    if k == 50: break
            
            if k != 50: j_tmp[k:] = np.ones(50-k)*jj[-1]/R[k:]
            jstars[i] = j_tmp
            
        fs = h5.File(Base + 'Galaxies/0%i_%s_%iMpc_%i_%sgalaxies.hdf5' 
                             %(self.snap, self.fend, self.Lbox, self.NP, self.pp), 'a')
        
        try: fs.create_dataset('Kinematics/Stars/jstars', data = jstars)
        except: fs['Kinematics/Stars/jstars'][...] = jstars
        fs.close()
    
    def add_r(self):      
        # Computes r20, r50, r80, r90 and r50_H2
        Pos_cop     = self.data['Pos_cop']
        COPc        = Pos_cop * self.h * self.a**-1 * 1.e-3
        LenGal      = len(Pos_cop)
        BSkpc       = self.Lboxkpc
        hmr         = self.data['hmr']
        sgns        = self.data['sgn']
        
        fh = h5.File(self.BasePart + '0%i_%s_%iMpc_%i_Gas.hdf5'
                    %(self.snap, self.fend, self.Lbox, self.NP), 'r')
        PosGas     = fh['PartData/PosGas'].value/self.h * self.a * 1000
        sgnGas     = fh['PartData/SubNum_Gas'].value 
        fh.close()
        
        fh = h5.File(self.BasePart + 'PartData/0%i_%s_%iMpc_%i_%sPartData.hdf5' 
                        %(self.snap, self.fend, self.Lbox, self.NP, self.pp), 'r')
        
        MassH2Part = fh['MassH2'].value * 1.e10 #[Msun]
        fh.close()
        
        fh = h5.File(Base + 'Galaxies/0%i_%s_%iMpc_%i_%sgalaxies.hdf5'
                     %(self.snap, self.fend, self.Lbox, self.NP, self.pp), 'r')
        
        TotalH2Mass_30kpc = fh['GasData/MassH2_30kpc'].value * 1.e10
        TotalMstell       = fh['StarData/TotalMstell']/self.h
        fh.close()
        
        fh =  h5.File(self.BasePart + '0%i_%s_%iMpc_%i_Star.hdf5' 
                      %(self.snap, self.fend, self.Lbox, self.NP), 'r')
        
        PosStar     = fh['PartData/PosStar'].value/self.h * self.a * 1000
        sgnStar     = fh['PartData/SubNum_Star'].value 
        MassStar    = fh['PartData/MassStar'].value/self.h
        fh.close()
        
        r50_H2 = np.zeros(LenGal, dtype=float)
        r20 = np.zeros(LenGal, dtype=float)
        r50 = np.zeros(LenGal, dtype=float)
        r80 = np.zeros(LenGal, dtype=float)
        r90 = np.zeros(LenGal, dtype=float)
        
        for i in range(LenGal):
            centre = Pos_cop[i]
            self.TreeGas.search(center=COPc[i], radius=0.35*self.h)
            self.TreeStar.search(center=COPc[i], radius=0.35*self.h)
            idx1 = self.TreeGas.getIndices()
            idx2 = self.TreeStar.getIndices()
            if idx1 is None: continue
    
            maskGas = sgnGas[idx1] == sgns[i]
            maskStar = sgnStar[idx2] == sgns[i]
            
            PosGas_gal = PosGas[idx1][maskGas]
            MassH2_gal = MassH2Part[idx1][maskGas]
            
            PosStar_gal = PosStar[idx2][maskStar]
            MassStar_gal = MassStar[idx2][maskStar]
            
            LenStars = len(PosStar_gal)
            rstar = np.zeros(LenStars, dtype=float)
            for k in range(LenStars): rstar[k] = distance(PosStar_gal[k], centre, BSkpc) 
            tt = np.argsort(rstar)
            rstar = rstar[tt]
            
            MassCumStar = np.cumsum(MassStar_gal[tt])
            Massr20    = 0.2 * TotalMstell[i]
            Massr50    = 0.5 * TotalMstell[i]
            Massr80    = 0.8 * TotalMstell[i]
            Massr90    = 0.9 * TotalMstell[i]
            
            k = 0
            cuts = [Massr20, Massr50, Massr80, Massr90]
            rs = np.zeros(4, dtype=float)
            cut = cuts[0]
            for j, mass in zip(range(LenStars), MassCumStar): 
                if mass > cut:
                    rs[k] = rstar[j]
                    k += 1
                    if k == 4: break
                    cut = cuts[k]
            
            r20[i], r50[i], r80[i], r90[i] = rs
            
            LenGas = len(PosGas_gal)
            if LenGas == 0: continue
            rgas = np.zeros(LenGas, dtype=float)
            for k in range(LenGas): rgas[k] = distance(PosGas_gal[k], centre, BSkpc) 
            tt = np.argsort(rgas)
            rgas = rgas[tt]        
        
            MassCumH2    = np.cumsum(MassH2_gal[tt])
            HalfMassH2   = 0.5 * TotalH2Mass_30kpc[i]
            if HalfMassH2< 1.0e6: continue  #~Mass of 1 gas particle
            
            for j, mass in zip(range(LenGas), MassCumH2):
                if mass > HalfMassH2: break
            
            r50_H2[i] = rgas[j]
        
        fs = h5.File(Base + 'Galaxies/0%i_%s_%iMpc_%i_%sgalaxies.hdf5' 
                             %(self.snap, self.fend, self.Lbox, self.NP, self.pp), 'a')
        
        try: fs.create_dataset('GasData/r50_H2', data = r50_H2)
        except: fs['GasData/r50_H2'][...] = r50_H2
        
        try: fs.create_dataset('StarData/r20', data = r20)
        except: fs['StarData/r20'][...] = r20
        
        try: fs.create_dataset('StarData/r50', data = r50)
        except: fs['StarData/r50'][...] = r50
        
        try: fs.create_dataset('StarData/r80', data = r80)
        except: fs['StarData/r80'][...] = r80
        
        try: fs.create_dataset('StarData/r90', data = r90)
        except: fs['StarData/r90'][...] = r90
        fs.close()
    
    def compute_sigma(self, itype, NumP=4, SF=True, NSF=False, neutral=False):
        # should be in a separate function # Fix this function. 
        BSkpc        = self.Lboxkpc
        Pos_cop      = self.data['Pos_cop']
        COPc         = Pos_cop * self.h * self.a**-1 * 1.e-3
        Vcom         = self.data['Vcom']
        L            = self.data['Lstars']   # [pkpc km/s 1e10 Msun]
        hmr          = self.data['hmr']
        sgns         = self.data['sgn']
        LenGal       = len(Pos_cop)
        sigma        = np.zeros((LenGal,30), dtype=np.float32) 
        
        if itype == 0:
            fh = h5.File(self.BasePart + '0%i_%s_%iMpc_%i_Gas.hdf5'
                        %(self.snap, self.fend, self.Lbox, self.NP), 'r')
        
            MassPart     = fh['PartData/MassGas'].value/self.h
            PosPart      = fh['PartData/PosGas'].value/self.h * self.a * 1000
            VPart        = fh['PartData/VelGas'].value * self.a**(1/2.)
            SFR          = fh['PartData/SFR'].value     # Msun/yr
            Density      = fh['PartData/Density'].value 
            Entropy      = fh['PartData/Entropy'].value
            sgnGas       = fh['PartData/SubNum_Gas'].value
            UnitPressure = fh['Constants/UnitPressure'].value
            UnitDensity  = fh['Constants/UnitDensity'].value
            gamma        = fh['Constants/gamma'].value
            
            Density  = Density * self.h**2 * self.a**-3  * UnitDensity
            Entropy  = Entropy * self.h**(2-2*gamma) * UnitPressure * UnitDensity**(-1*gamma)
            Pressure = Entropy*Density**gamma
            del Entropy
            fh.close()
            
            if neutral:
                fs = h5.File(self.BasePart + 'PartData/0%i_%s_%iMpc_%i_%sPartData.hdf5' 
                            %(self.snap, self.fend, self.Lbox, self.NP, self.pp), 'r')
                
                fneutral = fs['fneutral'].value
                fs.close()
            
            
        if itype == 4:
            fh = h5.File(self.BasePart + '0%i_%s_%iMpc_%i_Star.hdf5'
                    %(self.snap, self.fend, self.Lbox, self.NP), 'r')

            MassPart    = fh['PartData/MassStar'].value/self.h
            PosPart     = fh['PartData/PosStar'].value/self.h * self.a * 1000
            VPart       = fh['PartData/VelStar'].value * self.a**(1/2.)
            sgnStar     = fh['PartData/SubNum_Star'].value
            fh.close()
                
        for i in range(LenGal):
            if itype == 0:
                self.TreeGas.search(center=COPc[i],radius=0.035*self.h)
                idx = self.TreeGas.getIndices()
                if idx is None: continue
                sgns_gal = sgnGas[idx]
                masksgn = sgns_gal == sgns[i]
                if len(sgns_gal[masksgn]) == 0: continue
                
                if SF: 
                    SFR_gal = SFR[idx][masksgn]
                    mask   = SFR_gal > 0
                    LenPart = len(SFR_gal[mask])
                    if LenPart < NumP: continue
                if NSF:
                    SFR_gal = SFR[idx][masksgn]
                    mask   = SFR_gal == 0
                    LenPart = len(SFR_gal[mask])
                    if LenPart < NumP: continue
                if neutral:
                    fneutral_gal = fneutral[idx][masksgn]
                    mask   = fneutral_gal > 1.0e-5
                    LenPart = len(fneutral_gal[mask])
                    if LenPart < NumP: continue
                    
                Pressure_gal     = Pressure[idx][masksgn][mask]
                Density_gal      = Density[idx][masksgn][mask]
                MassPart_gal = MassPart[idx][masksgn][mask]
                PosPart_gal  = PosPart[idx][masksgn][mask]
                VPart_gal    = VPart[idx][masksgn][mask] 
            
            if itype == 4:
                self.TreeStar.search(center=COPc[i],radius=0.035*self.h)
                idx = self.TreeStar.getIndices()
                if idx is None: continue
                sgns_gal = sgnStar[idx]
                masksgn = sgns_gal == sgns[i]
                LenPart = len(sgns_gal[masksgn])
                if len(sgns_gal[masksgn]) == 0: continue
                
                MassPart_gal = MassPart[idx][masksgn]
                PosPart_gal  = PosPart[idx][masksgn]
                VPart_gal    = VPart[idx][masksgn]
                LenPart      = len(MassPart_gal)
            
            centre = Pos_cop[i]
            vcom   = Vcom[i]
            
            # Compute sigma
            r = np.zeros(LenPart, dtype=float)
            for k in range(LenPart): r[k] = distance(PosPart_gal[k], centre, BSkpc) 
            tt = np.argsort(r)
            r = r[tt]
            if r[0] > 30.0: continue
            
            delta_V = VPart_gal[tt] - vcom
            MassPart_sorted = MassPart_gal[tt]
            e_gas = np.zeros(LenPart, dtype=float)
            
            if itype == 0: 
                sigmaP  = (np.sqrt(Pressure_gal/Density_gal))[tt] * 1e-5  # km/s  
                for k in range(LenPart): e_gas[k] = MassPart_sorted[k] * ( (np.dot(delta_V[k], L[i])/float(norm(L[i])))**2. + (sigmaP[k]/3.)**2.)
            
            if itype == 4:
                for k in range(LenPart): e_gas[k] = MassPart_sorted[k] * ( (np.dot(delta_V[k], L[i])/float(norm(L[i])))**2. )
                
            
            sigma_val = np.sqrt( np.cumsum(e_gas)/np.cumsum(MassPart_sorted) )
            
            
            R = np.arange(1, 30+1, 1).astype(float)
            Sigma_tmp = np.zeros(30, dtype=float)
            
            k = 0
            while R[k] < r[0]: k+=1 
        
            for j in range(1, LenPart):
                if k == 30: break
                if R[k] > r[j]: continue
                while R[k] < r[j]:
                    Sigma_tmp[k] = (sigma_val[j] - sigma_val[j-1])/(r[j]-r[j-1]) * (R[k] - r[j-1]) + sigma_val[j-1]
                    k += 1
                    if k == 30: break
            
            if k != 30: Sigma_tmp[k:] = np.ones(30-k)*sigma_val[-1]
            sigma[i] = Sigma_tmp
        
        return sigma
    
    def add_ToomreParam(self):
        pi = 3.1415926
        km_per_kpc = 3.2408e-17
        myG = G.to(u.kpc**2 * u.km * u.Msun**-1 * u.s**-2).value
        
        fs = h5.File(Base + 'Galaxies/0%i_%s_%iMpc_%i_%sgalaxies.hdf5'
                     %(self.snap, self.fend, self.Lbox, self.NP,self.pp), 'r+')
    
        kappa       = fs['Kinematics/kappa'].value[:,:30] * km_per_kpc  # [s^-1]
        VelDispGas  = fs['Kinematics/Gas/SF/VelDisp'].value # [km/s]
        VelDispStar = fs['Kinematics/Stars/VelDisp'].value  # [km/s]
        SigmaGas    = fs['GasData/SigmaGas'].value          # [Msun kpc^-2]
        SigmaStar   = fs['StarData/SigmaStar'].value        # [Msun kpc^-2]
        rmax = 30
        
        LenGal   = len(self.data['Pos_cop'])
        Qgas     = np.zeros((LenGal,rmax), dtype=float)
        Qstar    = np.zeros((LenGal,rmax), dtype=float)
        Qnet     = np.zeros((LenGal,rmax), dtype=float)
        
        for i in range(LenGal):
            Qgas[i]  = kappa[i]*VelDispGas[i]/(pi*myG*SigmaGas[i])
            Qstar[i] = kappa[i]*VelDispStar[i]/(pi*myG*SigmaStar[i])
            
            W = 2.0*VelDispGas[i]*VelDispStar[i]/(VelDispGas[i]**2. + VelDispStar[i]**2.)
            for k in range(rmax):
                qgas, qstar = Qgas[i][k], Qstar[i][k]
                if qstar >= qgas: Qnet[i][k] = (W[k]/qstar + qgas**-1)**-1
                if qgas >= qstar: Qnet[i][k] = (W[k]/qgas + qstar**-1)**-1
            
        try: fs.create_dataset('Kinematics/Qgas',  data = Qgas)
        except: fs['Kinematics/Qgas'][...] = Qgas
        
        try: fs.create_dataset('Kinematics/Qstar',  data = Qstar)
        except: fs['Kinematics/Qstar'][...] = Qstar
        
        try: fs.create_dataset('Kinematics/Qnet',  data = Qnet)
        except: fs['Kinematics/Qnet'][...] = Qnet

        fs.close()
    
if __name__ == '__main__':
    snap = int(sys.argv[1])
    Lbox = int(sys.argv[2])
    NP   = int(sys.argv[3])
    p    = sys.argv[4]
    if p == "REF": REF=True
    if p == "REC": REF=False

    start_time = time.time()
    ctg = GalCat(snap, NP=NP, Lbox=Lbox,REF=REF, gas=True, stars=True, dm=True)
    print("--- %2s: Tree Loaded ---" % (time.time() - start_time))
    print("--- Adding Prop ---")
    ctg.add_r()
    #ctg.add_fneutral()
    ctg.add_HI_H2_masses()
    #ctg.add_sigma(0)
    #ctg.add_sigma(0, SF=False, NSF=True)
    #ctg.add_sigma(0, SF=False, neutral=True)
    #ctg.add_sigma(4)
    #ctg.add_jstars()
    #ctg.add_Gas_mass()
    #ctg.add_SigmaGas()
    #ctg.add_SigmaSFR()
    #ctg.add_SigmaStar()
    ctg.add_Vcirc()
    #ctg.add_ToomreParam()
    print("--- %2s: Done ---" % (time.time() - start_time))

    

