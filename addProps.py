from prody.kdtree.kdtree import KDTree
import numpy as np
import h5py as h5
import getMass
import getVelDisp
import getSFR
import time
import sys

snips = False
cluster = 'ozstar' # change to False if run in local computer

if snips: from Snipshots import get_zstr
else: from Snapshots import get_zstr

class GalCat:
    def __init__(self, snap, NP, L, phy, SF=False, gas=False, stars=False, dm=False):
        self.snap = str(snap).zfill(3)
        self.fend = get_zstr(snap)
        self.NP = NP
        self.L = L

        if phy == 'sSNII':  phy = 'StrongSNII'
        if phy == 'wSNII':  phy = 'WeakSNII'
        if phy == 'nofeed': phy = 'NoFeedback'
        if phy == 'noAGN':  phy = 'NOAGN'
        if phy == 'eos1':   phy = 'EOS1p000'
        if phy == 'eos5/3': phy = 'EOS1p666'
        if phy == 'REF':    phy = 'REFERENCE'
        if phy == 'REC':    phy = 'RECALIBRATED'
        if phy == 'FBconst': phy = 'FBconst'
        
        box = 'L%sN%s' %(str(L).zfill(4), str(NP).zfill(4))
        
        if type(cluster) == str: 
            if cluster == 'ozstar': self.Base = '/fred/oz009/ejimenez/'
            if cluster == 'hyades': self.Base = '/mnt/su3ctm/ejimenez/'
            
            if snips: 
                self.BasePart = self.Base + '%s/snipshots/' %box
                self.BaseGal = self.Base + 'Galaxies/%s/snipshots/' %box 
            else:  
                # Eagle Variations for 25Mpc and 50Mpc box only
                if phy == 'RECALIBRATED': 
                    self.BasePart = self.Base + '%s/%s/' %(box, phy)
                    self.BaseGal = self.Base + 'Galaxies/%s/%s/' %(box, phy)
                else:
                    self.BasePart = self.Base + '%s/EagleVariation_%s/' %(box, phy) if phy != 'REFERENCE' and phy != 'FBconst' else self.Base + '%s/%s/' %(box, phy)
                    self.BaseGal = self.Base + 'Galaxies/%s/EagleVariation_%s/' %(box, phy) if phy != 'REFERENCE' and phy != 'FBconst' else self.Base + 'Galaxies/%s/%s/' %(box, phy)
    
        else: 
            self.Base = '/home/esteban/Documents/EAGLE/'
            self.BasePart = self.Base + 'processed_data/'
            self.BaseGal = self.Base + 'Galaxies/%s/%s/' %(box,phy)
            #self.BaseOut = self.Base + 'Data/%s/' %box
        
        with h5.File(self.BaseGal + '%s_%s_%iMpc_%i_galaxies_all.hdf5' %(self.snap, self.fend, L, self.NP), 'r') as f:
            self.a    = f['Header/a'][()]               # Scale factor
            self.h    = f['Header/h'][()]               # h
            self.Lbox = f['Header/BoxSize'][()]/self.h     # L [cMpc]
                        
        BS  = self.Lbox
        
        # Inititalize the Trees
        self.data = self.get_data() 
        
        if SF:
            # SF=True when analyzing SF particles ONLY 
            fh = h5.File(self.BasePart + '%s_%s_%iMpc_%i_Gas.hdf5'
                            %(self.snap, self.fend, self.L, self.NP), 'r')
            SFR     = fh['PartData/SFR'][()]
            mask = SFR > 0.0
            del SFR
            PosSF  = fh['PartData/PosGas'][()][mask]/self.h #[cMpc]
            fh.close()
            self.TreeSF  = KDTree(unitcell=np.asarray([BS,BS,BS]), coords = PosSF, bucketsize=10)
            del PosSF
            
        if gas: 
            fh = h5.File(self.BasePart + '%s_%s_%iMpc_%i_Gas.hdf5'
                            %(self.snap, self.fend, self.L, self.NP), 'r')
            PosGas  = fh['PartData/PosGas'][()]/self.h #[cMpc]
            fh.close()
            self.TreeGas  = KDTree(unitcell=np.asarray([BS,BS,BS]), coords = PosGas, bucketsize=10)
            del PosGas
            
        if stars: 
            fh = h5.File(self.BasePart + '%s_%s_%iMpc_%i_Star.hdf5'
                            %(self.snap, self.fend, self.L, self.NP), 'r')
            PosStar  = fh['PartData/PosStar'][()]/self.h #[cMpc]
            fh.close()
            self.TreeStar  = KDTree(unstarsitcell=np.asarray([BS,BS,BS]), coords = PosStar, bucketsize=10) 
            del PosStar
            
        if dm: 
            fh = h5.File(self.BasePart + '%s_%s_%iMpc_%i_DM.hdf5'
                            %(self.snap, self.fend, self.L, self.NP), 'r')
            PosDM  = fh['PartData/PosDM'][()]/self.h #[cMpc]
            fh.close()
            self.TreeDM  = KDTree(unitcell=np.asarray([BS,BS,BS]), coords = PosDM, bucketsize=10) 
            del PosDM
            
    def get_data (self):
        fs = h5.File(self.BaseGal + '%s_%s_%iMpc_%i_galaxies_all.hdf5' %(self.snap, self.fend, self.L, self.NP), 'r')
    
        TotalMstell = fs['StarData/TotalMstell'][()]/self.h
        LogMstell   = np.log10(TotalMstell +1e-10 ) + 10.0
        gns         = fs['SubHaloData/MainHaloID'][()]
        sgns        = fs['SubHaloData/SubHaloID'][()]
        Pos_cop     = fs['SubHaloData/Pos_cop'][()]/self.h  # [cMpc]
        Pos_com     = fs['SubHaloData/Pos_com'][()]/self.h  # [cMpc]
        Vcom        = fs['SubHaloData/V_com'][()]           # peculiar velocity, no a 
        SpinStar    = fs['StarData/StarSpin'][()]/self.h * 1000.0 # [pkpc km/s]
        SpinSF      = fs['GasData/SF/Spin'][()]/self.h * 1000.0
        fs.close()
        del TotalMstell
        
        params = {'a': self.a, 'h': self.h, 'snap': self.snap, 'fend': self.fend,
                  'NP': self.NP, 'Lbox': self.Lbox, 'L': self.L, 'Base': self.Base, 'BasePart': self.BasePart }
        
        return {'gn': gns, 'sgn':sgns, 'Pos_cop': Pos_cop, 'Pos_com': Pos_com, 'LogMstell': LogMstell,
                'Vcom':Vcom, 'SpinSF': SpinSF, 'SpinStar': SpinStar, 'params':params} 
        
    def add_GasMass(self, sf=False, rmax=30):
        fs = h5.File(self.BaseGal + '%s_%s_%iMpc_%i_galaxies.hdf5' %(self.snap, self.fend, self.Lbox, self.NP), 'a')
        
        if sf: 
            GasMass = getMass.Aperture(itype=0, data=self.data, Tree=self.TreeSF, ap=rmax, sf=True)
            try: fs.create_dataset('GasData/MassSF_%ikpc'%rmax, data = GasMass) 
            except: fs['GasData/MassSF_%ikpc' %rmax][...] = GasMass
        
        else:
            GasMass = getMass.Aperture(itype=0, data=self.data, Tree=self.TreeGas, ap=rmax)
            try: fs.create_dataset('GasData/MassGas_%ikpc'%rmax, data = GasMass) 
            except: fs['GasData/MassGas_%ikpc' %rmax][...] = GasMass
        fs.close()
        
    def add_StarMass(self, rmax=30):
        StarMass = getMass.Aperture(itype=4, data=self.data, Tree=self.TreeStar, ap=rmax)
        
        fs = h5.File(self.BaseGal + '%s_%s_%iMpc_%i_galaxies.hdf5' 
                        %(self.snap, self.fend, self.Lbox, self.NP), 'a')
        
        try: fs.create_dataset('StarData/MassStar_%ikpc'%rmax, data = StarMass) 
        except: fs['StarData/MassStar_%ikpc' %rmax][...] = StarMass
        fs.close()
        
    # TODO compute all at once using a list
    def add_VelDispGas(self, rmax=50, SF=True, All=False, feedback=False, nofeed=False):
        fs = h5.File(self.BaseGal + '%s_%s_%iMpc_%i_galaxies_all.hdf5' %(self.snap, self.fend, self.Lbox, self.NP), 'a')
        
        if All:
            if feedback:
                sigmas, NumPart = getVelDisp.Gas(self.data, self.TreeGas, rmax=rmax, SF=False, All=True, feedback=True); ff = '_Feed'
            elif nofeed:
                sigmas, NumPart = getVelDisp.Gas(self.data, self.TreeGas, rmax=rmax, SF=False, All=True, nofeed=True); ff = '_noFeed'                
            else:
                sigmas, sigmas_nth, NumPart = getVelDisp.Gas(self.data, self.TreeGas, rmax=rmax, SF=False, All=True, return_noth=True); ff = ''
                try: fs.create_dataset('Kinematics/Gas/VelDisp_nth', data = sigmas_nth)  
                except: fs['Kinematics/Gas/VelDisp_nth'][...] = sigmas_nth
    
            try: fs.create_dataset('Kinematics/Gas/VelDisp%s' %ff, data = sigmas)  
            except: fs['Kinematics/Gas/VelDisp%s' %ff][...] = sigmas
            
            try: fs.create_dataset('Kinematics/Gas/NumPart%s' %ff, data = NumPart)  
            except: fs['Kinematics/Gas/NumPart%s' %ff][...] = NumPart
            
        if SF:
            sigmas, sigmas_nth, NumPart = getVelDisp.Gas(data=self.data, Tree=self.TreeSF, rmax=rmax, return_noth=True)
                
            try: fs.create_dataset('Kinematics/Gas/SF/VelDisp', data = sigmas)  
            except: fs['Kinematics/Gas/SF/VelDisp'][...] = sigmas
            
            try: fs.create_dataset('Kinematics/Gas/SF/VelDisp_nth', data = sigmas_nth)  
            except: fs['Kinematics/Gas/SF/VelDisp_nth'][...] = sigmas_nth
            
            try: fs.create_dataset('Kinematics/Gas/SF/NumPart', data = NumPart)  
            except: fs['Kinematics/Gas/SF/NumPart'][...] = NumPart
            
        fs.close()
            
    def add_VelDispStar(self, rmax=50):
        fs = h5.File(self.BaseGal + '%s_%s_%iMpc_%i_galaxies.hdf5' %(self.snap, self.fend, self.Lbox, self.NP), 'a')
        sigmas, NumPart = getVelDisp.Stars(data=self.data, Tree=self.TreeStar, rmax=rmax)
        
        try: fs.create_dataset('Kinematics/Stars/VelDisp', data = sigmas) 
        except: fs['Kinematics/Stars/VelDisp'][...] = sigmas
        
        try: fs.create_dataset('Kinematics/Stars/NumPart', data = NumPart)  
        except: fs['Kinematics/Stars/NumPart'][...] = NumPart
        fs.close()
        
    def add_VelDispDM(self, rmax=50):
        fs = h5.File(self.BaseGal + '%s_%s_%iMpc_%i_galaxies.hdf5' %(self.snap, self.fend, self.Lbox, self.NP), 'a')
        sigmas, NumPart = getVelDisp.DM(data=self.data, Tree=self.TreeDM, rmax=rmax)
        
        try: fs.create_dataset('Kinematics/DM/VelDisp', data = sigmas) 
        except: fs['Kinematics/DM/VelDisp'][...] = sigmas
        
        try: fs.create_dataset('Kinematics/DM/NumPart', data = NumPart)  
        except: fs['Kinematics/DM/NumPart'][...] = NumPart
        fs.close()
    
    def add_StellarMassRadii(self):
        fs = h5.File(self.BaseGal + '%s_%s_%iMpc_%i_galaxies.hdf5'
                     %(self.snap, self.fend, self.Lbox, self.NP), 'a')
        
        radii = getRadii.Stellar(data=self.data, Tree=self.TreeSF)
        
        try: fs.create_dataset('StarData/r20', data = radii['r20'])
        except: fs['StarData/r20'][...] = radii['r20']
        
        try: fs.create_dataset('StarData/r50', data = radii['r50'])
        except: fs['StarData/r50'][...] = radii['r50']
        
        try: fs.create_dataset('StarData/r80', data = radii['r80'])
        except: fs['StarData/r80'][...] = radii['r80']
        
        try: fs.create_dataset('StarData/r90', data = radii['r90'])
        except: fs['StarData/r90'][...] = radii['r90']
        fs.close()
        
    def add_GasTransferFractions(self):
        #print('TODO')
        getTransfer.Gas(data=self.data, Tree=self.TreeSF) 
        
    def add_SFRav(self):
        fs = h5.File(self.BaseGal + '%s_%s_%iMpc_%i_galaxies.hdf5' %(self.snap, self.fend, self.Lbox, self.NP), 'a')
        SFR_av = getSFR.getSFR(data=self.data, Tree=self.TreeStar)
        
        try: fs.create_dataset('StarData/SFR_30Myr_av', data = SFR_av) 
        except: fs['StarData/SFR_30Myr_av'][...] = SFR_av
        fs.close()
    
if __name__ == '__main__':
    snap = int(sys.argv[1])
    L = int(sys.argv[2])
    NP  = int(sys.argv[3])
    phy  = sys.argv[4]
    
    start_time = time.time()
    ctg = GalCat(snap, NP=NP, L=L, phy=phy, SF=True)
    print("--- %2s: Tree Loaded ---" % (time.time() - start_time))
    print("--- Adding Prop ---")
    #ctg.add_SFRav()
    #ctg.add_GasTransferFractions()
    ctg.add_VelDispGas()
    #ctg.add_GasMass()
    #ctg.add_GasMass(sf=True)
    #ctg.add_StarMass()
    #ctg.add_VelDispGas(SF=False, All=True)
    #ctg.add_VelDispGas(SF=False, All=True, feedback=True)
    #ctg.add_VelDispGas(SF=False, All=True, nofeed=True)
    #ctg.add_VelDispStar()
    #ctg.add_VelDispDM()
    print("--- %2s: Done ---" % (time.time() - start_time))

    

