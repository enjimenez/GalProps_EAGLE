from scipy.spatial import cKDTree as KDTree
import getRotationAxis
import numpy as np
import h5py as h5
import pickle
import getMass
import getVelDisp
import getSFR
import time
import sys

snips = False
cluster =  'ozstar' # change to False if run in local computer

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
        
        with h5.File(self.BaseGal + '%s_%s_%iMpc_%i_galaxies.hdf5' %(self.snap, self.fend, L, self.NP), 'r') as f:
            self.a    = f['Header/a'][()]               # Scale factor
            self.h    = f['Header/h'][()]               # h
            self.Lbox = f['Header/BoxSize'][()]/self.h     # L [cMpc]
                        
        BS  = self.Lbox
        
        # Inititalize the Trees
        self.data = self.get_data() 
        
        if SF:
            try:
                with open(self.BasePart + 'Trees/TreeSF_%s.pickle' %self.snap, 'rb') as f:
                    self.TreeSF = pickle.load(f)      
            
            except:
                with h5.File(self.BasePart + '%s_%s_%iMpc_%i_Gas.hdf5'
                            %(self.snap, self.fend, self.L, self.NP), 'r') as fh:
                    SFR     = fh['PartData/SFR'][()]
                    mask = SFR > 0.0
                    PosSF  = fh['PartData/PosGas'][()][mask]/self.h #[cMpc]
                    del SFR
                
                self.TreeSF  = KDTree(PosSF, leafsize=10, boxsize=self.Lbox)
                del PosSF
                with open(self.BasePart + 'Trees/TreeSF_%s.pickle' %self.snap, 'wb') as f:
                    pickle.dump(self.TreeSF, f, protocol=4)
                    
        if gas: 
            try:
                with open(self.BasePart + 'Trees/TreeGas_%s.pickle' %self.snap, 'rb') as f:
                    self.TreeGas = pickle.load(f)       
            
            except:
                with h5.File(self.BasePart + '%s_%s_%iMpc_%i_Gas.hdf5'
                            %(self.snap, self.fend, self.L, self.NP), 'r') as fh:
                    PosGas  = fh['PartData/PosGas'][()]/self.h 
                    
                self.TreeGas  = KDTree(PosGas, leafsize=10, boxsize=self.Lbox)
                del PosGas
                with open(self.BasePart + 'Trees/TreeGas_%s.pickle' %self.snap, 'wb') as f:
                    pickle.dump(self.TreeGas, f, protocol=4) 
            
        if stars:
            try:
                with open(self.BasePart + 'Trees/TreeStar_%s.pickle' %self.snap, 'rb') as f:
                    self.TreeStar = pickle.load(f)        
            except:
                with h5.File(self.BasePart + '%s_%s_%iMpc_%i_Star.hdf5'
                            %(self.snap, self.fend, self.L, self.NP), 'r') as fh:
                    PosStar  = fh['PartData/PosStar'][()]/self.h 
                
                self.TreeStar  = KDTree(PosStar, leafsize=10, boxsize=self.Lbox)
                del PosStar
                with open(self.BasePart + 'Trees/TreeStar_%s.pickle' %self.snap, 'wb') as f:
                    pickle.dump(self.TreeStar, f, protocol=4)           
            
        if dm:
            try:
                with open(self.BasePart + 'Trees/TreeDM_%s.pickle' %self.snap, 'rb') as f:
                    self.TreeDM = pickle.load(f)      
            except:
                with h5.File(self.BasePart + '%s_%s_%iMpc_%i_DM.hdf5'
                            %(self.snap, self.fend, self.L, self.NP), 'r') as fh:
                    PosDM  = fh['PartData/PosDM'][()]/self.h 
                
                self.TreeDM  = KDTree(PosDM, leafsize=10, boxsize=self.Lbox)
                del PosDM
                with open(self.BasePart + 'Trees/TreeDM_%s.pickle' %self.snap, 'wb') as f:
                    pickle.dump(self.TreeDM, f, protocol=4)          
            
    def get_data (self):
        fs = h5.File(self.BaseGal + '%s_%s_%iMpc_%i_galaxies.hdf5' %(self.snap, self.fend, self.L, self.NP), 'r')
    
        TotalMstell = fs['StarData/TotalMstell'][()]/self.h
        LogMstell   = np.log10(TotalMstell +1e-10 ) + 10.0
        gns         = fs['SubHaloData/MainHaloID'][()]
        sgns        = fs['SubHaloData/SubHaloID'][()]
        Pos_cop     = fs['SubHaloData/Pos_cop'][()]/self.h  # [cMpc]
        Pos_com     = fs['SubHaloData/Pos_com'][()]/self.h  # [cMpc]
        Vcom        = fs['SubHaloData/V_com'][()]           # peculiar velocity already, no a 
        SpinStar    = fs['StarData/StarSpin'][()]/self.h * 1000.0 # [pkpc km/s]
        SpinSF      = fs['GasData/SF/Spin'][()]/self.h * 1000.0
        #RotAxis_30kpc = fs['GasData/RotAxes/L_30kpc'][()] * 1e3
        #RotAxis_70kpc = fs['GasData/RotAxes/L_70kpc'][()] *1e3
        #RotAxis_100kpc = fs['GasData/RotAxes/L_100kpc'][()] *1e3
        fs.close()
        del TotalMstell
        
        params = {'a': self.a, 'h': self.h, 'snap': self.snap, 'fend': self.fend,
                  'NP': self.NP, 'Lbox': self.Lbox, 'L': self.L, 'Base': self.Base, 
                  'BasePart': self.BasePart, 'BaseGal': self.BaseGal }
        
        return {'gn': gns, 'sgn':sgns, 'Pos_cop': Pos_cop, 'Pos_com': Pos_com, 'LogMstell': LogMstell,
                'Vcom':Vcom, 'SpinSF': SpinSF, 'SpinStar': SpinStar, 'params':params} 
    
    def addL(self, itype):
        Aps = [5,7,10,20,30,70,100]
        if itype == 0:
            Larr = getRotationAxis.getL(0, self.data, self.TreeGas)
            st = 'GasData'
        if itype == 4:
            Larr = getRotationAxis.getL(4, self.data, self.TreeStar)
            st = 'StarData'
        
        with h5.File(self.BaseGal + '%s_%s_%iMpc_%i_galaxies.hdf5' %(self.snap, self.fend, self.Lbox, self.NP), 'a') as f:
            for j, Ap in enumerate(Aps):
                try: f.create_dataset('%s/RotAxes/L_%ikpc' %(st, Ap), data = Larr[j])
                except: f['%s/RotAxes/L_%ikpc' %(st, Ap)][...] = Larr[j]
                     
    def add_GasMass(self, sf=False, rmax=30, Mult=False):
        fs = h5.File(self.BaseGal + '%s_%s_%iMpc_%i_galaxies.hdf5' %(self.snap, self.fend, self.Lbox, self.NP), 'a')
        
        if Mult:
            radii = [3,5,7,10,20,30]
            GasMassAp = getMass.ApertureMult(itype=0, data=self.data, Tree=self.TreeGas, aps=radii)
            for j, rad in enumerate(radii):
                try: fs.create_dataset('GasData/Aperture/MassGas_%ikpc' %rad, data = GasMassAp[:,j]) 
                except: fs['GasData/Aperture/MassGas_%ikpc' %rad][...] = GasMassAp[:,j]
            
        elif sf: 
            GasMass = getMass.Aperture(itype=0, data=self.data, Tree=self.TreeSF, ap=rmax, sf=True)
            try: fs.create_dataset('GasData/MassSF_%ikpc'%rmax, data = GasMass) 
            except: fs['GasData/MassSF_%ikpc' %rmax][...] = GasMass
        
        else:
            GasMass = getMass.Aperture(itype=0, data=self.data, Tree=self.TreeGas, ap=rmax)
            try: fs.create_dataset('GasData/MassGas_%ikpc'%rmax, data = GasMass) 
            except: fs['GasData/MassGas_%ikpc' %rmax][...] = GasMass
        fs.close()
        
    def add_StarMass(self, rmax=30, Mult=False):
        fs = h5.File(self.BaseGal + '%s_%s_%iMpc_%i_galaxies.hdf5' %(self.snap, self.fend, self.Lbox, self.NP), 'a')

        if Mult:
            radii = [3,5,7,10,20,30]
            StarMassAp = getMass.ApertureMult(itype=4, data=self.data, Tree=self.TreeStar, aps=radii)
            for j, rad in enumerate(radii):
                try: fs.create_dataset('GasData/Aperture/MassStar_%ikpc' %rad, data = StarMassAp[:,j]) 
                except: fs['GasData/Aperture/MassStar_%ikpc' %rad][...] = StarMassAp[:,j]
    
        else:
            StarMass = getMass.Aperture(itype=4, data=self.data, Tree=self.TreeStar, ap=rmax)
            try: fs.create_dataset('StarData/MassStar_%ikpc'%rmax, data = StarMass) 
            except: fs['StarData/MassStar_%ikpc' %rmax][...] = StarMass
        fs.close()
        
    # TODO compute all at once using a list
    def add_VelDispGas(self, rmax=50, SF=True, All=False, feedback=False, nofeed=False):
        fs = h5.File(self.BaseGal + '%s_%s_%iMpc_%i_galaxies.hdf5' %(self.snap, self.fend, self.Lbox, self.NP), 'a')
        
        if All:
            if feedback:
                sigmas, NumPart = getVelDisp.Gas(self.data, self.TreeGas, rmax=rmax, SF=False, All=True, feedback=True); ff = '_Feed'
            elif nofeed:
                sigmas, NumPart = getVelDisp.Gas(self.data, self.TreeGas, rmax=rmax, SF=False, All=True, nofeed=True); ff = '_noFeed'                
            else:
                sigmas, NumPart = getVelDisp.Gas(self.data, self.TreeGas, rmax=rmax, SF=False, All=True, return_noth=True); ff = ''
                for j,Lang in enumerate([5,7,10,20,30,70,100]):
                    try: fs.create_dataset('Kinematics/Gas/VelDisp_L%i' %Lang, data = sigmas[j])  
                    except: fs['Kinematics/Gas/VelDisp_L%i' %Lang][...] = sigmas[j]
                    
                #try: fs.create_dataset('Kinematics/Gas/VelDisp_nth', data = sigmas_nth)  
                #except: fs['Kinematics/Gas/VelDisp_nth'][...] = sigmas_nth
    
            #try: fs.create_dataset('Kinematics/Gas/VelDisp%s' %ff, data = sigmas)  
            #except: fs['Kinematics/Gas/VelDisp%s' %ff][...] = sigmas
            
            try: fs.create_dataset('Kinematics/Gas/NumPart%s_f' %ff, data = NumPart)  
            except: fs['Kinematics/Gas/NumPart%s_f' %ff][...] = NumPart
            
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
        
        try: fs.create_dataset('StarData/SFR_30Myr', data = SFR_av) 
        except: fs['StarData/SFR_30Myr'][...] = SFR_av
        fs.close()
    
if __name__ == '__main__':
    snap = int(sys.argv[1])
    L = int(sys.argv[2])
    NP  = int(sys.argv[3])
    phy  = sys.argv[4]
    
    start_time = time.time()
    ctg = GalCat(snap, NP=NP, L=L, phy=phy, gas=True, stars=True)
    print("--- %.3f: Tree Loaded ---" % (time.time() - start_time))
    print("--- Adding Prop ---")
    #ctg.add_SFRav()
    #ctg.add_GasTransferFractions()
    #ctg.add_VelDispGas()
    #ctg.addL(0)
    #ctg.add_GasMass(Mult=True)
    #ctg.add_GasMass(sf=True)
    #ctg.add_StarMass(Mult=True)
    #ctg.add_VelDispGas(SF=False, All=True)
    #ctg.add_VelDispGas(SF=False, All=True, feedback=True)
    #ctg.add_VelDispGas(SF=False, All=True, nofeed=True)
    #ctg.add_VelDispStar()
    #ctg.add_VelDispDM()
    print("--- %.3f: Done ---" % (time.time() - start_time))

    

