#script recieved from Aditya Manuwal (UWA)

import h5py  as h5
import os
import numpy as np

def ReadEagleSubfind_halo(Base,DirList,fend,exts):
   fn            = Base+DirList+'/groups_'+exts+fend+'/'+'eagle_subfind_tab_'+exts+fend+'.0.hdf5'
   print(' __'); print(' Directory:',Base+DirList+'/groups_'+exts+fend+'/'+'eagle_subfind_tab_'+exts+fend+' ...')
   fs               = h5.File(fn,"r")
   Header           = fs['Header'].attrs
   Ntask            = Header['NTask']
   TotNgroups       = Header['TotNgroups']
   TotNsubgroups    = Header['TotNsubgroups']

   fs.close()
   
   # --- Define Group Arrays
   if TotNgroups > 0:
      Group_M_Crit200  = np.empty(TotNgroups,     dtype=float)
      Group_R_Crit200  = np.empty(TotNgroups,     dtype=float)
      GroupPos         = np.empty((TotNgroups,3), dtype=float)
      FirstSub         = np.empty(TotNgroups,     dtype=int)
      NumOfSubhalos    = np.empty(TotNgroups,     dtype=int)
      
   # --- Define Subhalo Arrays
      HalfMassRad      = np.empty((TotNsubgroups,6), dtype=float)
      MassType         = np.empty((TotNsubgroups,6), dtype=float)
      StarSpin         = np.empty((TotNsubgroups,3), dtype=float)
      SFSpin           = np.empty((TotNsubgroups,3), dtype=float)
      SubVelocity      = np.empty((TotNsubgroups,3), dtype=float)
      Pcom             = np.empty((TotNsubgroups,3), dtype=float)
      SubPos           = np.empty((TotNsubgroups,3), dtype=float)
      Vbulk            = np.empty((TotNsubgroups,3), dtype=float)
      Vmax             = np.empty(TotNsubgroups,     dtype=float)
      VmaxRadius       = np.empty(TotNsubgroups,     dtype=float)
      SFRate           = np.empty(TotNsubgroups,     dtype=float)
      SubMass          = np.empty(TotNsubgroups,     dtype=float)
      SubLen           = np.empty(TotNsubgroups,     dtype=int)
      SubGroupNumber   = np.empty(TotNsubgroups,     dtype=int)
      GroupNumber      = np.empty(TotNsubgroups,     dtype=int)
      Mass_30kpc       = np.empty((TotNsubgroups,6), dtype=float)
      SFR_5kpc         = np.empty(TotNsubgroups,     dtype=float)

      NGrp_c           = 0
      NSub_c           = 0

      for ifile in range(0,Ntask):
         if os.path.isdir(Base+DirList+'data/'):
            fn         = Base+DirList+'data/groups_'+exts+fend+'/'+'eagle_subfind_tab_'+exts+fend+'.'+str(ifile)+'.hdf5'
         else:
            fn         = Base+DirList+'/groups_'+exts+fend+'/'+'eagle_subfind_tab_'+exts+fend+'.'+str(ifile)+'.hdf5'
         fs            = h5.File(fn,"r")      
         Header        = fs['Header'].attrs
         HubbleParam   = Header['HubbleParam']
         #PartMassDM    = Header['PartMassDM']
         Redshift      = Header['Redshift']
         Om            = [Header['Omega0'],Header['OmegaLambda'],Header['OmegaBaryon']]
         Ngroups       = Header['Ngroups']
         BoxSize       = Header['BoxSize']
         Time          = Header['Time']
         
         #TotNgroups    = Header['TotNgroups']
         Nsubgroups    = Header['Nsubgroups']
         #TotNsubgroups = Header['TotNsubgroups']
         HubbleH       = Header['H(z)']
         

         if Ngroups > 0:
            Group_M_Crit200[NGrp_c:NGrp_c+Ngroups]     = fs["FOF/Group_M_Crit200"][()]
            Group_R_Crit200[NGrp_c:NGrp_c+Ngroups]     = fs["FOF/Group_R_Crit200"][()]
            GroupPos[NGrp_c:NGrp_c+Ngroups]            = fs["FOF/GroupCentreOfPotential"][()]
            FirstSub[NGrp_c:NGrp_c+Ngroups]            = fs["FOF/FirstSubhaloID"][()]
            NumOfSubhalos[NGrp_c:NGrp_c+Ngroups]       = fs["FOF/NumOfSubhalos"][()]

            NGrp_c += Ngroups

         if Nsubgroups > 0:
            StarSpin[NSub_c:NSub_c+Nsubgroups,:]       = fs["Subhalo/Stars/Spin"][()]
            SFSpin[NSub_c:NSub_c+Nsubgroups,:]         = fs["Subhalo/SF/Spin"][()]
            SubVelocity[NSub_c:NSub_c+Nsubgroups,:]    = fs["Subhalo/Velocity"][()]
            MassType[NSub_c:NSub_c+Nsubgroups,:]       = fs["Subhalo/MassType"][()]
            SubPos[NSub_c:NSub_c+Nsubgroups,:]         = fs["Subhalo/CentreOfPotential"][()]
            Pcom[NSub_c:NSub_c+Nsubgroups,:]           = fs["Subhalo/CentreOfMass"][()]
            Vmax[NSub_c:NSub_c+Nsubgroups]             = fs["Subhalo/Vmax"][()]
            VmaxRadius[NSub_c:NSub_c+Nsubgroups]       = fs["Subhalo/VmaxRadius"][()]
            SubLen[NSub_c:NSub_c+Nsubgroups]           = fs["Subhalo/SubLength"][()]
            SFRate[NSub_c:NSub_c+Nsubgroups]           = fs["Subhalo/StarFormationRate"][()]
            SubMass[NSub_c:NSub_c+Nsubgroups]          = fs["Subhalo/Mass"][()]
            SubGroupNumber[NSub_c:NSub_c+Nsubgroups]   = fs["Subhalo/SubGroupNumber"][()]
            GroupNumber[NSub_c:NSub_c+Nsubgroups]      = fs["Subhalo/GroupNumber"][()]
            HalfMassRad[NSub_c:NSub_c+Nsubgroups,:]    = fs["Subhalo/HalfMassRad"][()]
            Mass_30kpc[NSub_c:NSub_c+Nsubgroups,:]     = fs["Subhalo/ApertureMeasurements/Mass/030kpc"][()]
            SFR_5kpc[NSub_c:NSub_c+Nsubgroups]         = fs["Subhalo/ApertureMeasurements/SFR/005kpc"][()]
 
            NSub_c += Nsubgroups
         
         fs.close()

      return {'Group_M_Crit200':Group_M_Crit200,    'Group_R_Crit200':Group_R_Crit200,      'GroupPos':GroupPos, 
              'FirstSub':FirstSub,                  'NumOfSubhalos':NumOfSubhalos,          'Vmax':Vmax, 
              'Vbulk':Vbulk,                        'SubPos':SubPos,                        'Pcom':Pcom,  
              'SubLen':SubLen,                      'SubGroupNumber':SubGroupNumber,        'GroupNumber':GroupNumber,        
              'TotNgroups':TotNgroups,              'TotNsubgroups':TotNsubgroups,          'Redshift':Redshift, 
              'BoxSize':BoxSize,                    'HubbleParam':HubbleParam,              'Omega':Om,
              'SFRate':SFRate,                      'MassType':MassType,                    'Mass_30kpc':Mass_30kpc[:,4],
              'SubMass':SubMass,                    'HalfMassRad':HalfMassRad,              'Time': Time,
              'H(z)':HubbleH,
              'StarSpin': StarSpin,
              'SFSpin': SFSpin,
              'Vcom': SubVelocity,
              'Rmax':VmaxRadius,
              'SFR_5kpc': SFR_5kpc
              }
   else:
      return {'Omega':Om,  'BoxSize':BoxSize, 'Redshift':Redshift, 'HubbleParam':HubbleParam}# 'PartMassDM':PartMassDM}
   
