import numpy as np

h = 0.6777

def getCube(sindex, Ns):
    i = int(sindex%(Ns*Ns)/Ns)
    j = (sindex%(Ns*Ns))%Ns
    k = int(sindex/(Ns*Ns))
    return np.array([i,j,k])

def extent(ext, region, Lbox, sindex, Ns):
    #ext: extension in cMpc (h-factor corrected)
    
    inds = getCube(sindex, Ns)
    ext = ext*h # extension in EAGLE units
    
    if np.any((inds == 0) | (inds == Ns-1)):
        # Deals with galaxies in the border of the simulation
        
        RegionUnion = (region,)
        newx=False
        newy=False
        newz=False
        
        if region[0] == 0:
            xlims = [Lbox-ext, Lbox]
            newx = True
        elif region[1] == Lbox:
            xlims = [0, ext]
            newx = True
            
        if region[2] == 0:
            ylims = [Lbox-ext, Lbox]
            newy = True
        elif region[3] == Lbox:
            ylims = [0, ext]
            newy = True
        
        if region[4] == 0:
            zlims = [Lbox-ext, Lbox]
            newz = True
        elif region[5] == Lbox:
            zlims = [0, ext]
            newz = True
        
        if newx:
            newregion_x = np.array([xlims[0], xlims[1], region[2], region[3], region[4], region[5]])
            RegionUnion = RegionUnion + (newregion_x,) 
        
        if newy: 
            newregion_y = np.array([region[0], region[1], ylims[0], ylims[1], region[4], region[5]])
            RegionUnion = RegionUnion + (newregion_y,) 
            
        if newz: 
            newregion_z = np.array([region[0], region[1], region[2], region[3], zlims[0], zlims[1]])
            RegionUnion = RegionUnion + (newregion_z,) 
            
        if newx & newy:
            newregion_xy = np.array([xlims[0], xlims[1], ylims[0], ylims[1], region[4], region[5]])
            RegionUnion = RegionUnion + (newregion_xy,) 
        
        if newx & newz:
            newregion_xz = np.array([xlims[0], xlims[1], region[2], region[3], zlims[0], zlims[1]])
            RegionUnion = RegionUnion + (newregion_xz,) 
            
        if newy & newz:
            newregion_yz = np.array([region[0], region[1], ylims[0], ylims[1], zlims[0], zlims[1]])
            RegionUnion = RegionUnion + (newregion_yz,) 
            
        if newx & newy & newz: 
            newregion_xyz = np.array([xlims[0], xlims[1], ylims[0], ylims[1], zlims[0], zlims[1]])
            RegionUnion = RegionUnion + (newregion_xyz,) 
        return RegionUnion
    
    else:
        # Deals with the galaxies in the border of the subvolume
        xlims = [region[0]-ext, region[1]+ext]
        ylims = [region[2]-ext, region[3]+ext]
        zlims = [region[4]-ext, region[5]+ext]
        return (np.array([xlims[0], xlims[1], ylims[0], ylims[1], zlims[0], zlims[1]]), )
    
def getRegion(region, Lbox):
    if not np.any((region < 0) | (region > Lbox)): return (region,)
    else:
        region0 = np.where(region < 0, 1e-10, region)
        region0 = np.where(region0 > Lbox, Lbox-1e-10, region0)
        
        RegionUnion = (region0,)
        newx=False
        newy=False
        newz=False
        
        if region[0] < 0:
            xlims = [Lbox+region[0], Lbox-1e-10]
            newx = True
        elif region[1] > Lbox:
            xlims = [1e-10, region[1]-Lbox]
            newx = True
        if region[2] < 0:
            ylims = [Lbox+region[2], Lbox-1e-10]
            newy = True
        elif region[3] > Lbox:
            ylims = [1e-10, region[3]-Lbox]
            newy = True
        if region[4] < 0:
            zlims = [Lbox+region[4], Lbox-1e-10]
            newz = True
        elif region[5] > Lbox:
            zlims = [1e-10, region[5]-Lbox]
            newz = True
        
        if newx:
            region_x = np.array([xlims[0], xlims[1], region0[2], region0[3], region0[4], region0[5]])
            RegionUnion = RegionUnion + (region_x,) 
        
        if newy: 
            region_y = np.array([region0[0], region0[1], ylims[0], ylims[1], region0[4], region0[5]])
            RegionUnion = RegionUnion + (region_y,) 
            
        if newz: 
            region_z = np.array([region0[0], region0[1], region0[2], region0[3], zlims[0], zlims[1]])
            RegionUnion = RegionUnion + (region_z,) 
            
        if newx & newy:
            region_xy = np.array([xlims[0], xlims[1], ylims[0], ylims[1], region0[4], region0[5]])
            RegionUnion = RegionUnion + (region_xy,) 
        
        if newx & newz:
            region_xz = np.array([xlims[0], xlims[1], region0[2], region0[3], zlims[0], zlims[1]])
            RegionUnion = RegionUnion + (region_xz,) 
            
        if newy & newz:
            region_yz = np.array([region0[0], region0[1], ylims[0], ylims[1], zlims[0], zlims[1]])
            RegionUnion = RegionUnion + (region_yz,) 
            
        if newx & newy & newz: 
            region_xyz = np.array([xlims[0], xlims[1], ylims[0], ylims[1], zlims[0], zlims[1]])
            RegionUnion = RegionUnion + (region_xyz,) 
        
        return RegionUnion
    
def getGalRegion(COP, R200, Lbox):
    # Note R200 is the one from the descendant (see getAcrretion.py)
    
    x,y,z = COP.T
    Rmax = 2*R200 
    xmin, xmax = (x-Rmax, x+Rmax)
    ymin, ymax = (y-Rmax, y+Rmax)
    zmin, zmax = (z-Rmax, z+Rmax)
    
    region = np.array([xmin, xmax, ymin, ymax, zmin, zmax])
                      
    # Coordinates fixing -- addtional regions not added yet
    RegionUnion = np.where(region < 0, 0, region)
    RegionUnion = np.where(RegionUnion > Lbox, Lbox, RegionUnion)
    
    boxes = (RegionUnion, )
    
    newx=False
    newy=False
    newz=False
    
    if xmin < 0:
        xlims = [Lbox+xmin, Lbox]
        newx = True
    elif xmax > Lbox:
        xlims = [0, xmax-Lbox]
        newx = True
        
    if ymin < 0:
        ylims = [Lbox+ymin, Lbox]
        newy = True
    elif ymax > Lbox:
        ylims = [0, ymax-Lbox]
        newy = True
        
    if zmin < 0:
        zlims = [Lbox+zmin, Lbox]
        newz = True
    elif zmax > Lbox:
        zlims = [0, zmax-Lbox]
        newz = True
    
    if newx:
        newregion_x = np.array([xlims[0], xlims[1], RegionUnion[2], RegionUnion[3], RegionUnion[4], RegionUnion[5]])
        boxes = boxes + (newregion_x,) 
    
    if newy: 
        newregion_y = np.array([RegionUnion[0], RegionUnion[1], ylims[0], ylims[1], RegionUnion[4], RegionUnion[5]])
        boxes = boxes + (newregion_y,) 
        
    if newz: 
        newregion_z = np.array([RegionUnion[0], RegionUnion[1], RegionUnion[2], RegionUnion[3], zlims[0], zlims[1]])
        boxes = boxes + (newregion_z,) 
        
    if newx & newy:
        newregion_xy = np.array([xlims[0], xlims[1], ylims[0], ylims[1], RegionUnion[4], RegionUnion[5]])
        boxes = boxes + (newregion_xy,) 
    
    if newx & newz:
        newregion_xz = np.array([xlims[0], xlims[1], RegionUnion[2], RegionUnion[3], zlims[0], zlims[1]])
        boxes = boxes + (newregion_xz,) 
        
    if newy & newz:
        newregion_yz = np.array([RegionUnion[0], RegionUnion[1], ylims[0], ylims[1], zlims[0], zlims[1]])
        boxes = boxes + (newregion_yz,) 
        
    if newx & newy & newz: 
        newregion_xyz = np.array([xlims[0], xlims[1], ylims[0], ylims[1], zlims[0], zlims[1]])
        boxes = boxes + (newregion_xyz,) 

    return boxes



