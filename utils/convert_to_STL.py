# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 17:46:27 2022

@author: s146407
"""
import pickle, numpy, stl
from nutils import export, mesh

def load_pickle(filename):
      with open(filename + '.pickle', 'rb') as f: 
          return pickle.load(f)    
      
def save_stl(topo, geom, filename: str = 'filename', nrsamples: int = 10, boundaries: tuple = ()):       

        bezier  = topo.boundary['base_l,epi,endo_l'].sample('bezier', nrsamples) # Connect
        #bezier  = topo.boundary['base_l,epi,endo_l,base_r,endo_r'].sample('bezier', nrsamples) # Connect
        GEOM    = bezier.eval(geom)                # Coords
                        
        data = numpy.zeros(len(bezier.tri), dtype=stl.mesh.Mesh.dtype)
        stl_mesh = stl.mesh.Mesh(data, remove_empty_areas=False)
        stl_mesh.x[:] = GEOM[:,0][bezier.tri]
        stl_mesh.y[:] = GEOM[:,1][bezier.tri]
        stl_mesh.z[:] = GEOM[:,2][bezier.tri]
        stl_mesh.save(filename + '.stl')
        return
            


direct        = 'input/'  
geometry_file = 'p7_s5_fullsolid' #'LV_3D_ref2' #p7_s5_fullsolid
cpsn, w, patchvertsn, patches, nelems, knotval, knotmult, boundaries = load_pickle(direct + geometry_file)
cps        = cpsn*60
patchverts = patchvertsn*60
#patches = patchesn.astype(numpy.int64)
nrefine = 2 #2 # <-- Make sure tnelems is 'correct'


topo, lingeom = mesh.multipatch(patches=patches, patchverts=patchverts, nelems=nelems)
topo     = topo.withboundary(**boundaries)
bsplines = topo.basis('spline', degree=2, patchcontinuous=False, knotvalues=knotval, knotmultiplicities=knotmult)
weight   = bsplines.dot(w)
geom     = bsplines.vector(3).dot((cps*w[:,numpy.newaxis]).ravel())/weight


save_stl(topo, geom, 'output/vtk/PatientspecificCOMBATVT_STL', nrsamples=15)
