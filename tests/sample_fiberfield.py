from cardiga.geometry import Geometry
from cardiga import multimodels as multm
import treelog
from nutils import export, cli

## Check fiber function
def main(Usample : int):
   '''
    .. arguments::

       Usample [60]
         Number of uniform samples
         
   '''
   filename = 'LV_GEOMETRY.pickle'
   nrefine  = 3

   Ventricle   = Geometry(filename)
   topoI,geom  = Ventricle.get_topo_geom() # <-- Make sure tnelems is 'correct', of the same order as the global refinement
   treelog.info('Number of elements: {}'.format(len(topoI.integrate_elementwise(geom, degree=0)))) # Print number of elements in topology after refinement

   topo = topoI.refine(nrefine)

   boundary_cond = {}
   submodel = multm.Models(topo, geom, boundary_cond) # ..,fiber=Rossi), if not selected, isotropic behavior is used and or x-direction for activation stress 
   fibers   = submodel.fiber('analytic','lv')
   #fibers   = submodel.fiber('rossi','lv',angles_input=angles_input)         # dict with fiber solution, constants/variables are stored in namespace    

   bezier  = topoI.sample('bezier', Usample) 
   Ubezier     = topo.locate(geom, bezier.eval(geom), eps=1e-10)
   
   #X, u, v, ef, ec, el, et, ah, at, xi, theta, phi   = bezier.eval(['x_i','U','V','ef_i','ec_i','el_i','et_i', 'αh', 'αt','ξcoord','θcoord','φcoord'] @ submodel.ns, **fibers)
   #export.vtk('leftventr_fiber_check', bezier.tri, X, u=u, v=v, ef=ef, ec=ec, el=el, et=et, at=at, ah=ah, xi=xi, theta=theta, phi=phi)
   X, ef   = Ubezier.eval(['x_i','ef_i'] @ submodel.ns, **fibers)
   export.vtk('leftventricle_fibers', bezier.tri, X, ef=ef)
   return

if __name__ == "__main__":
   cli.run(main)