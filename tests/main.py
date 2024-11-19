# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 15:31:15 2021

@author: s146407
"""

from cardiga import multimodels as multm
from cardiga.geometry import Geometry
from nutils import function, mesh, cli, export
import numpy, treelog, pickle, os
directC    = os.path.realpath(os.path.dirname(__file__))


##########################     
## Check fiber function ##
##########################
def main_fibers():
   name     = 'LONG' #'LONG','THK'
   direct   = 'geometries'
   directI  = os.path.join(os.path.split(directC)[0], direct)
   filename =  f'BV_GEOMETRY_{name}.pickle'
   Ventricle  = Geometry(filename, direct=directI)
   topo,geom  = Ventricle.get_topo_geom(nrefine=2)
   #topo, geom = get_topo_geom(filename, direct=direct, nrefine=1)


   boundary_cond = {}
   deg = numpy.pi/180
   angles_input  = {'α_epi_l' :-60.*deg,
                    'α_endo_l':+60.*deg,
                    'β_epi_l' :+20.*deg,
                    'β_endo_l':-20.*deg}
                                           
   submodel = multm.Models(topo, geom, boundary_cond) # ..,fiber=Rossi), if not selected, isotropic behavior is used and or x-direction for activation stress 
   fibers   = submodel.fiber('rossi','bv')         # dict with fiber solution, constants/variables are stored in namespace
 
   # Sample the topology
   bezier   = topo.sample('bezier', 15)
   X, phi, el, et, ec, alpha, beta, ef, es, en, Xi   = bezier.eval(['x_i','φ','el_i','et_i','ec_i','α','β','ef_i','es_i','en_i', 'ξ'] @ submodel.ns, **fibers)
   export.vtk('biventr_fiber_check', bezier.tri, X, phi=phi, el=el, et=et, ec=ec, alpha=alpha, beta=beta, ef=ef, es=es, en=en, Xi=Xi)

#   X, el, ec, et, xi, EC, ET, EL   = bezier.eval(['x_i','el_i','ec_i','et_i','ξcoord','EC_i','ET_i','EL_i'] @ submodel.ns, **fibers)
#   export.vtk('leftventr_fiber_check', bezier.tri, X, el=el, et=et, ec=ec, xi=xi, EC=EC, ET=ET, EL=EL)

#   gauss     = topo.sample('bezier', 10) #topo.sample('gauss', 4)  
#   X, u, v, ef, ah, at   = gauss.eval(['x_i','U','V','ef_i', 'αh', 'αt'] @ submodel.ns, **fibers)
#   export.vtk('leftventr_fiber_check', gauss.tri, X, u=u, v=v, ef=ef, at=at, ah=ah)
   return


def main_lex():

   direct   = 'geometries'
   directI  = os.path.join(os.path.split(directC)[0], direct)
   filename = 'LV_GEOMETRY_PS.pickle'
   nrefine  = 0 #2
   Ventricle  = Geometry(filename, direct=directI)
   topo,geom  = Ventricle.get_topo_geom(nrefine=nrefine)
   #topo, geom = get_topo_geom(filename, direct=direct, nrefine=nrefine)

   boundary_cond = {}
   deg = numpy.pi/180
   angles_input  = {'α_epi_l' :-60.*deg,
                    'α_endo_l':+60.*deg,
                    'β_epi_l' :+20.*deg,
                    'β_endo_l':-20.*deg}
                                           
   submodel = multm.Models(topo, geom, boundary_cond) # ..,fiber=Rossi), if not selected, isotropic behavior is used and or x-direction for activation stress 
   #fibers   = submodel.fiber('analytic','lv')
   fibers   = submodel.fiber('rossi','lv',angles_input=angles_input)         # dict with fiber solution, constants/variables are stored in namespace    
  
   bezier   = topo.sample('bezier', 6)
   X, phi, el, et, ec, alpha, beta, ef, es, en   = bezier.eval(['x_i','φ','el_i','et_i','ec_i','α','β','ef_i','es_i','en_i'] @ submodel.ns, **fibers)
   export.vtk('leftventricle_lex', bezier.tri, X, phi=phi, el=el, et=et, ec=ec, alpha=alpha, beta=beta, ef=ef, es=es, en=en)

## Check fiber function
def main_lv():
   direct   = 'geometries'
   directI  = os.path.join(os.path.split(directC)[0], direct)
   filename = 'LV_GEOMETRY.pickle'
   nrefine  = 2
   Ventricle  = Geometry(filename, direct=directI)
   topo,geom  = Ventricle.get_topo_geom(nrefine=nrefine)
   #topo, geom = get_topo_geom(filename, direct=direct, nrefine=nrefine)

   boundary_cond = {}
   
   submodel = multm.Models(topo, geom, boundary_cond) # ..,fiber=Rossi), if not selected, isotropic behavior is used and or x-direction for activation stress 
   fibers   = submodel.fiber('analytic','lv', quad_degree=5)#, bdegree=3, quad_degree=5)
   #fibers   = submodel.fiber('rossi','lv',angles_input=angles_input)         # dict with fiber solution, constants/variables are stored in namespace    
 
#   bezier   = topo['patch5'].sample('bezier', 6)
#   X, phi, el, et, ec, alpha, beta, ef, es, en, Xi   = bezier.eval(['x_i','φ','el_i','et_i','ec_i','α','β','ef_i','es_i','en_i', 'ξ'] @ submodel.ns, **fibers)
#   export.vtk('biventr_fiber_check', bezier.tri, X, phi=phi, el=el, et=et, ec=ec, alpha=alpha, beta=beta, ef=ef, es=es, en=en, Xi=Xi)
#   X, el, ec, et, xi, EC, ET, EL   = bezier.eval(['x_i','el_i','ec_i','et_i','ξcoord','EC_i','ET_i','EL_i'] @ submodel.ns, **fibers)
#   export.vtk('leftventr_fiber_check', bezier.tri, X, el=el, et=et, ec=ec, xi=xi, EC=EC, ET=ET, EL=EL)

   gauss     = topo.sample('bezier', 6) #topo.sample('gauss', 4)  
   X, u, v, ef, ec, el, et, ah, at, xi, theta, phi   = gauss.eval(['x_i','U','V','ef_i','ec_i','el_i','et_i', 'αh', 'αt','ξcoord','θcoord','φcoord'] @ submodel.ns, **fibers)
   export.vtk('leftventr_fiber_check', gauss.tri, X, u=u, v=v, ef=ef, ec=ec, el=el, et=et, at=at, ah=ah, xi=xi, theta=theta, phi=phi)
   return

def main_element_lv():
   direct   = 'geometries'
   directI  = os.path.join(os.path.split(directC)[0], direct)
   filename = 'LV_GEOMETRY.pickle'
   nrefine  = 3
   Ventricle  = Geometry(filename, direct=directI)
   topo,geom  = Ventricle.get_topo_geom(nrefine=nrefine)
   #topo, geom = get_topo_geom(filename, direct=direct, nrefine=nrefine)
   treelog.info(numpy.mean(topo.integrate_elementwise(function.J(geom), degree=8)**(1/3)))   
      
      
if __name__ == "__main__":
    #cli.run(main_element_lv)
    #cli.run(main_fibers)    
    #cli.run(main_lex)
    cli.run(main_lv)