from nutils import solver, cli, export, function
import treelog, math, os
from cardiga.geometry import Geometry
from cardiga.infarct import Infarct 
from cardiga.ablation import Ablation
import numpy as np
            
## Left ventricle (idealized) with infarct simulation
def main(nrefine: int, btype: str, bdegree: int, qdegree: int, scar_reflvl : int, scar_degr : int, saveScarVTK : bool, Usample : int, VTKboundaryOnly : bool, convert2Uniform: bool):
    '''
    .. arguments::

       nrefine [0]
         Number of uniform refinements.

       btype [th-spline]
         Basis function type, i.e. splines, std, th-spline, ...

       bdegree [3]
         Basis function degree to be used.

       qdegree [5]
         Quadrature degree to be used.

       scar_reflvl [4]
         Refinement of the scar location/topology.

       scar_degr [3]
         Polynomial degree of the scar stiffness.    

       saveScarVTK [False]
         Save results to vtk file.
         
       Usample [20]
         Number of uniform sampling of the base topology.   
         
       VTKboundaryOnly [False]
         Save only boundary not internal/solid.   
         
       convert2Uniform [False]
         Convert the hierarchical mesh to a uniform one, which is used for post processing tasks (save vtk, computations etc.)

    '''
    ml=1e-6;mm=1e-3;mmHg=133.3223684;ms=1e-3;s=1;kPa=1e3;cm=1e-2;   

    directC    = os.path.realpath(os.path.dirname(__file__)) # Current directory
    directI    = os.path.split(directC)[0] # Remove the last folder to get in the main working directory 
    direct     = 'geometries'  
    filename   = 'LV_GEOMETRY.pickle'
    Ventricle  = Geometry(filename, direct=os.path.join(directI, direct))
    topo,geom  = Ventricle.get_topo_geom(nrefine=nrefine)
    
    # Load scar topology from separate file if applicable
    # directInfarct   = "data infarct"
    # filenameInfarct = "ScarTissue_ref{}_degree{}_gaussdeg5".format(scar_ref, scar_degr)
    # topoHierarch, scarValues, gauss = load_pickle(os.path.join(directC, directInfarct, filenameInfarct))
    # infarct = Infarct(topoHierarch, geom) # Construct scar 
    # topoInf, scar = infarct.project(scarValues, gauss, bdegree=scar_degr, gdegree=5)   # Project data onto basis 
    # topoInf, scar = infarct.convolute() # Convolute data onto basis
    
    ## Perform ablation (testing)
    lnames = ["5L"]#[4,5,6] #[3,4,5,6,7] # [4,5,6]
    Lnames =     {1 : [0.01470, 0., -0.02075], 2 : [0.01430, 0., -0.02275], 3 : [0.01385, 0., -0.02475],
                  4 : [0.01335, 0., -0.02675], 5 : [0.01280, 0., -0.02875], 6 : [0.01220, 0., -0.03075],
                  7 : [0.01150, 0., -0.03275], 8 : [0.01070, 0., -0.03475], 9 : [0.00970, 0., -0.03675],
                "5L": [0.0112,  0.0055, -0.02875], "5R": [0.0112, -0.0055, -0.02875]}
    Ablation_sites = np.array([Lnames[Lname] for Lname in lnames])              
    #Ablation_sites = np.array([[0.0115, 0., -0.0325 ]]) #np.array([[0.012, 0, -0.031 ]]) # np.array([[0.0115, 0., -0.0325 ]]) #np.array([[0.0115, 0., -0.0325 ]])
    
#                      np.array([[0.0160, 0., -0.01 ],
#                               [0.0162, 0., -0.005],
#                               [0.0163, 0.,  0.   ],
#                               
#                               [0.010, 0.003, -0.036],
#                               [0.006, 0.006, -0.037],
#                               [0.003, 0.009, -0.038]])
                               
    Ablation_radii  = np.ones(len(Ablation_sites))*5*mm
    Ablation_δradii = np.ones(len(Ablation_sites))*2*mm # Border zone radius increment/thickness
    Ablate = Ablation(topo, geom)
    #ablated_sites = Ablate.ablate_sites(Ablation_sites, Ablation_radii, δRsites=Ablation_δradii) # Perform ablation (i.e. construct appropriate field)
    ablated_sites = 0 # If we want to ignore ablation
    
    infarct = Infarct(topo, geom)#, ablation = Ablate) # Construct scar
#    θscar   = (1.75, 2.50)
#    φscar   = (-0.350,0.350)#(-0.175, 0.175)
#    ξscar   = (0.371296808757, 0.678355651828)#(1.35*0.371296808757, 0.8*0.678355651828)
#    Lborder = 0.016
    
    
#    topo, scar, scarlhs = infarct.analytic(θscar, φscar, ξscar, improve=True, btype=btype, bdegree=scar_degr, gaussdegr=qdegree, 
#                                           reflvl=scar_reflvl, Lborder=Lborder, constrainBase=True, saveVTK=False, saveGpoints=True, 
#                                           refineInterface=False, refineExtrNode=False)  # Construct based on analytic expression/input

    # Construct scar, made up of 4 individual scars:
#    θscar_Triang1  = (1.6, 2.1)
#    φscar_Triang1  = (-0.50,0.50)
#    ξscar_Triang1  = (0.371296808757, 0.678355651828)
#    
#    θscar_Rec1  = ( 2.1 , 2.6)
#    φscar_Rec1  = (-0.50, 0.50)
#    ξscar_Rec1  = (0.371296808757, 0.4)
#    
#    θscar_Rec2  = ( 2.6, 2.4, 2.6, 2.2) #( 2.4 , 2.6)
#    φscar_Rec2  = (-0.50, 0.50)
#    ξscar_Rec2  = (0.40, 0.55)
#    
#    θscar_Rec3  = ( 2.1 , 2.6)
#    φscar_Rec3  = (-0.50, 0.50)
#    ξscar_Rec3  = (0.55, 0.678355651828)
    
    
    θscar_Triang1  = (1.6, 2.1)
    φscar_Triang1  = (-0.50,0.50)
    ξscar_Triang1  = (0.371296808757, 0.678355651828)
    
    θscar_Rec1  = ( 2.1 , 2.6)
    φscar_Rec1  = (-0.50, 0.50)
    ξscar_Rec1  = (0.371296808757, 0.4)
    
    θscar_Rec2  = ( 2.4 , 2.6)
    φscar_Rec2  = (-0.50, 0.50)
    ξscar_Rec2  = (0.371296808757, 0.371296808757 + 0.15) #(0.371296808757, 0.371296808757 + 0.15) #(0.40, 0.55)
    
    θscar_Rec3  = ( 2.1 , 2.6)
    φscar_Rec3  = (-0.50, 0.50)
    ξscar_Rec3  = (0.371296808757 + 0.15, 0.678355651828) #(0.371296808757 + 0.15, 0.678355651828) #(0.55, 0.678355651828)
    
    
    
    θscar_ChannelEndo  = ( 2.1 , 2.4)
    φscar_ChannelEndo  = (-0.50, 0.50)
    ξscar_ChannelEndo  = (0.371296808757, 0.371296808757 + 0.15) #(0.371296808757 + 0.15, 0.678355651828) #(0.55, 0.678355651828)
    
    
    
    Lborder = 0.01
    
    if VTKboundaryOnly:
      bezier = topo.boundary.sample('bezier', Usample)
    else:
      bezier = topo.sample('bezier', Usample)
    X, abl_site   = bezier.eval([geom, ablated_sites])
    scar_objectT1 = infarct.triangle(θscar_Triang1, φscar_Triang1, ξscar_Triang1, Lborder=Lborder, saveManifold=True, filename= "Triangle1.obj")
    
    #scar_objectR1 = infarct.rectangle(θscar_Rec1, φscar_Rec1, ξscar_Rec1, Lborder=Lborder, saveManifold=True, filename= "Rectangle1.obj") 
    
    scar_objectR2 = infarct.rectangle(θscar_Rec2, φscar_Rec2, ξscar_Rec2, Lborder=Lborder, saveManifold=True, filename= "Rectangle2.obj") 
    scar_objectR3 = infarct.rectangle(θscar_Rec3, φscar_Rec3, ξscar_Rec3, Lborder=Lborder, saveManifold=True, filename= "Rectangle3.obj") 
    scar_ChannelEndo = infarct.rectangle(θscar_ChannelEndo, φscar_ChannelEndo, ξscar_ChannelEndo, Lborder=Lborder, saveManifold=True, filename= "ChannelEndo.obj") 
    
    distance_ChannelEndo = infarct.distance_to_manifolds(scar_ChannelEndo, X=X, normalize=False)
    
    #distances = infarct.distance_to_manifolds(scar_objectT1, scar_objectR1, scar_objectR2, scar_objectR3, X=X)#, normalize=True)
    distances = infarct.distance_to_manifolds(scar_objectT1, scar_objectR2, scar_objectR3, X=X, normalize=False)
    
    
    scarred_tissue = distances*( 1 - abl_site ) + abl_site
    
    
    export.vtk('Distancefield_channelEndo_boundary', bezier.tri, X, distance=distances, distance_channel=distance_ChannelEndo)#ablation=abl_site, distance=distances, distance_channel=distance_ChannelEndo)
    #export.vtk('Distancefield_channelInfarctEndo_Ablated', bezier.tri, X, scarred_tissue=scarred_tissue)
    #export.vtk('Distancefield_channelInfarctEndo'.format(scar_reflvl,scar_degr), bezier.tri, X, distances=distances)
    
    treelog.info('Number of elements: {}'.format(len(topo.integrate_elementwise(geom, degree=0)))) # Print number of elements in topology after refinement
    
    # Do a post-processing step
#    infarctPP = Infarct(topo, geom) 
#    topo, scar, scarlhs = infarctPP.analytic(θscar, φscar, ξscar, improve=True, btype=btype, bdegree=6, gaussdegr=15, reflvl=1, constrainBase=True, saveVTK=False, saveGpoints=True)  # Construct based on analytic expression/input
#    treelog.info('Number of elements: {}'.format(len(topo.integrate_elementwise(geom, degree=0)))) # Print number of elements in topology after refinement
#

    ns = function.Namespace()
    ns.scar   = scar
    ns.asites = ablated_sites 
    ns.deadtissue = ns.scar*( 1 - ns.asites ) + ns.asites
    
    ns.a0     = (scar*3.6 + 0.4)*1e3 # passive stiffness relation
    ns.x      = geom
#    
#    # save gauss point values again (should ideally match the once analytically determined)
#    gsample = topo.sample('gauss', qdegree)
#    gX, gscar = gsample.eval(['x_i','scar'] @ ns, **scarlhs)
#    np.savetxt('GaussPoints projected.txt', np.concatenate([gscar[:,np.newaxis],gX],axis=1), delimiter=',',header='scar, X, Y, Z', comments='') # First save to relevant file
    
    # Convert sample to a uniform sampling if specified (based on hierarchical sample)
    if convert2Uniform:
      if VTKboundaryOnly:
         Ubezierbase = topo.basetopo.boundary.sample('bezier', Usample)
      else:
         # just for now 
         topoI,geom  = Ventricle.get_topo_geom(nrefine=0) 
         Ubezierbase = topoI.basetopo.sample('bezier', Usample)
         #Ubezierbase = topo.basetopo.sample('bezier', Usample)
      Ubezier     = topo.locate(ns.x, Ubezierbase.eval(ns.x), eps=1e-10)
    else:
      if VTKboundaryOnly:
        Ubezier = topo.boundary.sample('bezier', Usample)
      else:
        Ubezier = topo.sample('bezier', Usample)
      Ubezierbase = Ubezier
      
    
      
      
      
    
    ## Save the distribution of the scar tissue------------------------------------------------------------------------------------------------------
    if saveScarVTK:
      X, a0, asites, deadtissue = Ubezier.eval(['x_i','a0','asites', 'deadtissue'] @ ns, **scarlhs)
      #X, scar = Ubezier.eval(['x_i','scar'] @ ns, **scarlhs)
      export.vtk('IGA_ScarResult_scarref{}_scardegr{}'.format(scar_reflvl,scar_degr), Ubezierbase.tri, X, a0=a0, asites=asites, deadtissue=deadtissue)
    #------------------------------------------------------------------------------------------------------------------------------------------------
        
if __name__ == "__main__":
   cli.run(main)