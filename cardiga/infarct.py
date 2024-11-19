from nutils.topology import Topology, HierarchicalTopology
from nutils.sample import Sample
import numpy as np
from nutils import function, solver, export, _util as util
from typing import Union, List
from nutils.function import _Wrapper
from .prolate_functions import Ellips_Functs
import treelog
from pygel3d import hmesh
from scipy.spatial import ConvexHull, Delaunay
from .geometry import regularize_mesh, InterfaceElem
from .ablation import Ablation

_ = np.newaxis

class Infarct():

    def __init__(self, topo: Topology, geom: function.Array, ablation : Ablation = Ablation(None, None)):
        self.topo = topo
        self.geom = geom
        self.ns   = function.Namespace()
        self.ns.x = geom
        self.ablation  = ablation # If ablation is specified, we have a class, otherwise it returned None
        self.manifolds = []
        return


    def project(self, infarct : np.array, gaussSample: Sample, btype : str = 'th-spline', bdegree : int = 3, gdegree : int = 2) -> Union[HierarchicalTopology, function.Array]:
        """
        Project the infarct quantities onto a suitable basis given a Gausspoint sample.
          Parameters
          ---------- 
            infarct     : Values that indicate infarct tissue (infarct=1) and healthy tissue (infarct=0)
            gaussSample : Gausspoint sample of the topology
            btype       : Desired basis function type (typically th-spline) 
            bdegree     : Desired basis function degree
            gdegree     : Quadrature degree of the projection step

          Returns
          -------
            The topology, projected infarct
        """
        ns = self.ns
        ns.Scarsampled = gaussSample.asfunction(infarct) 
        ns.scarbasis   = self.topo.basis(btype, degree=bdegree)
        ns.scar        = 'scarbasis_n ?scarlhs_n'

        # previously used because it was known that the value a0 had a specific value at the base
        # limit=(400,4000)
        # min_arg = '( a0T - {} )^2 d:x'.format(limit[0])
        # sqra0C  = self.topo.boundary['base_l'].integral(min_arg @ self.ns, degree=8)                
        # consa0  = solver.optimize('lhsa0', sqra0C, tol=1e-6, droptol=1e-15) 

        ## Projection
        scarsqr      = self.topo.integral('( Scarsampled - scar )^2 d:x' @ ns, degree=gdegree)                
        scarlhs      = solver.optimize('scarlhs', scarsqr, tol=1e-6, droptol=1e-15) #, constrain=consa0)
        ns.scarTrunc = np.minimum( np.maximum(ns.scar, 0), 1) # Truncate to make sure no over or undershoots are obtained after projection
        return self.topo, ns(scarlhs=scarlhs).scarTrunc
    

    def convolute(self, infarct : np.array, gaussSample: Sample, btype : str = 'th-spline', bdegree : int = 3, gdegree : int = 2) -> Union[HierarchicalTopology, function.Array]:
        """
        Convolute the infarct quantities onto a suitable basis given a Gausspoint sample.
          Parameters
          ---------- 
            infarct     : Values that indicate infarct tissue (infarct=1) and healthy tissue (infarct=0)
            gaussSample : Gausspoint sample of the topology
            btype       : Desired basis function type (typically th-spline) 
            bdegree     : Desired basis function degree
            gdegree     : Quadrature degree of the projection step

          Returns
          -------
            The topology, convoluted infarct
        """
        ns = self.ns
        ns.Scarsampled = gaussSample.asfunction(infarct) 
        ns.scarbasis   = self.topo.basis(btype, degree=bdegree)
        ns.scar        = 'scarbasis_n ?scarlhs_n'

        scarlhs = self.topo.project(self.ns.Scarsampled, self.ns.scarbasis, self.ns.x, degree=gdegree, ptype='convolute')
        ns.scarTrunc = function.min( function.max(ns.scar, 0), 1) # Truncate to make sure no over or undershoots are obtained after projection
        return self.topo, ns(scarlhs=scarlhs).scarTrunc
    

    def analytic(self, θ: tuple, φ: tuple, ξ: tuple, C : float = 0.043, Lborder : float = 0.016, constrainBase=False, improve : bool = False, 
                 gaussdegr: int = 5, reflvl : int = 3, btype : str = 'th-spline', bdegree : int = 3, saveVTK : bool = False, saveGpoints : bool = False, 
                 refineInterface : bool =False, refineExtrNode : bool = False, returnField=True) -> Union[HierarchicalTopology, function.Array]: 
        """
        Projects the infarct quantities onto a suitable basis given an analytic description.
          Parameters
          ---------- 
            θ           : Longitudinal (min, max) values 
            φ           : Circumferential (min, max) values 
            ξ           : Transmural (min, max) values 
            C           : Focal length of the ellipsoid
            Lborder     : Length of the border zone
            constrainBase   : Constrain the base to 0 (Not implemented yet)
            improve     : Improve the sampling strategy of the infarct manifold (recommended = True, but might slow down computation)
            gaussdegr   : Gauss point sampling degree
            reflvl      : Number of hierarchical refinement levels  
            btype       : Basis function type, typically 'th-spline'  
            bdegree     : Basis function degree onto which the scar properties are projected 
            saveVTK     : Save the hierarchical refinement results as vtk's
            saveGpoints : Save the Gauss quadrature points with the sampled values. This is the ground-truth onto which we project a basis.
            refineInterface : Refine the patch-interfaces once more at the end of the refinement procedure, where the gradient of the value of interest is sufficiently large.
            refineExtrNode  : Refine the elements that are connected to an extraordinary node and are win the region of interest
          Returns
          -------
            The topology, projected infarct
        """

        

        ns   = self.ns
        #lhs  = {}     
        if refineInterface or refineExtrNode:
           reflvl += 1
           
        with treelog.iter.fraction('level', range(reflvl)) as lrange:
            for irefine in lrange:

                if irefine: # First refinement does not require indicator field yet    
                    # Compute the refinement indicator
                    if irefine == reflvl-1 and (refineInterface or refineExtrNode): # We only want to refine the patch-interfaces
                      treelog.info("Refining patch-interfaces")
                      Ielem = InterfaceElem(ns, self.topo)
                      
                      if refineInterface:
                        ref_elems = Ielem.interface_elems_unique # interface element indices
                        # In this step, we do not perform a uniform refinement!
                        #indicator   = self.topo.integral('scarbasis_n ( scar_,k scar_,k ) d:x' @ ns, degree=bdegree*2).eval(**lhs)
                        #supp        = ns.scarbasis.get_support(indicator**2 > np.mean(indicator**2))
                      else: # refineExtrNode
                        ref_elems = Ielem.extr_elem # Element indices that are connected to extraordinary nodes
                      treelog.info(scarlhs)  
                      supp        = ns.scarbasis.get_support(scarlhs > 0.98)#0.95)
                      u, c        = np.unique(np.concatenate([supp, ref_elems]), return_counts=True)
                      refelems_indices = u[c > 1] # Filter such that only regions of interest are refined
                      self.topo   = self.topo.refined_by(refelems_indices) # perform final refinement at the patch-interfaces or at the extraordinary node (elements connected to this node)
                        
                           
                    else:   # Else we perform our regular refinement strategy
                      reftopo     = self.topo.refined
                      ns.refbasis = reftopo.basis(btype, degree=bdegree)
#                      indicator   = reftopo.integral('sqrt( scar_,k scar_,k ) d:x' @ ns, degree=bdegree*2).eval(**lhs)
#                      indicator  -= reftopo.boundary.integral('refbasis_n scar_,k n_k d:x' @ ns, degree=bdegree*2).eval(**lhs)
#                      supp        = ns.refbasis.get_support(indicator**2 > np.mean(indicator**2))
                      

                      indicator   = reftopo.integrate_elementwise('gradientfield d:x' @ ns(**lhs), degree=bdegree*2)
                      indicator  /= reftopo.integrate_elementwise('d:x' @ ns(**lhs), degree=bdegree*2)
                      treshold    = np.mean(indicator) #5 #np.mean(indicator)
                      supp        = np.where( indicator > treshold )[0]
                      #treelog.info(np.mean(indicator / elem_volume))
                      
                      #indicator   = reftopo.integral('refbasis_n ( scar_,k scar_,k ) d:x' @ ns, degree=bdegree*2).eval(**lhs)
                      #supp        = ns.refbasis.get_support(indicator > np.mean(indicator))
                      self.topo   = self.topo.refined_by(reftopo.transforms[supp])
                      
                val = 'scarlhs{}'.format(irefine)
               
            
                ## Sample the left ventricle and assign values to the gauss points corresponding the scar tissue and border zone
                #GaussSample, infarctValues = self.sample_analytic(θ, φ, ξ=ξ, C=C, Lborder=Lborder, gaussdegr=gaussdegr, improve=improve, saveGpoints=saveGpoints)
                GaussSample, infarctValues = self.sample_analytic_V2(C=C, Lborder=Lborder, gaussdegr=gaussdegr, improve=improve, saveGpoints=saveGpoints)
                #treelog.info(np.where(infarctValues > 0.1))
                ns.sampled = GaussSample.asfunction(infarctValues)
                
                ## Build suitable basis
                ns.scarbasis = self.topo.basis(btype, degree=bdegree)
                ns.scar      = 'scarbasis_n ?{}_n'.format(val)  
                ns.scarTrunc = np.minimum( np.maximum(ns.scar, 0), 1) # Truncate to make sure no over or undershoots are obtained after projection



                ## Add ablation field if specified
                if self.ablation != None:
                   ns.asites     = self.ablation.ablated_sites
                   ns.deadtissue = ns.scar*( 1 - ns.asites ) + ns.asites # Combine the ablation scars and the infarct scar
                   ns.gradientfield = 'sqrt( deadtissue_,k deadtissue_,k )'
                else:
                   ns.gradientfield = 'sqrt( scar_,k scar_,k )'      
                        
                        
                        
                        
                        
                           
                ## Constrain at the top (max 0)      
                if constrainBase:       
                   scarsqrC  = self.topo.boundary['base_l'].integral('scar^2 d:x' @ ns, degree=gaussdegr) #TODO check if this is desired?
                   scarcons  = solver.optimize(val, scarsqrC, tol=1e-6, droptol=1e-15) 
                else:
                   scarcons = None
                   
 
                #scarlhs = self.topo.project(ns.sampled, ns.scarbasis, ns.x, degree=gaussdegr, ptype='convolute') #ptype='convolute' #constrain=consa0,
                #scarlhs = self.topo.project(ns.sampled, ns.scarbasis, ns.x, degree=gaussdegr)#, constrain=scarcons)#, ptype='projection')
                if irefine < 2:
                  scarsqr  = self.topo.integral('( sampled - scar )^2 d:x' @ ns, degree=gaussdegr)        
                  scarlhs  = solver.optimize(val, scarsqr, tol=1e-9, droptol=1e-15, constrain=scarcons)
                  treelog.info(scarlhs)
                else:
                  nanvec_init    = util.NanVec(len(scarcons))
                  indices_notnan = np.logical_not(np.isnan(nanvec_init))
                  nanvec_init[indices_notnan] = scarcons[indices_notnan]
                  
                  scarlhs = self.topo.project(ns.sampled, ns.scarbasis, ns.x, degree=gaussdegr, constrain=nanvec_init)#, exact_boundaries=True)
              
                #scarsqr  = self.topo.integral('( sampled - scar )^2 d:x' @ ns, degree=gaussdegr)        
                #scarlhs  = solver.optimize(val, scarsqr, tol=1e-6, droptol=1e-15, constrain=scarcons)
                  
                lhs = {val: scarlhs}
                #lhs[val] = scarlhs
                #if irefine: lhs.pop(val[:-1]+str(irefine-1))  # pop previous solution

                
                
                if irefine == reflvl-1: # We are the last hierarchical refinement step -> Regularize the mesh
                  treelog.info("Performing mesh regularization")
                  nrelems_  = len(self.topo.integrate_elementwise(self.geom, degree=0))
                  self.topo = regularize_mesh(self.topo, difference=2) # max 2 levels of difference
                  
                  if nrelems_ != len(self.topo.integrate_elementwise(self.geom, degree=0)): # Additional elements were introduced, recalculate scar array (distance)
                     GaussSample, infarctValues = self.sample_analytic_V2(C=C, Lborder=Lborder, gaussdegr=gaussdegr, improve=improve, saveGpoints=saveGpoints)
                
                     if returnField:
                        treelog("CAUTION!: The scar field is not re-projected onto the regularized mesh (mplement this later on).") 
                         
                  scarGauss = GaussSample.asfunction(infarctValues)
                  
                treelog.info('Number of elements: {}'.format(len(self.topo.integrate_elementwise(self.geom, degree=0)))) # Print number of elements in topology after refinement
    
    
    
    
#                if irefine == reflvl-1: # We are the last hierarchical refinement step -> Perform a final gauss-sampling with a higher quadrature degree and project it onto a higher order degree basis function to produce a better fit visually, this is required for post-processing since the values of scar are used in a different program(not only for our simulations). We do not, however, require the quadrature degree to be the same as in the simulation for this case, since it is only postprocessing.
#                  treelog.info("Performing final post-processing projection")
#                  gaussdegr_final = 10
#                  GaussSample, infarctValues = self.sample_analytic(θ, φ, ξ=ξ, C=C, Lborder=Lborder, gaussdegr=gaussdegr_final, improve=improve, saveGpoints=saveGpoints)
#                  ns.sampledPP = GaussSample.asfunction(infarctValues)
#                  
#                  ## Build suitable basis
#                  ns.scarbasisPP = self.topo.basis(btype, degree=6)
#                  ns.scarPP      = 'scarbasis_n ?{}_n'.format(val)  
#                  ns.scarTruncPP = np.minimum( np.maximum(ns.scarPP, 0), 1) # Truncate to make sure no over or undershoots are obtained after projection
          
                
                if saveVTK:
                   bezier = self.topo.sample('bezier', max(3,int(16/2**irefine)))
                   if not irefine:
                      Xg, scar = bezier.eval(['x_i', 'scarTrunc'] @ ns(**lhs))
                      export.vtk('Scar_{}_nrefine{}'.format(btype,irefine), bezier.tri, Xg, scar=scar)
                   else:
                      # Compute indicator
                      ns.refbasis = self.topo.basis(btype, degree=bdegree)
                      valref      = 'reflhs{}'.format(irefine)
                      ns.ind      = 'refbasis_n ?{}_n'.format(valref) 
                      #indicator   = topo.integral('refbasis_n,k scar_,k d:x' @ ns, degree=bdegree*2).eval(**lhs)
                      indicator   = self.topo.integral('refbasis_n ( scar_,k scar_,k ) d:x' @ ns, degree=bdegree*2).eval(**lhs)
                      lhs[valref] = indicator
                      
                      if self.ablation != None:
                        Xg, scar, indic, asites, deadtissue = bezier.eval(['x_i', 'scarTrunc', 'ind', 'asites', 'deadtissue'] @ ns(**lhs))
                        export.vtk('Scar_{}_nrefine{}'.format(btype,irefine), bezier.tri, Xg, scar=scar, indic=indic, asites=asites, deadtissue=deadtissue)
                        
                      else:
                        Xg, scar, indic = bezier.eval(['x_i', 'scar', 'ind'] @ ns(**lhs))
                        export.vtk('Scar_{}_nrefine{}'.format(btype,irefine), bezier.tri, Xg, scar=scar, indic=indic)


        if returnField:
           return self.topo, ns(scarlhs=scarlhs).scarTrunc, lhs
        else:
           return self.topo, scarGauss
        
    
    
    # Only works for PyGEL3D.version >= 0.3.2
    def combine_scars(self, *manifolds : List[hmesh.Manifold], saveManifold : bool = False, filename : str = "Merged Manifolds.obj"):
        assert len(manifolds) > 1, "provide atleast >=2 scar manifolds to be merged."
        
        # Below only works for pygel3D version > 0.3.2 (requires up-to-date linux)
        manifold0 = manifolds[0]
        #print(dir(manifold0))
        for manifold in manifolds[1:]: 
            manifold0.merge_with(manifold)
            
        if saveManifold:
           hmesh.obj_save(filename, manifold0)
        return manifold0
    
    
    
    def distance_to_manifolds(self, *manifolds : List[hmesh.Manifold], X : np.ndarray = np.ndarray([]), normalize : bool = False):
        """
        Compute the (normilzed) distance of spatial coordinates X to the infarct/scar manifolds.
          Parameters
          ---------- 
            *manifolds  : List of manifold objects
            X           : Spatial coordinates [x,y,z] to which the distance is to be computed
            normalize   : Set True if field should be normalized 

          Returns
          -------
            The (normalized) distance for each point in X
        """
        Distances = np.zeros(len(X)) + 1e10 # Add sufficiently large initial value to initial array
        
        # Calculate values for points within the border zone
        for i, manifold in enumerate(manifolds):
            treelog.info(f"Determining distance to manifold {i+1}")
            distance  = np.abs( hmesh.MeshDistance(manifold).signed_distance(X) ) # Computes the absolute distance to the point (current version of pygel3d==0.3.2 is faulty, so we have to correct for it using np.abs() and with the next lines)
            inside    = hmesh.MeshDistance(manifold).ray_inside_test(X).astype(bool) # Bool array indicating whether we are inside or outside the manifold 
            distance[inside] = 0 
            
            if normalize:
               distance /= manifold.Lborder
               
            Distances = np.minimum( distance , Distances )   
            
        if normalize:
           Distances = np.maximum( 1 -  np.maximum( Distances, 0), 0 )  # Field defined in [0,1]
        else:
           Distances = np.maximum( Distances, 0 )
   
        return Distances
    
    
    def rectangle(self, θ: tuple, φ: tuple, ξ: tuple, C : float = 0.043, Lborder : float = 0.016, savePoints : bool = False, saveManifold : bool = False, filename : str = "Rectangle.obj") -> Union[HierarchicalTopology, function.Array]: 
        """
        Sample the domain and assign scar properties given an analytic description.
          Parameters
          ---------- 
            θ           : Longitudinal (min, max) values, also supports (θminφmin, θmaxφmin, θminφmax, θmaxφmax)
            φ           : Circumferential (min, max) values 
            ξ           : Transmural (min, max) values 
            C           : Focal length of the ellipsoid
            Lborder     : Length of the border zone
            saveManifold: Save the ractangular scar manifold to an .obj file
            savePoints  : Save the points that define the manifold triangulation (mainly for debuggin purposes)
            filename    : Name of the .obj file to save the manifold to

          Returns
          -------
            The manifold object
        """
        
        ## TODO: Allow for individual control of point locations
        if len(θ) == 2:
          θmin, θmax = θ
          θminφmin = θminφmax = θmin 
          θmaxφmin = θmaxφmax = θmax 
        else:
          θminφmin, θmaxφmin, θminφmax, θmaxφmax = θ
          
        φmin, φmax = φ
        ξmin, ξmax = ξ

        # The ordering is important! 
        Xverts = np.array([[θminφmin, φmin, ξmin],
                           [θminφmin, φmin, ξmax],
                           [θmaxφmin, φmin, ξmin],
                           [θmaxφmin, φmin, ξmax],
                           
                           [θminφmax, φmax, ξmin],
                           [θminφmax, φmax, ξmax],
                           [θmaxφmax, φmax, ξmin],
                           [θmaxφmax, φmax, ξmax]])  


        # Make a manifold out of a set of points that define the scar
        Rect   = Rectangle(Xverts)
        XProlscar3D, triProl = Rect.sample()

        XProlscar3D = XProlscar3D[:, [2, 0, 1]] # Reshape [θ, φ, ξ] to [ξ, θ, φ] because of prolate_to_cartesian() requirement
        Object      = Rect.manifold(XProlscar3D, triProl)

        # We are going to change the position of the vertices of the manifold object inplace!
        vertx_pos = Object.positions()
        Xscar     = Ellips_Functs.prolate_to_cartesian(vertx_pos, C)
        vertx_pos[:] = Xscar # Deform manifold vertices inplace
        
        # Store borderzone length
        Object.Lborder = Lborder

        self.manifolds.append(Object)

        # Saving options
        if saveManifold:
           hmesh.obj_save(filename, Object)         
        if savePoints:          
           np.savetxt('GaussPoints.txt', np.concatenate([scar[:,np.newaxis],XCartsample],axis=1), delimiter=',',header='scar, X, Y, Z', comments='') # First save to relevant file
        return Object

        
    def triangle(self, θ: tuple, φ: tuple, ξ: tuple, C : float = 0.043, Lborder : float = 0.016, gaussdegr: int = 5, improve: bool = True, saveGpoints : bool = False, saveManifold : bool = False, filename : str = "Triangle.obj") -> Union[HierarchicalTopology, function.Array]: 
        """
        Sample the domain and assign scar properties given an analytic description.
          Parameters
          ---------- 
            θ           : Longitudinal (min, max) values 
            φ           : Circumferential (min, max) values 
            ξ           : Transmural (min, max) values 
            C           : Focal length of the ellipsoid
            Lborder     : Length of the border zone
            saveManifold: Save the ractangular scar manifold to an .obj file
            savePoints  : Save the points that define the manifold triangulation (mainly for debuggin purposes)
            filename    : Name of the .obj file to save the manifold to

          Returns
          -------
            The manifold object
        """
        

        θmin, θmax = θ
        φmin, φmax = φ
        ξmin, ξmax = ξ
        
        # 3 points that the define the triangle
        P1 = np.array([θmax,φmin])
        P2 = np.array([θmin,0])
        P3 = np.array([θmax,φmax])
        
        # Make a manifold out of a set of points that define the scar
        triang = Triangle(P1, P2, P3, ξ)
        XProlscar3D, triProl = triang.sample()
        
        Object = triang.manifold(XProlscar3D, triProl)

        # We are going to change the position of the vertices of the manifold object inplace!
        vertx_pos = Object.positions()
        Xscar     = Ellips_Functs.prolate_to_cartesian(vertx_pos, C)
        vertx_pos[:] = Xscar # Deform manifold vertices inplace

        # Store borderzone length
        Object.Lborder = Lborder
        
        self.manifolds.append(Object)

   
        if saveManifold:
           hmesh.obj_save(filename, Object)          
        if saveGpoints:          
           np.savetxt('GaussPoints.txt', np.concatenate([scar[:,np.newaxis],XCartsample],axis=1), delimiter=',',header='scar, X, Y, Z', comments='') # First save to relevant file
        return Object
    
        
        
        
        
        
        
        
        
        
        
        
        
        
    def sample_analytic_V2(self, C : float = 0.043, Lborder : float = 0.016, gaussdegr: int = 5, improve: bool = False, saveGpoints : bool = False, normalize : bool =True) -> Union[HierarchicalTopology, function.Array]: 
        """
        Sample the domain and assign scar properties given an analytic description.
          Parameters
          ---------- 
            θ           : Longitudinal (min, max) values 
            φ           : Circumferential (min, max) values 
            ξ           : Transmural (min, max) values 
            C           : Focal length of the ellipsoid
            Lborder     : Length of the border zone
            gaussdegr   : Gauss point sampling degree
            improve     : Improve the sampling strategy of the infarct manifold (recommended = True, but might slow down computation)
            saveGpoints : Save the Gauss quadrature points with the sampled values. This is the ground-truth onto which we project a basis.
            normalize   : Normalize the distance field.

          Returns
          -------
            The infarct sampled points
        """
        
        gaussSample = self.topo.sample('gauss', gaussdegr)
        XCartgauss  = gaussSample.eval(self.geom) # Gauss point coordinates

        scar = self.distance_to_manifolds(*self.manifolds, X=XCartgauss, normalize=normalize)

        if saveGpoints:          
           np.savetxt('GaussPoints.txt', np.concatenate([scar[:,np.newaxis],XCartgauss],axis=1), delimiter=',',header='scar, X, Y, Z', comments='') # First save to relevant file
           
        return gaussSample, scar
        
        
        
        
        
        
        
        
        
        
            
    #TODO Add possibility to define your own infarct shape in the θ-φ coordinate system.
    def sample_analytic(self, θ: tuple, φ: tuple, ξ: tuple, C : float = 0.043, Lborder : float = 0.016, gaussdegr: int = 5, improve: bool = False, saveGpoints : bool = False) -> Union[HierarchicalTopology, function.Array]: 
        """
        Sample the domain and assign scar properties given an analytic description.
          Parameters
          ---------- 
            θ           : Longitudinal (min, max) values 
            φ           : Circumferential (min, max) values 
            ξ           : Transmural (min, max) values 
            C           : Focal length of the ellipsoid
            Lborder     : Length of the border zone
            gaussdegr   : Gauss point sampling degree
            improve     : Improve the sampling strategy of the infarct manifold (recommended = True, but might slow down computation)
            saveGpoints : Save the Gauss quadrature points with the sampled values. This is the ground-truth onto which we project a basis.

          Returns
          -------
            The infarct sampled points
        """
        

        θmin, θmax = θ
        φmin, φmax = φ
        ξmin, ξmax = ξ
        # ξmin = 0  if ξmin == None else ξmin  
        # ξmax = 10 if ξmax == None else ξmax

        gaussSample = self.topo.sample('gauss', gaussdegr)
        XCartgauss  = gaussSample.eval(self.geom) # Gauss point coordinates
        XProlgauss  = Ellips_Functs.cartesian_to_prolate(XCartgauss, C)
        scar        = np.zeros(len(XProlgauss))
        # Within a rectangle
        # θmask = ( XProlgauss[:,1] >= θmin ) & ( XProlgauss[:,1] <= θmax )  # Be careful, we use <= and >= for floats..
        # φmask = ( XProlgauss[:,2] >= φmin ) & ( XProlgauss[:,2] <= φmax )
        # ξmask = ( XProlgauss[:,0] >= ξmin ) & ( XProlgauss[:,0] <= ξmax )
        # mask  = θmask & φmask & ξmask
        
        
        # Make a manifold out of a set of points that define the scar
        XProlscar3D = self.sample_of_triangle_3D(np.array([θmax,φmin]), np.array([θmin,0]), np.array([θmax,φmax]), (ξmin, ξmax), improve=improve) #TODO add functionality such that a transmurally smaller infarct can be generated as well
        Xscar = Ellips_Functs.prolate_to_cartesian(XProlscar3D, C)
        #np.savetxt('Infarct.txt', Xscar, delimiter=',',header='X, Y, Z', comments='')
        DistObject = self.manifoldDistObject(Xscar) # Returns an object-type of the scar manifold from which we can call/calculate the distance to
        
        # Calculate values forpoints within the border zone
        for i, X in enumerate(XCartgauss):
           distance, inside = self.dist(DistObject, X, info=True)
           if inside:
              scar[i] = 1
           elif distance <= Lborder:
              scar[i] = (1-distance/Lborder) 
                    
                    
        
        # Within a simple triangle (quick and dirty way)
        if not improve: 
          mask  = self.point_in_triangle(np.array([θmax,φmin]), np.array([θmax,φmax]), np.array([θmin,0]), XProlgauss[:,1:])
          ξmask = ( XProlgauss[:,0] >= ξmin ) & ( XProlgauss[:,0] <= ξmax ) #TODO If xi is also given 
          mask  = mask & ξmask
          scar[mask] = 1 # set scar tissue to 1

          # Calculate values forpoints within the border zone
          for i, X in enumerate(XCartgauss):
              if mask[i]: # it is the core
                  continue
              else: # It is not the core, and thus find its distance to the core
                 distance = self.dist(DistObject, X)
                 #minimum_dist = np.amin(np.linalg.norm(X-Xscar,axis=1)) 
                 if distance <= Lborder:
                    scar[i] = (1-distance/Lborder)  
                  
                  
        if saveGpoints:          
           np.savetxt('GaussPoints.txt', np.concatenate([scar[:,np.newaxis],XCartgauss],axis=1), delimiter=',',header='scar, X, Y, Z', comments='') # First save to relevant file
        return gaussSample, scar
    
#    @staticmethod
#    def point_in_triangle(X1, X2, X3, Point):
#        n1 = (X2[0]-X1[0])*(Point[:,1]-X1[1])-(X2[1]-X1[1])*(Point[:,0]-X1[0])
#        n2 = (X3[0]-X2[0])*(Point[:,1]-X2[1])-(X3[1]-X2[1])*(Point[:,0]-X2[0])
#        n3 = (X1[0]-X3[0])*(Point[:,1]-X3[1])-(X1[1]-X3[1])*(Point[:,0]-X3[0])
#        return ( (n1<0) & (n2<0) & (n3<0) ) + ( (n1>0) & (n2>0) & (n3>0) )

#    @staticmethod
#    def manifoldDistObject(X):
#        hull  = ConvexHull(X)
#        # Construct PyGEL Manifold from the convex hull
#        m = hmesh.Manifold()
#        for s in hull.simplices:
#            m.add_face(hull.points[s])
#        return hmesh.MeshDistance(m)

#    @staticmethod
#    def manifoldDistObject_V2(points, tri): # Input are points and correponding triangulation (note this triangulation may be non-convex!)
#        # Construct PyGEL Manifold from the convex hull
#        #m = hmesh.Manifold()
#        #m = m.from_triangles(points, tri)
#        m = hmesh.Manifold()
#        for s in tri:
#            m.add_face(points[s])
#
#        #hmesh.obj_save("ScarManifold.obj",m)
#        #np.savetxt('Points.txt', points, delimiter=',',header='X, Y, Z', comments='')
#        #print("succes")
#
#        return m #hmesh.MeshDistance(m)
#        
                
#    @staticmethod
#    def dist(DistObject, point, info=False):
#        # Get the distance to the point
#        # But don't trust its sign, because of possible
#        # wrong orientation of mesh face
#        d = DistObject.signed_distance(point)
#        inside = False # Indicator which says if the point is inside the manifold or not
#        # Correct the sign with ray inside test
#        if DistObject.ray_inside_test(point):
#            inside = True
#            if d > 0:
#                d *= -1
#                print("Should not happen")
#        else:
#            if d < 0:
#                d *= -1
#                print("Should not happen")
#        if info:        
#          return d, inside
#        else:
#          return d  

#    @staticmethod
#    def sample_of_triangle_2D(P1, P2, P3, nsamples=10):
#        # Unpack coordinates of points
#        x1, y1 = P1
#        x2, y2 = P2
#        x3, y3 = P3
#
#        # Slopes of the triangle
#        k1 = (y2-y1)/(x2-x1)
#        k2 = (y3-y2)/(x3-x2)
#        k3 = (y1-y3)/(x1-x3)
#
#        # Expression of the triangle boundaries in 2D
#        f1 = lambda x : k1*x + (y1-k1*x1) # with x in [x1,x2)
#        f2 = lambda x : k2*x + (y2-k2*x2) # with x in [x2,x3)
#        f3 = lambda x : k3*x + (y3-k3*x3) # with x in [x3,x1)
#
#       # Sample the x-coordinate 
#        xsample1 = np.linspace(x1,x2,nsamples, endpoint=False)
#        xsample2 = np.linspace(x2,x3,nsamples, endpoint=False)
#        xsample3 = np.linspace(x3,x1,nsamples, endpoint=False)
#
#        triangleSample = np.concatenate([np.array([f1(xsample1)],xsample1), f1(xsample1), f1(xsample1)])  
#        return triangleSample
    
#    @staticmethod
#    def sample_of_rectangle_3D(Xbounds, Ybounds, Zbounds, lin_dir=0): 
#    
#        u=np.linspace(Xbounds[0], Xbounds[1], 10)
#        v=np.linspace(Ybounds[0], Ybounds[1], 10)
#        w=np.linspace(Zbounds[0], Zbounds[1], 10)
#        
#        Xu,Xv,Xw = np.meshgrid(u,v,w) 
#
#        points3D = np.vstack([Xu.flatten(),Xv.flatten(),Xw.flatten()]).T
#        
#        # Retrieve convex hull from solid (Delaunay) triangulation
#        tri_hull = Delaunay(points3D).convex_hull
#
#        # Create hull object
#        hull_delaunay = Delaunay(points3D[np.unique(tri_hull)])
#        
#        return hull_delaunay.points, hull_delaunay.convex_hull # Return points on the boundary + the connectivity/triangulation of the faces (convex hull)
        
        
#    def sample_of_triangle_3D_V2(self, P1, P2, P3, Z, improve=False): # Sample the 2D triangle in the third dimension z in [0,1]
#        zmin, zmax = Z # Z is the physical domain or range of the third dimension.
#
#        PointsProlateFace  = np.concatenate([P1[:,_], P2[:,_], P3[:,_]], axis=1).T
#        PointsProlateFace1 = np.concatenate([ np.ones((3,1))*zmin, PointsProlateFace ], axis=1)
#        PointsProlateFace2 = np.concatenate([ np.ones((3,1))*zmax, PointsProlateFace ], axis=1)
#        if improve: # We are going to sample the endo- and epicardial faces of the infarct to capture the curved surface of the LV
#           P1, P2, P3 = PointsProlateFace
#           Trianglesample2D   = self.uniform_sample_2Dtriangle(P1, P2, P3) 
#           PointsProlateFace1b = np.concatenate([ np.ones((len(Trianglesample2D),1))*zmin, Trianglesample2D ], axis=1)
#           PointsProlateFace2b = np.concatenate([ np.ones((len(Trianglesample2D),1))*zmax, Trianglesample2D ], axis=1)
#           PointsProlateFace1 = np.concatenate([PointsProlateFace1, PointsProlateFace1b])
#           PointsProlateFace2 = np.concatenate([PointsProlateFace2, PointsProlateFace2b])
#        PointsProlate      = np.concatenate([PointsProlateFace1, PointsProlateFace2])
#        
#        # Create hull object (only works if we have no points inside, otherwise indexing of the tri_hull is incorrect)
#        tri_hull = Delaunay(PointsProlate).convex_hull
#
#        # Create hull object
#        #hull_delaunay = Delaunay(points3D[np.unique(tri_hull)])
#        
#        return PointsProlate, tri_hull
#        
#    
#    def sample_of_triangle_3D(self, P1, P2, P3, Z, improve=False): # Sample the 2D triangle in the third dimension z in [0,1]
#        zmin, zmax = Z # Z is the physical domain or range of the third dimension.
#
#        PointsProlateFace  = np.concatenate([P1[:,_], P2[:,_], P3[:,_]], axis=1).T
#        PointsProlateFace1 = np.concatenate([ np.ones((3,1))*zmin, PointsProlateFace ], axis=1)
#        PointsProlateFace2 = np.concatenate([ np.ones((3,1))*zmax, PointsProlateFace ], axis=1)
#        if improve: # We are going to sample the endo- and epicardial faces of the infarct to capture the curved surface of the LV
#           P1, P2, P3 = PointsProlateFace
#           Trianglesample2D   = self.uniform_sample_2Dtriangle(P1, P2, P3) 
#           PointsProlateFace1b = np.concatenate([ np.ones((len(Trianglesample2D),1))*zmin, Trianglesample2D ], axis=1)
#           PointsProlateFace2b = np.concatenate([ np.ones((len(Trianglesample2D),1))*zmax, Trianglesample2D ], axis=1)
#           PointsProlateFace1 = np.concatenate([PointsProlateFace1, PointsProlateFace1b])
#           PointsProlateFace2 = np.concatenate([PointsProlateFace2, PointsProlateFace2b])
#        PointsProlate      = np.concatenate([PointsProlateFace1, PointsProlateFace2])
#        
#        return PointsProlate
#    
#    def init_basis_functions_cube(self,):
#        # We are going to do an isoparametric transformation of a cube
#        # Returns an evaluable function that outputs the transformation matrix
#        
#        # Basis functions (defined for u,v,w in [0,1])
#        self.basisN1 = lambda u,v,w: (1 - u)*(1 - v)*(1 - w)
#        self.basisN2 = lambda u,v,w: (1 - u)*(1 - v)*w
#        self.basisN3 = lambda u,v,w: u*(1 - v)*(1 - w)
#        self.basisN4 = lambda u,v,w: u*(1 - v)*w
#        
#        self.basisN5 = lambda u,v,w: (1 - u)*v*(1 - w)
#        self.basisN6 = lambda u,v,w: (1 - u)*v*w
#        self.basisN7 = lambda u,v,w: u*v*(1 - w)
#        self.basisN8 = lambda u,v,w: u*v*w
#
#        return
#    
#    def transformation_matrix_cube(self, Xlocal): # 3 x 8 np.array()
#        return np.array([ [ self.basisN1(*Xlocal), 0, 0, self.basisN2(*Xlocal), 0, 0, self.basisN3(*Xlocal), 0, 0, self.basisN4(*Xlocal), 0, 0, self.basisN5(*Xlocal), 0, 0, self.basisN6(*Xlocal), 0, 0, self.basisN7(*Xlocal), 0, 0, self.basisN8(*Xlocal), 0, 0 ],
#                   [ 0, self.basisN1(*Xlocal), 0, 0, self.basisN2(*Xlocal), 0, 0, self.basisN3(*Xlocal), 0, 0, self.basisN4(*Xlocal), 0, 0, self.basisN5(*Xlocal), 0, 0, self.basisN6(*Xlocal), 0, 0, self.basisN7(*Xlocal), 0, 0, self.basisN8(*Xlocal), 0 ],
#                   [ 0, 0, self.basisN1(*Xlocal), 0, 0, self.basisN2(*Xlocal), 0, 0, self.basisN3(*Xlocal), 0, 0, self.basisN4(*Xlocal), 0, 0, self.basisN5(*Xlocal), 0, 0, self.basisN6(*Xlocal), 0, 0, self.basisN7(*Xlocal), 0, 0, self.basisN8(*Xlocal)] ])
    
    
    
#    @staticmethod
#    def uniform_sample_2Dtriangle(P1, P2, P3, nsample : int = 50):
#        v1 = P2-P1
#        v2 = P3-P1
#        A  = np.concatenate([v1[:,_],v2[:,_]],axis=1) # Linear tranformation matrix
#
#        x   = np.linspace(0,1,nsample)
#        for i, ix in enumerate(x):
#            if i == 0:
#                Xsub = np.concatenate([x,np.ones(len(x))*ix]).reshape(2,-1).T 
#                X    = Xsub.copy()   
#            else:
#                Xsub = np.concatenate([x[:-i],np.ones(len(x[:-i]))*ix]).reshape(2,-1).T   
#                X = np.concatenate([X, Xsub])
#        return np.dot(A,X.T).T+P1 
        
        
        
        
        


   
       

class Rectangle:
      """
      
        Class used to defined rectangular-shaped infarct geometries. Yes, it is defined in 3D and should be named 'cuboid', 
              but the 'rectangle' is based on the θφ-coordinate system so we stick with this. Its goal is to create a manifold 
              object (PyGEL3D module) by sampling the object in points and defining the corresponding tri-connectivity. 
              Once these are defined, we can construct our manifold.
              
      """
        
      def __init__(self, Xvertices):
          """
          Input: Vertex bounds, can be of size 2 or 4. Size 2 indicates [min, max] values in the specific direction. 
                 Size 4 indicates individual vertex value.  
          """
          self.Xverts = Xvertices
          self._init_basisfunctions()
          return
      
      # Do thi scorrection outside of this class, it is too specific
#      def _reformat_input(self, X, tag="X"):
#          if len(X) == 2:
#            Xmin, Xmax = Xverts
#            XminYmin = XminYmax = Xmin  
#            XmaxYmin = XmaxYmax = Ymax 
#          elif len(X)==4:
#            XminYmin, XmaxYmin, XminYmax, XmaxYmax = X
#          else:
#            raise ValueError(f"Incorrect length '{len(X)}' for input {tag} of the rectangle, only lengths of 2 or 4 are supported.")
#          return XminYmin, XmaxYmin, XminYmax, XmaxYmax
       
      def sample(self, n=10):
          return self._sample_3D(n=n)
                  
      def _sample_3D(self, n=10): 
        """
          Sampling the 3D rectangle in a grid format defined in [0,1] for all dimensions given a number of refinements n. 
          The refinements are then used to construct triangulated data. 
        """
        
        u=np.linspace(0, 1, n)
        v=np.linspace(0, 1, n)
        w=np.linspace(0, 1, n)
        
        Xu,Xv,Xw = np.meshgrid(u,v,w) 

        Xlocal = np.vstack([Xu.flatten(),Xv.flatten(),Xw.flatten()]).T
        
        # Retrieve convex hull from solid (Delaunay) triangulation
        tri_hull = Delaunay(Xlocal).convex_hull

        # Create hull object
        Xlocal_b = Xlocal[np.unique(tri_hull)]
        Xconn_b  = Delaunay(Xlocal_b).convex_hull
        
        
                   
        # Transform the points accordingly
        Xglobal_b = np.zeros(Xlocal_b.shape)
        for i, (Uglobal, Vglobal, Wglobal) in enumerate(Xlocal_b):
            Xglobal_b[i,:] = np.dot( self._transmatrix([Uglobal, Vglobal, Wglobal]), self.Xverts.flatten()) # Xglobal = TransMatrix * Xlocal

        return Xglobal_b, Xconn_b # Return points on the boundary + the connectivity/triangulation of the faces (convex hull)
   
   
      def _init_basisfunctions(self,):
          """
            Useful function that defines the local/parametric basis functions of a cube. Used to map the sampled rectangle points to corresponding vertex input.
          """
          
          # Basis functions (defined for u,v,w in [0,1])
          self.basisN1 = lambda u,v,w: (1 - u)*(1 - v)*(1 - w)
          self.basisN2 = lambda u,v,w: (1 - u)*(1 - v)*w
          self.basisN3 = lambda u,v,w: u*(1 - v)*(1 - w)
          self.basisN4 = lambda u,v,w: u*(1 - v)*w
          
          self.basisN5 = lambda u,v,w: (1 - u)*v*(1 - w)
          self.basisN6 = lambda u,v,w: (1 - u)*v*w
          self.basisN7 = lambda u,v,w: u*v*(1 - w)
          self.basisN8 = lambda u,v,w: u*v*w
  
          return self
      
      def _transmatrix(self, Xlocal):
          """
            Returns the transformation matrix of the cube [3 x 8] (3 dimensions, 8 vertices)
          """
          return np.array([ [ self.basisN1(*Xlocal), 0, 0, self.basisN2(*Xlocal), 0, 0, self.basisN3(*Xlocal), 0, 0, self.basisN4(*Xlocal), 0, 0, self.basisN5(*Xlocal), 0, 0, self.basisN6(*Xlocal), 0, 0, self.basisN7(*Xlocal), 0, 0, self.basisN8(*Xlocal), 0, 0 ],
                     [ 0, self.basisN1(*Xlocal), 0, 0, self.basisN2(*Xlocal), 0, 0, self.basisN3(*Xlocal), 0, 0, self.basisN4(*Xlocal), 0, 0, self.basisN5(*Xlocal), 0, 0, self.basisN6(*Xlocal), 0, 0, self.basisN7(*Xlocal), 0, 0, self.basisN8(*Xlocal), 0 ],
                     [ 0, 0, self.basisN1(*Xlocal), 0, 0, self.basisN2(*Xlocal), 0, 0, self.basisN3(*Xlocal), 0, 0, self.basisN4(*Xlocal), 0, 0, self.basisN5(*Xlocal), 0, 0, self.basisN6(*Xlocal), 0, 0, self.basisN7(*Xlocal), 0, 0, self.basisN8(*Xlocal)] ])
    
          
      @staticmethod    
      def manifold(points, tri): # Input are points and correponding triangulation (note this triangulation may be non-convex!)
          """
            Create a Manifold object (PyGEL3D module) based on a given point set and the corresponding triangulation/connectivity.
          """
          m = hmesh.Manifold()
          for s in tri:
              m.add_face(points[s])

          return m


    
             
          
class Triangle:
      """
      
        Class used to defined triangular-shaped infarct geometries. Yes, it is defined in 3D and should be named 'pyramid' 
              (it is not a tetrahedron we define), but the 'triangle' is based on the θφ-coordinate system so we stick 
              with this. Its goal is to create a manifold object (PyGEL3D module) by sampling the object in points and 
              defining the corresponding tri-connectivity. Once these are defined, we can construct our manifold.
              
      """
      def __init__(self, P1, P2, P3, Z):
          """
          Input: Triangle points (x,y) and the Zbounds (Zmin, Zmax) 
          """
          self.P1 = P1
          self.P2 = P2
          self.P3 = P3
          self.Zmin, self.Zmax = Z
          return
      
      def sample(self,):
          return self._sample_3D()
          
          
      @staticmethod
      def _sample_2D(P1, P2, P3, n=50):
          """
              Samples a 2D triangle in a uniform manner.
          """
          v1 = P2-P1
          v2 = P3-P1
          A  = np.concatenate([v1[:,_],v2[:,_]],axis=1) # Linear tranformation matrix
  
          x   = np.linspace(0,1,n)
          for i, ix in enumerate(x):
              if i == 0:
                  Xsub = np.concatenate([x,np.ones(len(x))*ix]).reshape(2,-1).T 
                  X    = Xsub.copy()   
              else:
                  Xsub = np.concatenate([x[:-i],np.ones(len(x[:-i]))*ix]).reshape(2,-1).T   
                  X = np.concatenate([X, Xsub])
          return np.dot(A,X.T).T+P1 
            
            
            
      def _sample_3D(self,): # Sample the 2D triangle in the third dimension z in [0,1]
          """
            Samples the 'pyramid' in 3D and returns the boundary points incl. the connectivity
          """
          PointsProlateFace  = np.concatenate([self.P1[:,_], self.P2[:,_], self.P3[:,_]], axis=1).T
          PointsProlateFace1 = np.concatenate([ np.ones((3,1))*self.Zmin, PointsProlateFace ], axis=1)
          PointsProlateFace2 = np.concatenate([ np.ones((3,1))*self.Zmax, PointsProlateFace ], axis=1)
          
          
          # To capture the curvature of the triangle, sample the triangle uniformly
          Trianglesample2D   = self._sample_2D( self.P1, self.P2, self.P3 ) 
          PointsProlateFace1b = np.concatenate([ np.ones((len(Trianglesample2D),1))*self.Zmin, Trianglesample2D ], axis=1)
          PointsProlateFace2b = np.concatenate([ np.ones((len(Trianglesample2D),1))*self.Zmax, Trianglesample2D ], axis=1)
          PointsProlateFace1 = np.concatenate([PointsProlateFace1, PointsProlateFace1b])
          PointsProlateFace2 = np.concatenate([PointsProlateFace2, PointsProlateFace2b])
          PointsProlate      = np.concatenate([PointsProlateFace1, PointsProlateFace2])
          
          # Create hull object (only works if we have no points inside, otherwise indexing of the tri_hull is incorrect)
          tri_hull = Delaunay(PointsProlate).convex_hull
  
          
          return PointsProlate, tri_hull
          
      @staticmethod    
      def manifold(points, tri): # Input are points and correponding triangulation (note this triangulation may be non-convex!)
          """
            Create a Manifold object (PyGEL3D module) based on a given point set and the corresponding triangulation/connectivity.
          """
          m = hmesh.Manifold()
          for s in tri:
              m.add_face(points[s])

          return m
          
          