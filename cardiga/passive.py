# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 14:33:05 2021

@author: s146407
"""
import treelog, numpy
from nutils import solver, function


class Passive_models(): # Define the strings used in the namespace

    def __init__(self, ns, topo, geom, g_type, m_type, boundaries, constants={}): # Model type: string
        self.topo       = topo
        self.geom       = geom
        self.m_type     = m_type # model type
        self.g_type     = g_type # geometry type lv/bv
        self.ns         = ns 
        #self.constants  = constants
        self.boundaries = boundaries
        return
        
    def __str__(self,): # Print the constants used
        treelog.info('The following constants are used for the {} material model'.format(self.m_type))
        # Construct string to print
        print(self.constants)
        return

    def construct(self,const_input, btype, bdegree, quad_degree, surpressJump=False, interfaceClass=None, prestress={}):
        # First initialize material measures, adds useful quantities to the namespace
        self.init_material_measures(btype=btype, bdegree=bdegree, mixed=False, prestress=prestress) # TODO set mixed=False to a variable
    
        ## Construct a simple linear elastic model
        if self.m_type == 'linear':
            with treelog.context('Passive model (linear)'):
                res, targ, const = self.linear_elastic(const_input)
                
        ## Construct an incompressible linear elastic model (mixed formulation)
        elif self.m_type == 'linear-inc':
            with treelog.context('Passive model (linear incompressible)'):
                res, targ, const = self.linear_elastic(const_input,incompr=True)  
                
        ## Construct an incompressible Fung-type model (mixed formulation)
        elif self.m_type == 'fung':
            with treelog.context('Fung model (non-linear)'):
                res, targ, const = self.fung_nonlinear_elastic(const_input)
                
        ## Construct a nearly incompressible Fung-type model (single field formulation)
        elif self.m_type == 'fung-quasi':
            with treelog.context('Fung model quasi incompressible (non-linear)'):
                res, targ, const = self.fung_nonlinear_elastic(const_input,quasi_incompr=True)
                
        ## Construct the incompressible Bovendeerd constitutive model (mixed formulation)
        elif self.m_type == 'bovendeerd':
            with treelog.context('Bovendeerd model (non-linear)'):
                res, targ, const = self.BovendeerdMaterial(const_input,btype=btype,bdegree=bdegree, quad_degree=quad_degree, surpressJump=surpressJump, interfaceClass=interfaceClass)
             
        ## Construct the nearly incompressible Bovendeerd constitutive model (single field formulation)   
        elif self.m_type == 'bovendeerd-quasi':
            with treelog.context('Bovendeerd model (non-linear)'):
                res, targ, const = self.BovendeerdMaterial(const_input,quasi_incompr=True,btype=btype,bdegree=bdegree, quad_degree=quad_degree, surpressJump=surpressJump, interfaceClass=interfaceClass)
                
        else:
            raise Exception("Unknown model '{}' type".format(self.m_type)) 
            
        return res, targ, const
    
    def init_material_measures(self, btype='spline', bdegree=3, mixed=False, prestress={}):
        
        ## Construct basis functions for deformation and pressure field (if mixed formulation is specified)
        if mixed:
          self.ns.ubasis = self.topo.basis(btype, degree=bdegree,   continuity=-2).vector(self.topo.ndims)
          self.ns.pbasis = self.topo.basis(btype, degree=bdegree-1, continuity=-2) 
          # Current time step
          self.ns.u_i    = 'ubasis_ni ?ulhs_n' # Deformation vector
          self.ns.p      = 'pbasis_ni ?plhs_n' # Hydrostatic pressure
          # Previous time step
          self.ns.u0_i   = 'ubasis_ni ?ulhs0_n'
          self.ns.p0     = 'pbasis_ni ?plhs0_n'
        else:
          self.ns.ubasis = self.topo.basis(btype, degree=bdegree).vector(self.topo.ndims)
          # Current time step
          self.ns.u_i    = 'ubasis_ni ?ulhs_n'
          # Previous time step  
          self.ns.u0_i   = 'ubasis_ni ?ulhs0_n' 
        
        
        
        ## Check if geom or 'x_i' is already an attribute of the namespace
        if not hasattr(self.ns, "x"):
           self.ns.x = self.geom
           
        ## Define continuum measures (both current and previous time step)
        if not prestress: # There is no pre-stressing of any kind
        
          # Directional components & derivatives (used for supressing rigid body motion)
          self.ns.Ux       = 'u_0'
          self.ns.Uy       = 'u_1'
          self.ns.GradUx_i = 'u_0,i'
          self.ns.GradUy_i = 'u_1,i'
           
        
          # Current time step
          self.ns.X_i   = 'x_i + u_i'      # Deformed configuration
          self.ns.F_ij  = 'δ_ij + u_i,j'   # Deformation gradient tensor

          # Previous time step
          self.ns.X0_i  = 'x_i + u0_i'    # Deformed configuration
          self.ns.F0_ij = 'δ_ij + u0_i,j' # Deformation gradient tensor   

          # Fiber orientation
          self.ns.ef0_i = 'ef_i'
          
          # Specify the constitutive model measures
          self.ns.C_ij  = 'F_ki F_kj'            # C = F^T F     ; Right Cauchy-Green deformation tensor (Cauchy strain tensor)
          self.ns.B_ij  = 'F_ik F_jk'            # B = F F^T     ; 
          self.ns.E_ij  = '0.5 ( C_ij - δ_ij )'  # E = 0.5(C - I); Green-Lagrang strain tensor 
          self.ns.Ef    = 'ef0_i E_ij ef0_j'     # Ef = ef E ef  ; Fiber Green-Lagrange strain
          
          # Inverses
          self.ns.Finv  = numpy.linalg.inv( self.ns.F )
          self.ns.Finv0 = numpy.linalg.inv( self.ns.F0 )
        
          # Determinants
          self.ns.detF  = numpy.linalg.det(  self.ns.F  )
          self.ns.detF0 = numpy.linalg.det(  self.ns.F0 )            
            
          
          # Specify the integral (pull/pushback operators)
          self.ns.detFint  = self.ns.detF
          self.ns.detFint0 = self.ns.detF0
          self.ns.Finvint  = self.ns.Finv
          self.ns.Finvint0 = self.ns.Finv0
            
        elif prestress["type"] == "GPA": # We have a GPA (generalized prestressing algorithm) setting
    
          # Set Fpre quantity
          self.ns.upre_i     = 'ubasis_ni ?uprelhs_n'
          self.ns.Fpreinv_ij = 'δ_ij - upre_i,j'
          self.ns.Fpre       = numpy.linalg.inv(self.ns.Fpreinv)
    
    
          # Directional components & derivatives (used for supressing rigid body motion)
          self.ns.Ux       = 'u_0 + upre_0'
          self.ns.Uy       = 'u_1 + upre_1'
          self.ns.GradUx_i = 'u_0,i + upre_0,i'
          self.ns.GradUy_i = 'u_1,i + upre_1,i'
          
          # Current time step
          self.ns.X_i     = 'x_i + u_i'       # Deformed configuration
          self.ns.Farb_ij = 'δ_ij + u_i,j'    # Arbitrary deformation gradient tensor
          self.ns.F_ij    = 'Farb_ik Fpre_kj' # Deformation gradient tensor
          
          # Previous time step
          self.ns.X0_i     = 'x_i + u0_i'       # Deformed configuration
          self.ns.Farb0_ij = 'δ_ij + u0_i,j'    # Arbitrary deformation gradient tensor   
          self.ns.F0_ij    = 'Farb0_ik Fpre_kj' # Deformation gradient tensor

          # Inverses
          self.ns.Finv     = numpy.linalg.inv( self.ns.F  )
          self.ns.Finv0    = numpy.linalg.inv( self.ns.F0 )
          self.ns.Fpreinv  = numpy.linalg.inv( self.ns.Fpre )
          self.ns.Farbinv  = numpy.linalg.inv( self.ns.Farb )
          self.ns.Farbinv0 = numpy.linalg.inv( self.ns.Farb0 )
          
          # Fiber orientation
          self.ns.ef0_i = 'Fpreinv_ij ef_j / sqrt( ef_m Fpreinv_km Fpreinv_kl ef_l )'

          # Specify the constitutive model measures
          self.ns.C_ij  = 'F_ki F_kj'            # C = F^T F     ; Right Cauchy-Green deformation tensor (Cauchy strain tensor)
          self.ns.B_ij  = 'F_ik F_jk'            # B = F F^T     ; 
          self.ns.E_ij  = '0.5 ( C_ij - δ_ij )'  # E = 0.5(C - I); Green-Lagrang strain tensor 
          self.ns.Ef    = 'ef0_i E_ij ef0_j'       # Ef = ef E ef  ; Fiber Green-Lagrange strain
          
          # Determinants
          self.ns.detF     = numpy.linalg.det( self.ns.F    )
          self.ns.detFpre  = numpy.linalg.det( self.ns.Fpre )
          self.ns.detF0    = numpy.linalg.det( self.ns.F0   )            
          self.ns.detFarb  = numpy.linalg.det( self.ns.Farb )
          self.ns.detFarb0 = numpy.linalg.det( self.ns.Farb0 )
                      
          # Specify the integral (pull/pushback operators)
          self.ns.detFint  = self.ns.detFarb
          self.ns.detFint0 = self.ns.detFarb0
          self.ns.Finvint  = self.ns.Farbinv
          self.ns.Finvint0 = self.ns.Farbinv0
          
        else:
          raise ValueError()#f"Unknown prestressing type specified '{prestress["type"]}'")
        
        return
    
    def get_cons(self,):
        if not self.boundaries:
            cons = None
            
        else:    
            for boundtype, info in self.boundaries.items():
              boundary, value = info # get boundary type and corresponding value
              if  boundtype == 'fixed (normal)':
                  sign   = ' ' if numpy.sign(value) == 0 or numpy.sign(value) == 1 else str(numpy.sign(value))[0]
                  integr = '( u_i n_i - ( ' + sign + ' ' + str(value) + ' ) )^2 d:x' 
                  sqr    = self.topo.boundary[boundary].integral(integr @ self.ns, degree=4)
              elif boundtype == 'fixed (total)':
                  integr = '( u_i u_i ) d:x' 
                  #print(integr)
                  sqr    = self.topo.boundary[boundary].integral(integr @ self.ns, degree=4)                
              elif boundtype == 'rigid body' and value == 'nodal': # Specific for left-ventricle geometry only!
                  assert self.g_type == 'lv', "Nodal rigid body motion constrain only supports left ventricle geometry"
                  sqr += self.topo['patch0'].boundary['right'].boundary['front'].boundary['top'].integral(' ( u_0 u_0 ) d:x' @ self.ns, degree=4)  
                  sqr += self.topo['patch0'].boundary['right'].boundary['front'].boundary['bottom'].integral(' ( u_1 u_1 ) d:x' @ self.ns, degree=4)  
                  sqr += self.topo['patch2'].boundary['right'].boundary['front'].boundary['top'].integral(' ( u_1 u_1 ) d:x' @ self.ns, degree=4)                  
            #   else:
            #       raise Exception(f"Unknown boundary condition {boundtype} specified at {boundary}")  
            with treelog.context('passive model (constraint)'):
                cons = solver.optimize('ulhs', sqr, tol=1e-6, droptol=1e-15)            

        return dict( ulhs = cons )

    def get_initial(self,):
        with treelog.context('Passive model (initial)'):
            sqr  = self.topo.integral('u0_i u0_i d:x' @ self.ns, degree=4)                
            init = solver.optimize('ulhs0', sqr, tol=1e-6, droptol=1e-15)
        return {'ulhs':init, 'ulhs0':init}
    
  
    def interface_jump_res(self, interfaceClass,  nonlinear=False, βjump=100, gaussdeg=5):
    
        # Determine the element size (initialize basis functions for this)
        refifaces = interfaceClass.refifaces # Interface topology
           
        self.ns.βjump =  βjump
        # Compute average stiffness (can be local, but this is problematic when the stiffness is only defined in the interior quadrature points , not on the interfaces)
        #average_a0 = self.topo.integrate('a0 d:x' @ self.ns, degree=gaussdeg)/self.topo.integrate('d:x' @ self.ns, degree=gaussdeg)
        #print(f"average a0: {average_a0}")
        average_a0 = 800
        self.ns.βJUMP = self.ns.βjump * average_a0 * self.ns.elemf #'βjump a0 elemf' # jump parameter

        self.ns.jumpE = function.jump(self.ns.E)
        self.ns.jumpN = function.jump(function.grad(self.ns.ubasis, self.geom))
        
        if nonlinear:
          res = refifaces.integral('βJUMP ( Finvint_ti jumpN_ntj n_j ) ( jumpE_ik n_s Finvint_sk ) detFint d:x' @ self.ns, degree=gaussdeg)
        else:
          res = refifaces.integral('βJUMP ( jumpN_nik n_k ) ( jumpE_ij n_j ) d:x' @ self.ns, degree=gaussdeg) 
        return res
        
    def boundary_conditions(self, nonlinear=False, gaussdeg=5):
    
        res_list = [] # Store all residuals in a list
        ## TODO Add support for defining same bc for multiple boundaries
        for boundtype, info in self.boundaries.items():
          boundary, value = info # get boundary type and corresponding value
          
          ## Traction boundary condition
          ## TODO Make a distinction between normal traction and arbitrary vector
          if boundtype == 'traction':
              if nonlinear:
                integr = '( ubasis_nj ( {} ) Finvint_ij detFint ) d:x'.format(value)
              else:
                integr = 'ubasis_ni ( {} ) d:x'.format(value)
              res_list.append(-self.topo.boundary[boundary].integral(integr @ self.ns, degree=gaussdeg))  
              
          ## Robin boundary condition as a linear spring
          elif boundtype == 'spring (alldirect)':
              self.ns.Kpericar = value
              if nonlinear:
                res_list.append(-self.topo.boundary[boundary].integral('-Kpericar ubasis_ni u_i detFint d:x' @ self.ns, degree=gaussdeg))
              else:
                res_list.append(-self.topo.boundary[boundary].integral('-Kpericar ubasis_ni u_i d:x' @ self.ns, degree=gaussdeg))
                

        if len(res_list) == 0:
           return False, []
           
        elif len(res_list) == 1:
           return True, res_list[0]
           
        else:  
           res = res_list[0]
           for resl in res_list[1:]:
             res += resl 
           return True, res
        
    def surpress_rigid_body(self, nonlinear=False, gaussdeg=5):
    
        ldefined  = False
        residuals = {}
        for boundtype, info in self.boundaries.items():
          boundary, value = info # get boundary type and corresponding value
          
          if boundtype == 'rigid body' and value == 'nullspace': # Constrain x- and y-directions by default!
           if not ldefined:
               ldefined = True
               self.ns.lxbasis  = [1]
               self.ns.lybasis  = [1]
               self.ns.lxybasis = [1]
               self.ns.lx  = 'lxbasis_n ?lhslx_n'
               self.ns.ly  = 'lybasis_n ?lhsly_n'  
               self.ns.lxy = 'lxybasis_n ?lhslxy_n' 
               
               if nonlinear:
                #  reslx  = self.topo.boundary[boundary].integral('( lxbasis_n u_0 ) detFint d:x' @ self.ns, degree=gaussdeg) # Total x-displacement
                #  resly  = self.topo.boundary[boundary].integral('( lybasis_n u_1 ) detFint d:x' @ self.ns, degree=gaussdeg) # Total y-displacement    
                #  reslxy = self.topo.boundary[boundary].integral('lxybasis_n ( Finvint_j0 u_1,j - Finvint_i1 u_0,i ) detFint d:x' @ self.ns, degree=gaussdeg) # Total rotation
                #  resu   = self.topo.boundary[boundary].integral('( lx ubasis_n0 + ly ubasis_n1 ) detFint d:x' @ self.ns, degree=gaussdeg) # Translation
                #  resu  += self.topo.boundary[boundary].integral('lxy ( Finvint_j0 ubasis_n1,j - Finvint_i1 ubasis_n0,i ) detFint d:x' @ self.ns, degree=gaussdeg) # Rotation
#                 reslx  = self.topo.boundary[boundary].integral('( lxbasis_n ( u_0 + upre_0 ) ) detFint d:x' @ self.ns, degree=gaussdeg) # Total x-displacement
#                 resly  = self.topo.boundary[boundary].integral('( lybasis_n ( u_1 + upre_1 ) ) detFint d:x' @ self.ns, degree=gaussdeg) # Total y-displacement    
#                 reslxy = self.topo.boundary[boundary].integral('lxybasis_n ( Finvint_j0 ( u_1,j + upre_1,j ) - Finvint_i1 ( u_0,i + upre_0,i ) ) detFint d:x' @ self.ns, degree=gaussdeg) # Total rotation
#                 resu   = self.topo.boundary[boundary].integral('( lx ubasis_n0 + ly ubasis_n1 ) detFint d:x' @ self.ns, degree=gaussdeg) # Translation
#                 resu  += self.topo.boundary[boundary].integral('lxy ( Finvint_j0 ubasis_n1,j - Finvint_i1 ubasis_n0,i ) detFint d:x' @ self.ns, degree=gaussdeg) # Rotation            


                 reslx  = self.topo.boundary[boundary].integral('( lxbasis_n Ux ) detFint d:x' @ self.ns, degree=gaussdeg) # Total x-displacement
                 resly  = self.topo.boundary[boundary].integral('( lybasis_n Uy ) detFint d:x' @ self.ns, degree=gaussdeg) # Total y-displacement    
                 reslxy = self.topo.boundary[boundary].integral('lxybasis_n ( Finvint_j0 GradUy_j - Finvint_i1 GradUx_i ) detFint d:x' @ self.ns, degree=gaussdeg) # Total rotation
                 resu   = self.topo.boundary[boundary].integral('( lx ubasis_n0 + ly ubasis_n1 ) detFint d:x' @ self.ns, degree=gaussdeg) # Translation
                 resu  += self.topo.boundary[boundary].integral('lxy ( Finvint_j0 ubasis_n1,j - Finvint_i1 ubasis_n0,i ) detFint d:x' @ self.ns, degree=gaussdeg) # Rotation                      
               else:
                 #  reslx  = self.topo.boundary[boundary].integral('( lxbasis_n u_0 ) d:x' @ self.ns, degree=gaussdeg) # Total x-displacement
                 #  resly  = self.topo.boundary[boundary].integral('( lybasis_n u_1 ) d:x' @ self.ns, degree=gaussdeg) # Total y-displacement
                 #  # reslxy = self.topo.boundary[boundary].integral('lxybasis_n ( u_1,0 - u_0,1 ) d:x' @ self.ns, degree=gaussdeg) # Total rotation
                 #  reslxy = self.topo.boundary[boundary].integral('lxybasis_n ( Fpre_i0 u_1,i - Fpre_j1 u_0,j ) d:x' @ self.ns, degree=gaussdeg) # Total rotation
                 #  resu   = self.topo.boundary[boundary].integral('( lx ubasis_n0 + ly ubasis_n1 ) d:x' @ self.ns, degree=gaussdeg) # Translation
                 #  #resu  += self.topo.boundary[boundary].integral('lxy ( ubasis_n1,0 - ubasis_n0,1 ) d:x' @ self.ns, degree=gaussdeg) # Rotation
                 #  resu  += self.topo.boundary[boundary].integral('lxy ( Fpre_i0 ubasis_n1,i - Fpre_j1 ubasis_n0,j ) d:x' @ self.ns, degree=gaussdeg) # Rotation

#                 reslx  = self.topo.boundary[boundary].integral('( lxbasis_n ( u_0 + upre_0 ) ) d:x' @ self.ns, degree=gaussdeg) # Total x-displacement
#                 resly  = self.topo.boundary[boundary].integral('( lybasis_n ( u_1 + upre_1 ) ) d:x' @ self.ns, degree=gaussdeg) # Total y-displacement    
#                 reslxy = self.topo.boundary[boundary].integral('lxybasis_n ( ( u_1,0 + upre_1,0 ) - ( u_0,1 + upre_0,1 ) ) d:x' @ self.ns, degree=gaussdeg) # Total rotation
#                 resu   = self.topo.boundary[boundary].integral('( lx ubasis_n0 + ly ubasis_n1 ) d:x' @ self.ns, degree=gaussdeg) # Translation
#                 resu  += self.topo.boundary[boundary].integral('lxy ( ubasis_n1,0 - ubasis_n0,1 ) d:x' @ self.ns, degree=gaussdeg) # Rotation   
              
                 reslx  = self.topo.boundary[boundary].integral('( lxbasis_n Ux ) d:x' @ self.ns, degree=gaussdeg) # Total x-displacement
                 resly  = self.topo.boundary[boundary].integral('( lybasis_n Uy ) d:x' @ self.ns, degree=gaussdeg) # Total y-displacement    
                 reslxy = self.topo.boundary[boundary].integral('lxybasis_n ( GradUy_0 - GradUx_1 ) d:x' @ self.ns, degree=gaussdeg) # Total rotation
                 resu   = self.topo.boundary[boundary].integral('( lx ubasis_n0 + ly ubasis_n1 ) d:x' @ self.ns, degree=gaussdeg) # Translation
                 resu  += self.topo.boundary[boundary].integral('lxy ( ubasis_n1,0 - ubasis_n0,1 ) d:x' @ self.ns, degree=gaussdeg) # Rotation  
                 
               residuals = {"resu" : resu, "reslx" : reslx, "resly" : resly, "reslxy" : reslxy}
           else:
               raise NotImplementedError("Can only surpress 1 boundary for rigid body motions.")
               
        return ldefined, residuals
           
        
    def linear_elastic(self,const_input,rigidb_cons=False,incompr=False):
        # ns    = function.Namespace()
        # ns.x  = self.geom
        treelog.info("Constructing residual")
        
        
        # Create constants dict
        constants = {'k'      : 0.1,
                     'poisson': 0.3}
        constants.update(const_input)  # Update if necessary    
        const     = {'linear': constants}
        
        # Assign values to namespace
        #ns.basis     = self.topo.basis('spline', degree=2, patchcontinuous=True).vector(3)
        self.ns.ubasis     = self.topo.basis('spline', degree=2, continuity=-2).vector(self.topo.ndims)
        if rigidb_cons:
            self.ns.lrgbasis   = [1.]
            self.ns.lrg        = 'lrgbasis_n ?lrglhs_n' 
            self.rbmbound      = self.topo.boundary['rbm'].boundary['top']
            Xrbm      = self.rbmbound.boundary.sample('gauss',degree=1).eval(self.geom)
            assert Xrbm.shape == (2,3)
            self.ns.Xrbm = numpy.cross(Xrbm[0,:] - Xrbm[1,:], [0,0,1])
            numpy.testing.assert_allclose(0., self.ns.Xrbm.prepare_eval(npoints=None).eval()[-1], rtol=0, atol=1e-10)
            
            
            
        ## Provide it in strong format!
        self.ns.poisson   = constants['poisson']
        self.ns.k         = constants['k'] 
        self.ns.u_i       = 'ubasis_ni ?ulhs_n'
        self.ns.u0_i      = 'ubasis_ni ?ulhs0_n'
        
        self.ns.X_i       = 'x_i + u_i'
        self.ns.X0_i      = 'x_i + u0_i'
        self.ns.E         = 10*1e4
        self.ns.lmbda     = 'E / 3'
        self.ns.G         = 'E / ( 2 ( 1 + poisson ) )'
        self.ns.strain_ij = '(u_i,j + u_j,i) / 2'
        self.ns.F_ij      = 'X_i,j'
        self.ns.F0_ij     = 'X0_i,j'


        if incompr:
            self.ns.pbasis    = self.topo.basis('spline', degree=1, continuity=-2)
            self.ns.p         = 'pbasis_n ?plhs_n'
            self.ns.p0        = 'pbasis_n ?plhs0_n'
            self.ns.stress_ij = '- p δ_ij + 2 G strain_ij'
            resp         = self.topo.integral('( pbasis_n u_i,i ) d:x' @ self.ns, degree=4)
        else:
            self.ns.stress_ij = 'lmbda strain_kk δ_ij + 2 G strain_ij'
        
        # Create residual
        resu  = self.topo.integral('ubasis_ni,j stress_ij d:x' @ self.ns, degree=4)    
            
        ## Check if traction boundary is specified
        for tracb in list(self.boundaries):
            if self.boundaries[tracb][0] == 'traction':
                integr = '( ' + self.boundaries[tracb][1] + ' )' + ' ubasis_ni d:x'
                resu  -= self.topo.boundary[tracb].integral(integr @ self.ns, degree=4)
                self.boundaries.pop(tracb)    
                
        if rigidb_cons:
            reslrg  = self.rbmbound.boundary['left'].integral('u_i Xrbm_i lrgbasis_n' @ self.ns, degree=1)
            resu   += self.rbmbound.boundary['left'].integral('ubasis_ni Xrbm_i lrg' @ self.ns, degree=1)
            
        # if incompr:   
        #     return [resu,resp], ['ulhs','plhs'], const
        # else:
        #     return [resu], ['ulhs'], const
        if incompr:   
            if rigidb_cons:
                return {'resu':resu,'resp':resp,'reslrg':reslrg}, ['ulhs','plhs','lrglhs'], const
            else:
                return {'resu':resu,'resp':resp}, ['ulhs','plhs'], const
        else:
            if rigidb_cons:
                return {'resu':resu,'reslrg':reslrg}, ['ulhs','lrglhs'], const
            else:
                return {'resu':resu}, ['ulhs'], const
            
            
    def fung_nonlinear_elastic(self,const_input,rigidb_cons=False,quasi_incompr=False):
        # ns    = function.Namespace()
        # ns.x  = self.geom
        treelog.info("Constructing residual")
  
        
        degree    = 2
        gaussdeg  = max(3,degree)*2
        # Create constants dict
        constants = {'a0':0.5*1e3 ,
                     'a1':3.0     ,
                     'a2':6.0     ,
                     'a3':0.01*1e3,
                     'a4':60      ,
                     'a5':55.*1e3 }
        constants.update(const_input)  # Update if necessary    
        const     = {'fung': constants}
        
        self.ns.a0 = constants['a0']
        self.ns.a1 = constants['a1']
        self.ns.a2 = constants['a2']
        self.ns.a3 = constants['a3']
        self.ns.a4 = constants['a4']
        self.ns.a5 = constants['a5']
        
        # Assign values to namespace
        if quasi_incompr:
          self.ns.ubasis     = self.topo.basis('spline', degree=degree).vector(self.topo.ndims)
        else:
          self.ns.ubasis     = self.topo.basis('spline', degree=degree, continuity=-2).vector(self.topo.ndims)
          
        if rigidb_cons:
            self.fix_rigid_body() # Create a rigib body lagrange multiplier constrain

            
        # if incomp:
        #     ns.pbasis = topo.basis('spline', degree=degree-1) #continuity=degree-2
        #     ns.p = 'pbasis_n ?plhs_n'
        
        # Current time-step
        self.ns.u_i   = 'ubasis_ni ?ulhs_n'
        self.ns.X_i   = 'x_i + u_i'      # Deformed configuration
        self.ns.F_ij  = 'δ_ij + u_i,j'   #'X_j,i'          # Deformation gradient tensor
        self.ns.FF_ij = 'F_ki F_kj'
        self.ns.E_ij  = '0.5 ( FF_ij - δ_ij )'
        self.ns.EI_ij = '2 E_ij + δ_ij'
        self.ns.Ef    = 'ef_i E_ij ef_j' 

        self.ns.Econt_ij = 'E_ik E_kj' # Or use 'E_ik E_jk', takes the symmetric part!?
        # Prev. time-step        
        self.ns.u0_i   = 'ubasis_ni ?ulhs0_n'
        self.ns.X0_i   = 'x_i + u0_i'      # Deformed configuration
        self.ns.F0_ij  = 'δ_ij + u0_i,j'   #'X_j,i'          # Deformation gradient tensor
        # self.ns.FF0_ij = 'F0_ki F0_kj'
        # self.ns.E0_ij  = '0.5 ( FF0_ij - δ_ij )'
        # self.ns.EI0_ij = '2 E0_ij + δ_ij'
        # self.ns.Ef0    = 'ef_i E0_ij ef_j'        
        
        
        self.ns.detF  = numpy.linalg.det( self.ns.F  )
        self.ns.detF0 = numpy.linalg.det( self.ns.F0 )
        
        self.ns.detFF = numpy.linalg.det(  self.ns.FF )
        self.ns.detEI = numpy.linalg.det(  self.ns.EI ) 
        self.ns.Finv  = numpy.linalg.inv( self.ns.F )
        self.ns.Finv0 = numpy.linalg.inv( self.ns.F0 )
        self.ns.EIinv = numpy.linalg.inv( self.ns.EI )
        self.ns.unit  = function.diagonalize( numpy.array([1.]*self.topo.ndims) ) # Create unit tensor
        #self.ns.λeig  = function.eig( ns.F )
        
        # Invariants
        self.ns.I1    = numpy.trace( self.ns.E )                           # First invariant
        self.ns.I2    = 0.5*( self.ns.I1*self.ns.I1 - numpy.trace(  self.ns.Econt )  ) # Second invariant
        self.ns.I3    = self.ns.detF  

            
        self.ns.dWidE_ij = 'a0 exp( a1 ( I1^2 ) - a2 I2 ) ( 2 a1 I1 δ_ij - a2 I1 δ_ij + a2 E_ji )'
        #self.ns.dWfdE_ij = 'a3 exp( a4 ( Ef^2 ) ) ( 2 a4 Ef ) unit_ij'
        self.ns.dWfdE_ij = '2 a3 a4 Ef exp( a4 ( Ef^2 ) ) ef_i ef_j '
        if quasi_incompr:
            self.ns.dWvdE_ij  = '4 a5 ( detFF - 1 ) ( detEI EIinv_ij  )'
            self.ns.dWdE      = self.ns.dWidE + self.ns.dWfdE + self.ns.dWvdE
            self.ns.stress_ij = '( 1 / detF ) F_ik dWdE_kl F_jl'
            self.ns.detint    = self.ns.detF # determinant used in the integral
            self.ns.detint0   = self.ns.detF0
        else:
            treelog.info('Incompressible-----------------------------------------')
            self.ns.pbasis    = self.topo.basis('spline', degree=degree-1, continuity=-2) # 
            self.ns.p         = 'pbasis_n ?plhs_n' # Hydrostatic pressure
            self.ns.dWdE      = self.ns.dWidE + self.ns.dWfdE
            self.ns.detint    = self.ns.detF # determinant used in the integral, fixed at 1
            self.ns.detint0   = self.ns.detF0
            self.ns.bulk      = 5e6
            #self.ns.stress_ij = '- ( p / detint ) δ_ij + ( 1 / detint ) F_ik dWdE_kl F_jl'    # Cauchy stress tensor
            self.ns.stress_ij = '- p δ_ij + ( 1 / detint ) F_ik dWdE_kl F_jl'
            resp  = self.topo.integral('pbasis_n ( detF - 1 ) detint d:x' @ self.ns, degree=gaussdeg)
            resp -= self.topo.integral('pbasis_n ( p / bulk ) detint d:x' @ self.ns, degree=gaussdeg)
            #resp = self.topo.integral('pbasis_n ( 1 - ( 1 / detF ) ) d:x' @ self.ns, degree=gaussdeg)
            #resp = self.topo.integral('pbasis_n ( log( detF ) ) detint d:x' @ self.ns, degree=gaussdeg)
            


         
        ## Weak formulation (residual construction)    
        # sqr  = topo.boundary['left'].integral('u_0 u_0 d:x' @ ns, degree=4)
        # sqr  += topo.boundary['bottom'].integral('u_1 u_1 d:x' @ ns, degree=4)
        # sqr  += topo.boundary['back'].integral('u_2 u_2 d:x' @ ns, degree=4)
        # consu = solver.optimize('ulhs', sqr, droptol=1e-15)
         
        # init, zero displacement
        # sqr_init  = topo.integral('u_i u_i d:x' @ ns, degree=4)
        # lhs_init  = solver.optimize('ulhs', sqr_init, droptol=1e-15)
         
        # resu  = topo.integral('( ubasis_nj,i ( Finv_ik stress_kj ) detint ) d:x' @ ns, degree=4)
        # resu -= topo.boundary['right'].integral('( ubasis_nj ( ?pload n_i ) Finv_ij detint ) d:x' @ ns, degree=4)
        
        
        # if incomp:
        # targets = ('ulhs','plhs')
        # resp = topo.integral('pbasis_n ( detF - 1 ) d:x' @ ns, degree=4)
        # res  = dict( resu=resu, resp=resp )
        # cons = dict( ulhs=consu)
        # else:
        # targets = ('ulhs') 
        # res  = dict( resu=resu  )
        # cons = dict( ulhs=consu )                                                                       
        
        # Create residual
        resu  = self.topo.integral('( ubasis_nj,i ( Finv_ik stress_kj ) detint ) d:x' @ self.ns, degree=gaussdeg) 
        
        self.ns.Kpericar = 5e3#0.2e5 # Pericardium stiffness [Pa/m]  
        resu  -= self.topo.boundary['epi'].integral('-Kpericar ubasis_ni u_i detint d:x' @ self.ns, degree=gaussdeg)    
        
        ## Check if traction boundary is specified
        for tracb in list(self.boundaries):
            if self.boundaries[tracb][0] == 'traction':
                #integr = '( ' + self.boundaries[tracb][1] + ' )' + ' ubasis_ni d:x'
                integr = '( ubasis_nj ( {} ) Finv_ij detint ) d:x'.format(self.boundaries[tracb][1])
                print(integr)
                resu  -= self.topo.boundary[tracb].integral(integr @ self.ns, degree=gaussdeg)
                self.boundaries.pop(tracb)    
                
        if rigidb_cons:
            reslrg  = self.rbmbound.boundary['left'].integral('u_i Xrbm_i lrgbasis_n' @ self.ns, degree=1)
            resu   += self.rbmbound.boundary['left'].integral('ubasis_ni Xrbm_i lrg' @ self.ns, degree=1)
            
        # if incompr:   
        #     return [resu,resp], ['ulhs','plhs'], const
        # else:
        #     return [resu], ['ulhs'], const
        if not quasi_incompr:   
            if rigidb_cons:
                return {'resu':resu,'resp':resp,'reslrg':reslrg}, ['ulhs','plhs','lrglhs'], const
            else:
                return {'resu':resu,'resp':resp}, ['ulhs','plhs'], const
        else:
            if rigidb_cons:
                return {'resu':resu,'reslrg':reslrg}, ['ulhs','lrglhs'], const
            else:
                return {'resu':resu}, ['ulhs'], const
   
   
    def BovendeerdMaterial(self, const_input, quasi_incompr=False, btype='spline', bdegree=3, quad_degree=None, surpressJump=False, interfaceClass=None):

        treelog.info("Constructing residual")
  
        #TODO check default gaussdegree value (causes instability of to low)
        #gaussdeg  = int( numpy.round( (bdegree + 1)/2 ) ) #max(3,bdegree)*2
        gaussdeg = quad_degree if quad_degree else int( numpy.round( (bdegree + 1)/2 ) ) # If not specified, use minimum number
        
        # Create constants dict
        constants = {'a0':0.4*1e3 ,
                     'a1':3.0     ,
                     'a2':6.0     ,
                     'a3':3.0     , #3.0
                     'a4':0.      ,
                     'a5':55.*1e3 }
        constants.update(const_input)  # Update if necessary    
        const     = {'Passive-Bovendeerd': constants}
        
        self.ns.a0 = constants['a0']
        self.ns.a1 = constants['a1']
        self.ns.a2 = constants['a2']
        self.ns.a3 = constants['a3']
        self.ns.a4 = constants['a4']
        self.ns.a5 = constants['a5']

 
        # Continuum measures
        self.ns.Econt_ij       = 'E_ik E_kj' # Green-Lagrange strain contraction
        self.ns.detC           = numpy.linalg.det(self.ns.C)
        self.ns.Cinv           = numpy.linalg.inv(self.ns.C)  
        self.ns.Identity       = function.diagonalize( numpy.array([1.]*self.topo.ndims) ) # 2nd order identity tensor
        self.ns.Identity4_ijkl = 'Identity_ij Identity_kl'                                 # 4th order identity tensor
        
        ## Invariants & derivatives
        self.ns.I1    = numpy.trace( self.ns.E )                                       # 1st invariant
        self.ns.I2    = 0.5*( self.ns.I1*self.ns.I1 - numpy.trace(  self.ns.Econt )  ) # 2nd invariant
        self.ns.I3    = self.ns.detF         # 3rd invariant
        self.ns.I4    = self.ns.Ef           # 4th quasi-invariant
        self.ns.I5    = 'ef0_i Econt_ij ef0_j' # 5th quasi-invariant
            
        self.ns.dI1dE    = self.ns.Identity
        self.ns.dI2dE_ij = 'I1 Identity_ij - E_ji'
        self.ns.dI4dE_ij = 'ef0_i ef0_j'   
        self.ns.dI5dE_pk = 'ef0_i ( E_ij Identity4_jpkl + Identity4_iptk E_tl ) ef0_l'    
            
            
        # material expression
        self.ns.Q        = '( a1 I1^2 ) - ( a2 I2 ) + ( a3 - a4 ) ( I4^2 ) + a4 I5'     
        self.ns.dWidE_ij = 'a0 ( 2 a1 I1 dI1dE_ij - a2 dI2dE_ij + 2 ( a3 - a4 ) I4 dI4dE_ij + a4 dI5dE_ij ) exp( Q )'

        if quasi_incompr:
            self.ns.dWvdE_ij  = '4 a5 ( detC - 1 ) ( detC Cinv_ij )'
            self.ns.dWdE      = self.ns.dWidE + self.ns.dWvdE
            self.ns.stress_ij = '( 1 / detF ) F_ik dWdE_kl F_jl' # Cauchy-stress tensor
        else:
            treelog.info('Incompressible-----------------------------------------')
            self.ns.dWdE      = self.ns.dWidE 
            
            self.ns.stress_ij = 'p ( ( detC - 1 ) + 2 detF detF ) δ_ij + ( 1 / detF ) F_ik dWdE_kl F_jl' 
            resp  = self.topo.integral('4 pbasis_n detF ( detC - 1 ) d:x' @ self.ns, degree=gaussdeg)
            resp -= self.topo.integral('pbasis_n ( p / a5 ) d:x' @ self.ns, degree=gaussdeg)
            
            #self.ns.stress_ij = '- ( p / detint ) δ_ij + ( 1 / detint ) F_ik dWdE_kl F_jl'    # Cauchy stress tensor
            #self.ns.stress_ij = 'p δ_ij + ( 1 / detint ) F_ik dWdE_kl F_jl'    # Cauchy stress tensor 
            #resp = self.topo.integral('pbasis_n ( detF - 1 ) detint d:x' @ self.ns, degree=gaussdeg)
            #resp = self.topo.integral('pbasis_n ( 1 - ( 1 / detF ) ) d:x' @ self.ns, degree=gaussdeg)
            #resp = self.topo.integral('pbasis_n ( log( detF ) ) detint d:x' @ self.ns, degree=gaussdeg)
                                           
                                           
                                           
                                           
                                                                                
        ## Construct displacement residual--------------------------------------------------------------------------#
        
        ## A) Create constitutive residual
        resu  = self.topo.integral('( ubasis_nj,i ( Finvint_ik stress_kj ) detFint ) d:x' @ self.ns, degree=gaussdeg) 

        
        ## B) Check if we need to surpress the interface jumps
        if surpressJump:
           βjump = constants['βjump'] if 'βjump' in constants.keys() else 100
           resu += self.interface_jump_res(interfaceClass, nonlinear=True, βjump=βjump, gaussdeg=gaussdeg) 
           
        ## C) Check for (weakly imposed) boundary conditions
        boundcond, res_bc = self.boundary_conditions(nonlinear=True, gaussdeg=gaussdeg)
        if boundcond:
           resu += res_bc
        
        ## D) Check for (weakly imposed) rigid body constraint
        ldefined, residual = self.surpress_rigid_body(nonlinear=True, gaussdeg=gaussdeg)
        if ldefined:
           resu  += residual["resu"]
           reslx  = residual["reslx"]
           resly  = residual["resly"]
           reslxy = residual["reslxy"] 
        ##----------------------------------------------------------------------------------------------------------#
        
        
        
        ## Check for additional boundary conditions
#        ldefined = False
#        for boundtype, info in self.boundaries.items():
#            boundary, value = info # get boundary type and corresponding value
#            
#            if boundtype == 'traction':
#                #integr = '( ' + self.boundaries[tracb][1] + ' )' + ' ubasis_ni d:x'
#                integr = '( ubasis_nj ( {} ) Finvint_ij detFint ) d:x'.format(value)
#                print(integr)
#                resu  -= self.topo.boundary[boundary].integral(integr @ self.ns, degree=gaussdeg)   
#                
#                
#            elif boundtype == 'spring (alldirect)':
#                self.ns.Kpericar = value
#                resu  -= self.topo.boundary[boundary].integral('-Kpericar ubasis_ni u_i detFint d:x' @ self.ns, degree=gaussdeg)
#                
#                
#            elif boundtype == 'rigid body' and value == 'nullspace': # Constrain x- and y-directions by default!
#                 if not ldefined:
#                     ldefined = True
#                     self.ns.lxbasis  = [1]
#                     self.ns.lybasis  = [1]
#                     self.ns.lxybasis = [1]
#                     self.ns.lx  = 'lxbasis_n ?lhslx_n'
#                     self.ns.ly  = 'lybasis_n ?lhsly_n'  
#                     self.ns.lxy = 'lxybasis_n ?lhslxy_n' 
#                     reslx  = self.topo.boundary[boundary].integral('( lxbasis_n u_0 ) detFint d:x' @ self.ns, degree=gaussdeg) # Total x-displacement
#                     resly  = self.topo.boundary[boundary].integral('( lybasis_n u_1 ) detFint d:x' @ self.ns, degree=gaussdeg) # Total y-displacement    
#                     reslxy = self.topo.boundary[boundary].integral('lxybasis_n ( Finvint_j0 u_1,j - Finvint_i1 u_0,i ) detFint d:x' @ self.ns, degree=gaussdeg) # Total rotation
#                     #reslxy = self.topo.boundary[boundary].integral('lxybasis_n ( u_0,1 - u_1,0 ) detFint d:x' @ self.ns, degree=gaussdeg) # Total rotation
#                     resu += self.topo.boundary[boundary].integral('( lx ubasis_n0 + ly ubasis_n1 ) detFint d:x' @ self.ns, degree=gaussdeg) # Translation
#                     #resu += self.topo.boundary[boundary].integral('lxy ( ubasis_n0,1 - ubasis_n1,0 ) detFint d:x' @ self.ns, degree=gaussdeg) # Rotation
#                     resu += self.topo.boundary[boundary].integral('lxy ( Finvint_j0 ubasis_n1,j - Finvint_i1 ubasis_n0,i ) detFint d:x' @ self.ns, degree=gaussdeg) # Rotation
#                 else:
#                     assert self.g_type == 'bv', "{}-boundary has double rigid body motion boundary conditions".format(boundary)
#                     resu += self.topo.boundary[boundary].integral('( lx ubasis_n0 + ly ubasis_n1 ) detFint d:x' @ self.ns, degree=gaussdeg) # translation
#                     #resu += self.topo.boundary[boundary].integral('( lxy ( ubasis_n0 u_1 + u_0 ubasis_n1 ) detint d:x' @ self.ns, degree=gaussdeg) # Rotation
             




                      
        if not quasi_incompr:   
            if ldefined:
                return {'resu':resu, 'resp':resp, 'reslx':reslx, 'resly':resly, 'reslxy':reslxy}, ['ulhs','plhs','lhslx','lhsly','lhslxy'], const
            else:
                return {'resu':resu,'resp':resp}, ['ulhs','plhs'], const
        else:
            if ldefined:
                return {'resu':resu, 'reslx':reslx, 'resly':resly, 'reslxy':reslxy}, ['ulhs','lhslx','lhsly','lhslxy'], const
            else:
                return {'resu':resu}, ['ulhs'], const
                
                            
