# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 14:33:19 2021

@author: s146407
"""
import treelog, numpy
from nutils import solver, function
from .prolate_functions import Ellips_Functs

class Fiber_models():
        def __init__(self, ns, topo, geom, angles_input={}): # Model type: string
            self.topo       = topo
            self.geom       = geom
            self.ns         = ns
            self.angles_input = angles_input
            #self.boundaries   = boundaries
            return

        def construct(self, m_type, g_type, fiber_dir, btype, bdegree, quad_degree): # g-type is geometry type, left-ventricle (lv) or bi-ventricle (bv)

            # assert <<<<------ Check!
            assert g_type in ['lv','bv'], "Unkown fiber geometry {} specified".format(g_type)
            if m_type == 'default':
                treelog.info('The default fiber orientation is chosen, this is not advised!')
                lhs, const = self.default(fiber_dir)
            elif m_type == 'rossi':
                with treelog.context('Fiber field (Rossi)'):
                    lhs, const = self.rossi_et_al(g_type, btype=btype, bdegree=bdegree, quad_degree=quad_degree)
            elif m_type == 'analytic':
                with treelog.context('Analytic field (Peter)'):
                    lhs, const = self.peter_et_al(g_type, btype=btype, bdegree=bdegree, quad_degree=quad_degree)
            else:
                raise 'Not implemented, unknown fiber model type {}'.format(m_type)
            return lhs, const # solution dictionary = argument input for solver + angle constants

        # def angles():
        #     return
        # def rotate():
        #     return


        def default(self,fiber_dir): # Default fiber direction to be used(?), mention that no fiber direciton is specified!
            constants = {'fiber angles (default)': {}}
            if fiber_dir != None:
               self.ns.ef_i = fiber_dir
            else:
              if self.topo.ndims == 3:
                  self.ns.ef_i = '< 1., 0., 0.>_i'
                  self.ns.es_i = '< 0., 1., 0.>_i'
                  self.ns.en_i = '< 0., 0., 1.>_i'
              elif self.topo.ndims == 2:
                  self.ns.ef_i = '< 1., 0.>_i'
                  self.ns.es_i = '< 0., 1.>_i'
              else:
                  self.ns.ef_i = '< 1.>_i'
            return {}, constants

        def rossi_et_al(self, g_type, btype='spline', bdegree=3, quad_degree=None): # Rossi et al method
            ## Load input angles
            self.load_angles(g_type)
            constants = {'fiber angles (Rossi et al.):':  self.angles}
            self.boundary_names()
            ## Assign to namespace
            self.ns.Xibasis = self.topo.basis(btype, degree=bdegree)#, patchcontinuous=False)
            self.ns.fbasis  = self.topo.basis(btype, degree=bdegree)#, patchcontinuous=False)

            ## Assign angles
            self.ns.αepil  = self.angles['α_epi_l']
            self.ns.αendol = self.angles['α_endo_l']
            self.ns.βepil  = self.angles['β_epi_l']
            self.ns.βendol = self.angles['β_endo_l']

            if g_type == 'bv':
                self.ns.βepir  = self.angles['β_epi_r']
                self.ns.βendor = self.angles['β_endo_r']
                self.ns.αepir  = self.angles['α_epi_r']
                self.ns.αendor = self.angles['α_endo_r']
                self.ns.ξ      = 'Xibasis_n ?lhsξ_n' # Xi field required for combining left- right angles

            self.ns.φ  = 'fbasis_n ?lhsφ_n' # Laplace field in transmural direction
            self.ns.φn = self.ns.φ #function.min( function.max(self.ns.φ,0.0) , 1.0 )
            self.ns.k_i = '< 0, 0, 1 >_i'    # Basal normal vector = z-vector


            ## Local basis directions: Transmural, Longitudinal, Circumferential
            self.ns.et_i   = 'φn_,i / sqrt( φn_,j φn_,j )' # function.normalized(ns.φ_,i)
            self.ns.el1_i  = 'k_i - ( k_j et_j) et_i'
            self.ns.el     = function.normalized(self.ns.el1)
            self.ns.ec     = function.normalized(numpy.cross(self.ns.el,self.ns.et)) # Circumferential vector
            #fs.elc    = function.normalized(numpy.cross(fs.et,fs.ec))
            self.ns.localbase_ij = '<ec_i, el_i, et_i>_j' # Local basis in Matrix format

            # Rotate local basis
            self.ns.αl = 'αendol ( 1 - φn ) + αepil φn' # 'αepil ( 1 - φ ) + αendol φ'
            self.ns.βl = 'βendol ( 1 - φn ) + βepil φn' # 'βepil ( 1 - φ ) + βendol φ'
            if g_type == 'bv':
                self.ns.αr = 'αendor ( 1 - φn ) + αepir φn'
                self.ns.βr = 'βendor ( 1 - φn ) + βepir φn'

                self.ns.α = '0.5 (  αl ( ξ + 1 ) - αr ( ξ - 1)  )' # Combine left- right angles based on Xi
                self.ns.β = '0.5 (  βl ( ξ + 1 ) - βr ( ξ - 1)  )'
            else:
                self.ns.α = 'αl' # Combine left- right angles based on Xi
                self.ns.β = 'βl'

            # Rotate local base
#            self.ns.Rotα = function.asarray([ [function.cos(self.ns.α), -function.sin(self.ns.α),  0.],
#                                              [function.sin(self.ns.α),  function.cos(self.ns.α),  0.],
#                                              [                     0.,                       0.,  1.]])
##            self.ns.Rotβ = function.asarray([ [  1.,                       0.,                       0.],
##                                              [  0.,  function.cos(self.ns.β),  function.sin(self.ns.β)],
##                                              [  0., -function.sin(self.ns.β),  function.cos(self.ns.β)]])
#            self.ns.Rotβ = function.asarray([ [  function.cos(self.ns.β),  0.,  function.sin(self.ns.β)],
#                                              [                       0.,  1.,                       0.],
#                                              [ -function.sin(self.ns.β),  0.,  function.cos(self.ns.β)]])

            #self.ns.fiberbase_ij = 'localbase_ik Rotα_kt Rotβ_tj'# Rotα_kj '
            #self.ns.fiberbase_ij = 'Rotα_ik localbase_kt Rotα_tj'

            # Assign local directions
#            self.ns.ef_i = 'fiberbase_i0 / sqrt( fiberbase_j0 fiberbase_j0 )'#'fiberbase_i0' # Fiber direction
#            self.ns.es_i = 'fiberbase_i1' #
#            self.ns.en_i = 'fiberbase_i2' # normal direction

            ## Rotate vectors accordingly, first
#            self.ns.Rotα = self.rotate_about_axis(self.ns.et, self.ns.α)
#            self.ns.es_i = 'Rotα_ij el_j'
#            self.ns.Rotβ = self.rotate_about_axis(self.ns.es, self.ns.β)
#            self.ns.ef_i = 'Rotβ_ik ( Rotα_kj ec_j )'
#            self.ns.en_i = 'Rotβ_ij et_j'
#            self.ns.fiberbase_ij = '< ef_i, es_i, en_i >_j'

            self.ns.Rotα = self.rotate_about_axis(self.ns.et, self.ns.α)
            self.ns.ef_i = '( Rotα_ij ec_j )'
            self.ns.Rotβ = self.rotate_about_axis(self.ns.ef, self.ns.β)
            self.ns.es_i = '( Rotβ_ij et_j )'
            self.ns.en_i = 'Rotβ_ik ( Rotα_kj el_j )'
            self.ns.fiberbase_ij = '< ef_i, es_i, en_i >_j'


            ## Solving procedure
            self.ns.h    = self.topo.integrate_elementwise('d:x' @self.ns, degree=4, asfunction=True)**(1/self.topo.ndims)
            self.ns.N1   = 30 #1e-8
            self.ns.N2   = 30

            ## φ-field ----------------------------------------------------------------------------
            with treelog.context('Solving φ-field'):
                gaussdeg = 8 if quad_degree == None else quad_degree
                # Constrain
                sqr_φ  = self.topo.boundary[self.bound_endo_left].integral('φ^2 d:x' @ self.ns, degree=gaussdeg)
                sqr_φ += self.topo.boundary[self.bound_epi].integral('( φ - 1 )^2 d:x' @ self.ns, degree=gaussdeg)

#                if g_type == 'bv':
#                  sqr_φ += self.topo.boundary[self.bound_right_sep].integral('( φ - 1 )^2 d:x' @ self.ns, degree=gaussdeg)
                  # Patch Interface boundaries left-right-ventricles
#                  sqr_φ += self.topo['patch2'].boundary['left'].integral('( φ - 1 )^2 d:x' @ self.ns, degree=4)
#                  sqr_φ += self.topo['patch4'].boundary['left'].integral('( φ - 1 )^2 d:x' @ self.ns, degree=4)
#                  sqr_φ += self.topo['patch7'].boundary['left'].integral('( φ - 1 )^2 d:x' @ self.ns, degree=4)
#                  sqr_φ += self.topo['patch2'].boundary['left'].integral('( φ - 1 )^2 d:x' @ self.ns, degree=4)
#                  sqr_φ += self.topo['patch6'].boundary['left'].integral('( φ - 1 )^2 d:x' @ self.ns, degree=4)
#                  sqr_φ += self.topo['patch9'].boundary['left'].integral('( φ - 1 )^2 d:x' @ self.ns, degree=4)

                cons_φ = solver.optimize('lhsφ', sqr_φ, droptol=1e-15)

                # Solve Laplace
                res_φ  = self.topo.integral('fbasis_n,i φ_,i d:x' @ self.ns, degree=gaussdeg)
                if g_type == 'bv':
                    # Nitsche term
                    res_φ -= self.topo.boundary[self.bound_endo_right_nosep].integral('( fbasis_n φ_,i n_i ) d:x' @ self.ns, degree=gaussdeg)
                    res_φ += self.topo.boundary[self.bound_endo_right_nosep].integral('( ( ( N1 / h ) fbasis_n + fbasis_n,i n_i ) ( φ - 0.0 ) ) d:x' @ self.ns, degree=gaussdeg)
                    res_φ -= self.topo.boundary[self.bound_right_sep].integral('( fbasis_n φ_,i n_i ) d:x' @ self.ns, degree=gaussdeg)
                    res_φ += self.topo.boundary[self.bound_right_sep].integral('( ( ( N2 / h ) fbasis_n + fbasis_n,i n_i ) ( φ - 1.0 ) ) d:x' @ self.ns, degree=gaussdeg)
                # Solve
                lhsφ   = solver.solve_linear('lhsφ', res_φ, constrain=cons_φ)
                #lhsφ   = solver.newton('lhsφ', res_φ, constrain=cons_φ).solve(tol=1e-8)

            ## ξ-field ----------------------------------------------------------------------------
            with treelog.context('Solving ξ-field'):
                if g_type == 'bv':
                    sqr_ξ  = self.topo.boundary[self.bound_endo_left].integral('( ξ - 1 )^2 d:x' @ self.ns, degree=gaussdeg)
                    sqr_ξ += self.topo.boundary[self.bound_endo_right].integral('( ξ + 1)^2 d:x' @ self.ns, degree=gaussdeg)
                    cons_ξ = solver.optimize('lhsξ', sqr_ξ, droptol=1e-15)

                    res_ξ  = self.topo.integral('Xibasis_n,i ξ_,i d:x' @ self.ns, degree=gaussdeg)
                    lhsξ   = solver.solve_linear('lhsξ', res_ξ, constrain=cons_ξ)
                    return dict(lhsφ=lhsφ , lhsξ=lhsξ), constants
                else:
                    return dict(lhsφ=lhsφ), constants
            #ns.φn  = function.min(ns.φn1,1.0)
            #return dict(lhsφ=lhsφ , lhsξ=lhsξ), constants


        def arccosh(self, x):
            #return function.ln( x + function.sqrt( function.power(x, 2) - 1  ) )
            return numpy.log( x + numpy.sqrt( numpy.power(x, 2) - 1  ) ) # natural logarithm
            
        def peter_et_al(self, g_type, btype='spline', bdegree=3, quad_degree=None):

            ## Geometrical input
            C  = 43e-3              # Focal length of the LV geometry
            H  = 24.0e-3
            ξi = 0.371296808757 #0.37129680875745236
            ξo = 0.678355651828 #0.6783556518284651

            self.ns.xcorrect = self.ns.x #+ function.asarray([0, 0, H]) # Elevate again, because (0,0,0) is at the base currently
            self.ns.sqradd = numpy.linalg.norm( self.ns.xcorrect + function.asarray([0, 0, C]), axis=-1 ) 
            self.ns.sqrmin = numpy.linalg.norm( self.ns.xcorrect - function.asarray([0, 0, C]), axis=-1 ) 

            # Define prolate spheroidal coordinates
            self.ns.ξcoord = self.arccosh( ( self.ns.sqradd + self.ns.sqrmin  ) / ( 2*C ) ) 
            self.ns.θcoord = numpy.arccos( numpy.maximum(-1 + 1e-18, ( self.ns.sqradd - self.ns.sqrmin  ) / ( 2*C ) ) ) # Make sure nan is not possible with 1e-18 
            self.ns.φcoord = numpy.arctan2( self.ns.xcorrect[1], self.ns.xcorrect[0] ) 

            # Define local basis
            self.ns.frac = 1 / numpy.sqrt( numpy.sinh(self.ns.ξcoord)**2 + numpy.sin(self.ns.θcoord)**2 )
            self.ns.et   = function.asarray([ numpy.cosh(self.ns.ξcoord)*numpy.sin(self.ns.θcoord)*numpy.cos(self.ns.φcoord),
                                              numpy.cosh(self.ns.ξcoord)*numpy.sin(self.ns.θcoord)*numpy.sin(self.ns.φcoord),
                                              numpy.sinh(self.ns.ξcoord)*numpy.cos(self.ns.θcoord)             ]) * self.ns.frac
            self.ns.el   = function.asarray([ numpy.sinh(self.ns.ξcoord)*numpy.cos(self.ns.θcoord)*numpy.cos(self.ns.φcoord),
                                              numpy.sinh(self.ns.ξcoord)*numpy.cos(self.ns.θcoord)*numpy.sin(self.ns.φcoord),
                                             -numpy.cosh(self.ns.ξcoord)*numpy.sin(self.ns.θcoord)             ]) * self.ns.frac
            self.ns.ec   = function.asarray([-numpy.sin(self.ns.φcoord), numpy.cos(self.ns.φcoord), 0.])

            # Alternative vector formulation
            #self.basis_vector_alternative()

            ## Sample the geometry on the gauss quadrature points
            gaussdegree = quad_degree if quad_degree else 4 #int( numpy.round( (bdegree + 1)/2 ) )

            gauss   = self.topo.sample('gauss', gaussdegree) #self.topo.sample('bezier', 10) #
            ξ, θ, z = gauss.eval([self.ns.ξcoord, self.ns.θcoord, self.ns.xcorrect[2]]) # Evaluate specific coordinates at the gauss points

            # Extractnormalized wall corodinates
            vbound = [-1., 1. ]
            ubound = [-1., 0.5]
            ξbound = [ ξi, ξo ]
            zbound = H

            u = numpy.zeros(len(ξ))
            v = u.copy()
            for i, (ξi, θi, zi) in enumerate(zip(ξ, θ, z)):
                u[i], v[i] = Ellips_Functs.prolate_to_normalizedwall( ξi, θi, zi,  vbound, ubound, zbound, ξbound )

            self.ns.unew = gauss.asfunction(u)
            self.ns.vnew = gauss.asfunction(v)



            self.ns.ucoordbasis = self.topo.basis(btype, degree=bdegree)#self.topo.basis_discont(degree=1)
            self.ns.U = 'ucoordbasis_n ?lhsUcoord_n'
            self.ns.V = 'ucoordbasis_n ?lhsVcoord_n'

            sqrU  = self.topo.integral('( unew - U )^2 d:x' @ self.ns, degree=gaussdegree)
            lhsU  = solver.optimize('lhsUcoord', sqrU, tol=1e-6, droptol=1e-15)
            sqrV  = self.topo.integral('( vnew - V )^2 d:x' @ self.ns, degree=gaussdegree)
            lhsV  = solver.optimize('lhsVcoord', sqrV, tol=1e-6, droptol=1e-15)

            #dict(lhsUcoord=lhsU, lhsUcoord=lhsV)


            # Get Helix and transmural angles, dependend on u and v
            self.ns.αh, self.ns.αt, constants = self.legendre_angles(self.ns.U,self.ns.V)

            #a = function.sqrt( 1 / ( 1 + function.power(function.cos( self.ns.αt ), 2) * function.power(function.tan( self.ns.αh ), 2)  ) )
            #b = function.sqrt( function.power(function.cos( self.ns.αt ), 2) * function.power(function.tan( self.ns.αh ), 2) / ( 1 + function.power(function.cos( self.ns.αt ), 2) * function.power(function.tan( self.ns.αh ), 2)  ) )
            a = ( 1. + numpy.cos( self.ns.αt )**2 * numpy.tan( self.ns.αh )**2 )**-0.5
            self.ns.ef = a*( numpy.cos( self.ns.αt )*self.ns.ec  + numpy.sin( self.ns.αt )*self.ns.et - numpy.cos( self.ns.αt )*numpy.tan( self.ns.αh )*self.ns.el)


            #self.ns.ef = a * function.cos( self.ns.αt ) * self.ns.ec + a * function.cos( self.ns.αt ) * self.ns.et - b*self.ns.el

            return dict(lhsUcoord=lhsU, lhsVcoord=lhsV), constants




        def rotate_about_axis(self,axis,angle): # Return rotation matrix around a specified axis given an angle
            K = function.asarray([ [      0. , -axis[2],  axis[1] ],
                                   [  axis[2],       0., -axis[0] ],
                                   [ -axis[1],  axis[0],       0. ] ]) # cross-product matrix
            I = function.asarray([ [  1., 0.,  0. ],
                                   [  0., 1.,  0. ],
                                   [  0., 0.,  1. ] ]) # Identity matrix

            #return I + function.sin(angle)*K + ( 1 - function.cos(angle) )*( function.dot(K,K,axes=(1,0)) ) # Rodrigues
            #return ( I + function.sin(angle)*K +  ( 2*( function.sin(0.5*angle) )**2 ) *( function.cross( axis, axis ) - I ) ) # Same, but different
            return ( I + numpy.sin(angle)*K +  ( 2*( numpy.sin(0.5*angle) )**2 ) *( numpy.cross( axis, axis ) - I ) ) # Same, but different
            
        def basis_vector_alternative(self,):

            C     = 43e-3
            sigma =  ( self.ns.sqradd + self.ns.sqrmin  ) / ( 2*C )
            tau   =  ( self.ns.sqradd - self.ns.sqrmin  ) / ( 2*C )
            # Circumferential
            self.ns.EC = self.ns.ec # Same


            # Longitudinal
            term = numpy.sqrt( ( sigma**2 - 1 )*( 1 - tau**2 ) )
            Bl    = ( tau * ( 1 - sigma**2 ) ) / term
            self.ns.ELv = function.asarray([ Bl*numpy.cos( self.ns.φcoord ), Bl*numpy.sin( self.ns.φcoord ), sigma ])
            self.ns.EL  = self.ns.ELv / numpy.linalg.norm( self.ns.ELv , axis=1 ) # function.norm2( self.ns.ELv )

            # Transmural (Radial)
            Bt          = ( sigma * ( 1 - tau**2 ) ) / term
            self.ns.ETv = function.asarray([ Bt*numpy.cos( self.ns.φcoord ), Bt*numpy.sin( self.ns.φcoord ), tau ])
            self.ns.ET  = self.ns.ETv / numpy.linalg.norm( self.ns.ETv, axis=1 ) #function.norm2( self.ns.ETv )


#            # Longitudinal
#            term = function.sqrt( ( self.ns.ξcoord**2 - 1 )*( 1 - self.ns.θcoord**2 ) )
#            Bl    = ( self.ns.θcoord * ( 1 - self.ns.ξcoord**2 ) ) / term
#            self.ns.ELv = function.asarray([ Bl*function.cos( self.ns.φcoord ), Bl*function.sin( self.ns.φcoord ), self.ns.ξcoord ])
#            self.ns.EL  = self.ns.ELv / function.norm2( self.ns.ELv )
#
#            # Transmural (Radial)
#            Bt          = ( self.ns.ξcoord * ( 1 - self.ns.θcoord**2 ) ) / term
#            self.ns.ETv = function.asarray([ Bt*function.cos( self.ns.φcoord ), Bt*function.sin( self.ns.φcoord ), self.ns.θcoord ])
#            self.ns.ET  = self.ns.ETv / function.norm2( self.ns.ETv )
            return

        def boundary_names(self,):# Change these names if they have to be changed
            self.bound_endo_right       = 'endo_r'
            self.bound_epi              = 'epi'
            self.bound_endo_left        = 'endo_l'
            self.bound_base_right       = 'base_r'
            self.bound_base_left        = 'base_l'
            self.bound_right_sep        = 'septum_r'
            self.bound_endo_right_nosep = 'endo_r_nosep'
            return

        def legendre_angles(self,u,v):
            h = {'u0':  None, 'u1':  None, 'u2': 0.0984, 'u3':   None, 'u4': -0.0701,  # u4 -0.0501
                 'v0': 0.362, 'v1': -1.16, 'v2': -0.124, 'v3':  0.129, 'v4': -0.0614 }
            t = {'u0':  None, 'u1':  0.626, 'u2':   None, 'u3':  0.211, 'u4': None, 'u5': 0.038,
                 'v0':  None, 'v1': -0.626, 'v2':  0.502, 'v3':  None,  'v4': None }

            ## Legendre polynomials
            L0 = 1
            L1 = lambda x : x
            L2 = lambda x : 0.5*( 3*numpy.power(x,2) - 1)
            L3 = lambda x : 0.5*( 5*numpy.power(x,3) - 3*x )
            L4 = lambda x : 1/8 * ( 35*numpy.power(x,4) - 30*numpy.power(x,2) + 3 )
            L5 = lambda x : 1/8 * ( 63*numpy.power(x,5) - 70*numpy.power(x,3) + 15*x )

            αh = ( h['v0']*L0 + h['v1']*L1(v) + h['v2']*L2(v) + h['v3']*L3(v) + h['v4']*L4(v) )*(       1.      + h['u2']*L2(u) + h['u4']*L4(u) )
            αt = (     1.     + t['v1']*L1(v) + t['v2']*L2(v) ) * ( 1 - numpy.power(v,2) ) * ( t['u1']*L1(u) + t['u3']*L3(u) + t['u5']*L5(u) )
            const = {'Angles by Peter:':  {**h, **t}}
            return αh, αt, const



        def load_angles(self,g_type):
            deg         = numpy.pi/180 # convert deg to radians
            self.angles = {'α_epi_l' :-60.*deg,
                           'α_endo_l':+60.*deg,
                           'β_epi_l' :+20.*deg,
                           'β_endo_l':-20.*deg}
            if g_type == 'bv': # if bi-ventricle, add additional angles
                self.angles['α_epi_r']  = -25.*deg
                self.angles['α_endo_r'] = +90.*deg
                self.angles['β_epi_r']  = +20.*deg
                self.angles['β_endo_r'] = +0.0*deg
            self.angles.update(self.angles_input)
            return