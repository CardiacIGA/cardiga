# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 14:33:12 2021

@author: s146407
"""
import treelog, numpy
from nutils import solver, function


class Active_models():
        def __init__(self,ns,topo): # Model type: string
            self.ns   = ns
            self.topo = topo
            return    
        
        def construct(self,m_type,btype,bdegree,quad_degree,const_input={}):
            if m_type == 'kerckhoffs':
                with treelog.context('Active model (Kerckhoffs)'):
                    res, targ, const = self.active_kerckhoffs(const_input, btype=btype, bdegree=bdegree, quad_degree=quad_degree) 
                    self.initlc = const['active']['lsc0']       
            elif m_type == 'wall':
              with treelog.context('Active model (Wall)'):
                res, targ, const = self.active_wall(const_input)
            else:
                raise "Unknown model '{}' type".format(m_type) 
            return res, targ, const
        
        def get_initial(self,):
            # with treelog.context('Active model (initial)'):
            #     sqr  = self.topo.integral('( lc0 - lcc0 )^2 d:x' @ self.ns, degree=4)                
            #     init = solver.optimize('lhslc0', sqr, tol=1e-6, droptol=1e-15)
            init = numpy.ones(len(self.ns.lcbasis))*self.initlc
            return {'lhslc':init, 'lhslc0':init}
        
        def active_kerckhoffs(self, const_input, btype='spline', bdegree=2, quad_degree=None, arge_def=None): # In case large deformations are present enable this
            # Create constants dict
            #degree   = 2
            #gaussdeg = max(3,bdegree)*2
            gaussdeg = quad_degree if quad_degree else int( numpy.round( (bdegree + 1)/2 ) ) # If not specified, use minimum number
            #dt   = 2e-3   # [s]
#            tcyc = 10e-3 # [s] 
#            tact = 0.#1e-3 # [s]
            ## peter constants
#            constants = {'ta'    : '?ta',  # [s]
#                         'δt'    : '?dt',  # [s]
#                         'tcyc'  : tcyc ,  # [s]
#                         'tact'  : tact ,  # [s]
#                         'alphat': 0.5,   # [-], Crank-Nicholson
#                         'Ea'    : 20.,   # [1/μm]
#                         'T0'    : 160e3, # [Pa]
#                         'v0'    : 7.5,   # [μm/s]
#                         'lcc0'  : 1.5,   # [μm]
#                         'lsc0'  : 1.9,   # [μm]
#                         'tr'    : 0.075, # [s]
#                         'td'    : 0.150, # [s]
#                         'b'     : 0.16,  # [s/μm]
#                         'ld'    :-0.5,   # [μm]
#                         'a6'    : 2.0,   # [1/μm]
#                         'a7'    : 1.5,   # [μm]
#                         }
            constants = {'ta'    : '?ta',  # [s]
                         'δt'    : '?dt',  # [s]
                         'alphat': 0.,   # [-], Crank-Nicholson
                         'Ea'    : 20.,   # [1/μm]
                         'T0'    : 140e3, # [Pa]
                         'v0'    : 7.5,   # [μm/s]
                         'lcc0'  : 1.5,   # [μm]
                         'lsc0'  : 1.9,   # [μm]
                         'tr'    : 0.075, # [s]
                         'td'    : 0.150, # [s]
                         'b'     : 0.16,  # [s/μm]
                         'ld'    :-1.0,   # [μm]
                         'a6'    : 2.0,   # [1/μm]
                         'a7'    : 1.5,   # [μm]
                         }
                         
            constants.update(const_input)  # Update if necessary    
            const     = {'active': constants}
            
            self.ns.δt    = constants['δt']
            #self.ns.tcyc  = constants['tcyc']
            #self.ns.tact  = constants['tact']
            self.ns.ta    = constants['ta']
            self.ns.alpha = constants['alphat'] # Crank-Nicholson
            ## Active stress component-----------------------------------------------------
            
            ## Constants, according to https://doi.org/10.1114/1.1566447
            self.ns.Ea   = constants['Ea']   
            self.ns.T0   = constants['T0'] 
            self.ns.v0   = constants['v0']   
            self.ns.lcc0 = constants['lcc0'] 
            self.ns.lsc0 = constants['lsc0'] 
            self.ns.tr   = constants['tr']
            self.ns.td   = constants['td'] 
            self.ns.b    = constants['b']  
            self.ns.ld   = constants['ld']  
            self.ns.a6   = constants['a6']  
            self.ns.a7   = constants['a7']
            
            
            self.ns.lcbasis = self.topo.basis(btype, degree=bdegree)#, patchcontinuous=False) #self.topo.basis_discont(degree=1) #self.topo.basis('spline', degree=1)#, continuity=-3)
            
            
            ## Contractile element length
            self.ns.Ff_i     = 'F_ij ef0_j' # Vector (Deformation in fiber direction)
            self.ns.Ffnorm_i = 'Ff_i / sqrt( Ff_j Ff_j )' # Normalized deformed fiber vector
            self.ns.ls    = 'lsc0 ( Ff_i Ff_i )^0.5'
            self.ns.Ff0_i = 'F0_ij ef0_j' # Vector (Deformation in fiber direction)
            self.ns.ls0   = 'lsc0 ( Ff0_i Ff0_i )^0.5'
            
#            self.ns.lc    = 'lcbasis_n ?lhslc_n'
#            self.ns.lc0   = 'lcbasis_n ?lhslc0_n' 
#            self.ns.δlc   = '( lc - lc0 ) / δt'

            self.ns.lctot    = 'lcbasis_n ?lhslc_n'
            self.ns.lc0tot   = 'lcbasis_n ?lhslc0_n'
            self.ns.lc    = self.ns.lctot #( self.ns.ls  - self.ns.lctot  )*function.less( self.ns.ls   , self.ns.lctot  )  + self.ns.lctot 
            self.ns.lc0   = self.ns.lc0tot #( self.ns.ls0 - self.ns.lc0tot )*function.less( self.ns.ls0  , self.ns.lc0tot )  + self.ns.lc0tot
            
#            self.ns.lc    = self.ns.lctot *function.less( self.ns.lctot  , self.ns.ls )   + self.ns.ls *function.greater( self.ns.lctot  , self.ns.ls  ) + self.ns.lctot*function.equal( self.ns.lctot , self.ns.ls )# Restrict lc such that lc < ls
#            self.ns.lc0   = self.ns.lc0tot*function.less( self.ns.lc0tot , self.ns.ls0 )  + self.ns.ls0*function.greater( self.ns.lc0tot , self.ns.ls0 ) + self.ns.lc0tot*function.equal( self.ns.lc0tot , self.ns.ls0 )# idem
#            self.ns.lc    = self.ns.lctot * ( function.less( self.ns.lctot  , self.ns.ls ) + function.equal( self.ns.lctot , self.ns.ls ) - function.less( self.ns.lctot , self.ns.ls )*function.equal( self.ns.lctot , self.ns.ls ) )  + self.ns.ls *function.greater( self.ns.lctot  , self.ns.ls  ) # Restrict lc such that lc < ls
#            self.ns.lc0   = self.ns.lc0tot* ( function.less( self.ns.lc0tot , self.ns.ls0 ) + function.equal( self.ns.lc0tot , self.ns.ls0 ) - function.equal( self.ns.lc0tot , self.ns.ls0 )*function.less( self.ns.lc0tot , self.ns.ls0 ) ) + self.ns.ls0*function.greater( self.ns.lc0tot , self.ns.ls0 ) # idem
            
            self.ns.δlc   = '( lc - lc0 ) / δt'
            
            
            self.ns.dv    = '( Ea ( ls  -  lc ) - 1 ) v0'  # Backward Euler
            self.ns.dv0   = '( Ea ( ls0 - lc0 ) - 1 ) v0' # Forward Euler
            
            ## Iso contraction elements
            self.ns.fisof = 'T0 tanh( a6 ( lc - a7 ) )^2'
            
            self.ns.fiso  = self.ns.fisof*numpy.greater( self.ns.lc , self.ns.a7)
            
            # Assign asynchronous activation if specified
            if hasattr(self.ns, "tasync"):
              self.ns.tact = self.ns.ta - self.ns.tasync 
            else:
              self.ns.tact = self.ns.ta
              
            ## Twitch contraction element
            self.ns.tmax     = 'b ( ls - ld )'
            self.ns.ftwitchf = '( tanh( tact / tr )^2 ) ( tanh( ( tmax - tact ) / td  )^2 )'
            self.ns.ftwitch  = self.ns.ftwitchf*numpy.greater( self.ns.tact*numpy.less( self.ns.tact , self.ns.tmax), 0.0)
            
            ## Active stress component
            #self.ns.σa         = '( ls / lsc0 ) fiso ftwitch Ea ( ls - lc )'
            #self.ns.σa         = '( lsc0 / ls ) fiso ftwitch Ea ( ls - lc )'
            #self.ns.σa         = function.max( self.ns.σatot, 0 ) # Force ls = lc, i.e. 0 active stress
            #self.ns.stressa_ij = 'σa ( 1 / detF ) F_ip ef_p ef_k F_jk'
            #self.ns.stressa_ij = 'σa F_ip ef_p ef_k F_jk'
            #self.ns.stressa_ij = 'σa 0.5 ( ef_i ef_j + ef_j ef_i )'
            
            self.ns.σa         = '( ls / lsc0 ) fiso ftwitch Ea ( ls - lc )'
            self.ns.stressa_ij = 'σa ( 1 / detF ) Ffnorm_i Ffnorm_j'
            #self.ns.stressa_ij = 'σa Ffnorm_i Ffnorm_j'
            
            ##-----------------------------------------------------------------------------
        
            # Active component
            #resu   = self.topo.integral('ubasis_ni,j stressa_ij d:x' @ self.ns, degree=8)
            #self.topo.integral('( ubasis_nj,i ( Finv_ik stress_kj ) detFint ) d:x' @ self.ns, degree=4*degree)  
            resu   = self.topo.integral('( ubasis_nj,i ( Finvint_ik stressa_kj ) detFint ) d:x' @ self.ns, degree=gaussdeg)
            reslc  = self.topo.integral('lcbasis_n δlc d:x' @ self.ns, degree=gaussdeg) 
            reslc -= self.topo.integral('alpha ( lcbasis_n dv ) d:x' @ self.ns, degree=gaussdeg)        # forward-Euler
            reslc -= self.topo.integral('(1 - alpha) ( lcbasis_n dv0 ) d:x' @ self.ns, degree=gaussdeg)  # backward-Euler
            
            
            targets   = ['lhslc'] 
            res       = dict(resu=resu, reslc=reslc)
            return res, targets, const
            
            
            
                        
        def active_wall(self, const_input): 
            gaussdeg = 6
            
#            self.ns.efiber_i     = 'F_ij ef_j'
#            self.ns.efibernorm_i = 'efiber_i / sqrt( efiber_i efiber_i )'
#            self.ns.stressa_ij   = '?σa ( 1 / detint ) efibernorm_i efibernorm_j'
            
            #self.ns.stressa_ij = '?σa ( 1 / detint ) F_ip ef_p ef_k F_jk'
            self.ns.stressa_ij = '?σa ( 1 / detF ) F_ip ef_p ef_k F_jk'
            
            resu   = self.topo.integral('( ubasis_nj,i ( Finv_ik stressa_kj ) detFint ) d:x' @ self.ns, degree=gaussdeg)
            res       = dict(resu=resu)
            return res, [], {} # res, targets, constants

# Function that loads an external file containing the asynchronous activation times at each Gauss point. Returns a nutils array
def async_activation(filename, gauss_sample, return_coords=False):
    
    # Load the file
    tact_gauss = numpy.loadtxt(filename, delimiter=',', skiprows=1)
    assert tact_gauss.shape[1] == 4, f"Provided activation time in file are not in correct format, expected 4 (xyz-coordinates, t_act-values) columns got {tact_gauss.shape[1]}" 
    Xgauss = tact_gauss[:,:-1]
    tact   = tact_gauss[:, -1]*1e-3 # Convert to [ms]
    
    # Convert to nutils function and return
    return (gauss_sample.asfunction(tact), Xgauss) if return_coords else gauss_sample.asfunction(tact)         