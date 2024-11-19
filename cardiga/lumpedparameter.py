# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 14:33:28 2021

@author: s146407
"""

import treelog, numpy
from nutils import function
        
class Lumped_models():
        
        def __init__(self,ns,topo): # Model type: string
            self.ns   = ns
            self.topo = topo
            ## Unit conversions (From unit specified -> Standard unit [m,m3,Pa,..])
            self.kPa  = 1e3          # kiloPascal to Pascal
            self.ml   = 1e-6         # milliliter to cubic meter
            self.mmHg = 133.3223684  # mmHg to Pascal
            self.ms   = 1e-3         # millisecond to second
            self.s    = 1.
            return           
        
        def construct(self, m_type, p_type, inner_boundary, quad_degree=None, const_input=None): #inner_boundary string representing inner boundary
            ## Assert input
            assert p_type in ['open','closed'], "Unknown circulation type '{}' type".format(p_type)
            assert m_type in ['0D','lv','bv'], "Unknown cardiac type '{}' type".format(m_type)         
            if len(inner_boundary)==2:
                assert m_type=='bv', "Only 1 inner boundary is specified for the bi-ventricle geometry: '{}' type".format(m_type)  
 
            ## Construct constant dictionary
            const = self.constants(m_type,p_type)
            if m_type == '0D': 
                const.update(self.constants0D()) 
            const.update(const_input)    
             
            ## Load/construct the base residual(open/closed) for (0D/lv) and add residual of (bv) if specified
            with treelog.context('{} circulation ({})'.format(m_type.capitalize(),p_type.capitalize())):
                    res, targ, targ0 = self.get_res_open_close(m_type, p_type, const, inner_boundary, gaussdeg=quad_degree)    ## construct reidual for open/closed lv or 0D model
                    # if m_type == 'bv':
                    #     raise "Not Implemented"
                    #     res_add = self.biventricle()   
                    #     res     = self.add2res(res,res_add) ## Add residuals
                        #targ += targ_bv....
                        
            # Initialize unknowns        
            init0 = self.initialize(m_type,p_type,const,inner_boundary)      
            
            return res, targ, targ0, const, init0
        
        def add2res(self,res_init,res_add): # Add a residual dict to an already existing residual dictionary 
            keys_add  = res_add.keys()
            keys_init = res_init.keys()
            for i in keys_add:
                if i in keys_init:
                    res_init[i] += res_add[i]
                else:
                    res_init[i]  = res_add[i]
            return res_init
        
        def get_res_open_close(self, m_type, p_type, const, inner_boundary, gaussdeg = None):
            gaussdeg = gaussdeg if gaussdeg else 4 #gaussdeg = 4
            
            degree   = 2
            if len(inner_boundary)==2:
                inner_lv_bound = inner_boundary[0]
                inner_rv_bound = inner_boundary[1]
            else:
                inner_lv_bound = inner_boundary
                
            if 'α' in const and 'δt' in const: # add time integration constant and time-step incase no active model is used (this is where it normaly initialized)
                self.ns.α  = const['α']  # Else should give an error in the model class
                self.ns.δt = const['δt'] # Else should give an error in the model class
            
            # Get truncation height (important for volume calculation)
            if 'H' in const:
                self.ns.Htrunc = const['H']
                self.ns.H_i    = '< 0, 0, Htrunc >_i' 
            else:
                self.ns.H_i = '< 0, 0, 0 >_i'
                
                
            self.ns.Ainner      = self.topo.boundary[inner_lv_bound].integrate('d:x' @ self.ns, degree=gaussdeg)# inner surface (reference state)
            
            self.ns.CartS     =  const['CartS']
            self.ns.RartS     =  const['RartS']
            self.ns.VartSi0   =  const['VartS0Con']
            
            #self.ns.RvenP     =  const['RvenP']
            
            ## Lagrange multipliers
            self.ns.partSbasis  = [1.]
            self.ns.plvbasis   = [1.] 

            # Ventricle unknowns
            self.ns.plv     = 'plvbasis_n ?plvlhs_n'                # Left-ventricle pressure
            self.ns.plv0    = 'plvbasis_n ?plv0lhs_n'                # Left-ventricle pressure
            
            # Arterial unknowns (Systemic)
            # self.ns.VartS  = 'Vartbasis_n ?VartSlhs_n'  # Systemic arterial volume
            # self.ns.VartS0 = 'Vartbasis_n ?VartS0lhs_n' # Systemic arterial volume
            # self.ns.δVartS = '( VartS - VartS0 )  / ( δt Ainner )' 
      
            # self.ns.partS  = '(VartS  - VartSi0 ) / CartS' 
            # self.ns.partS0 = '(VartS0 - VartSi0 ) / CartS' 
                        
            self.ns.partS  = 'partSbasis_n ?partSlhs_n'  # Systemic arterial volume
            self.ns.partS0 = 'partSbasis_n ?partS0lhs_n' # Systemic arterial volume
            
      
            self.ns.VartS  = 'CartS partS  + VartSi0' 
            self.ns.VartS0 = 'CartS partS0 + VartSi0' 
            self.ns.δVartS = '( VartS - VartS0 )  / ( δt Ainner )' 
            
            self.ns.qartSopen   = '( plv  - partS  ) / RartS'
            self.ns.qartS0open  = '( plv0 - partS0 ) / RartS'
            
            self.ns.qartSintLV   = numpy.maximum(0, self.ns.qartSopen  / self.ns.Ainner ) #function.max(0, self.ns.qartSopen  / self.ns.Ainner )
            self.ns.qartS0intLV  = numpy.maximum(0, self.ns.qartS0open / self.ns.Ainner) #function.max(0, self.ns.qartS0open / self.ns.Ainner)
            
            ## Define arterial load
            self.ns.qartS   = 'qartSintLV  Ainner'
            self.ns.qartS0  = 'qartS0intLV Ainner'

            
            # if m_type == '0D':
            # else:
            # self.ns.qartS   = 'qartSint  / Ainner'
            # self.ns.qartS0  = 'qartS0int / Ainner'
                
            # Unknowns Periphery (Systematic)
            if p_type == 'closed':
                self.ns.Vtot     =  const['Vtotal']
                self.ns.CvenP    =  const['CvenP']
                self.ns.RvenP    =  const['RvenP']
                self.ns.VvenPi0  =  const['VvenP0Con']
                self.ns.RperS    =  const['RperS']
                
            
                self.ns.pvenPa   = '( Vtot - VartS  - VvenPi0 ) / CvenP' # -int( Vlvdx/CvenP :dx) First part of the venous pressure (excluding Vlv)
                self.ns.pvenP0a  = '( Vtot - VartS0 - VvenPi0 ) / CvenP'
                
                if p_type != 'bv':
                    self.ns.qperSa   = '( partS  - pvenPa   ) / RperS'
                    self.ns.qperS0a  = '( partS0 - pvenP0a  ) / RperS'
                
                # Venous unknowns (Pulmonary)
                # Determine q based on the total volume of blood Vtot
                self.ns.qvenPa   = '( pvenPa  - plv  ) / RvenP'  # this part should not be integrated over the inner boundary              
                self.ns.qvenP0a  = '( pvenP0a - plv0 ) / RvenP'  # this part should not be integrated over the inner boundary              

            else:
                self.ns.qvenPa  = '( pvenP  - plv  ) / RvenP'
                self.ns.qvenP0a = '( pvenP0 - plv0 ) / RvenP'
            
            if m_type == '0D':
                self.leftventricle0D(const) # Add variables to namespace
                self.ns.qvenPopen  = '( qvenPa  + qvenPb  )' 
                self.ns.qvenP0open = '( qvenP0a + qvenP0b )' 
                self.ns.qvenP      = numpy.maximum(0, self.ns.qvenPopen  )
                self.ns.qvenP0     = numpy.maximum(0, self.ns.qvenP0open )
                
                # self.ns.qvenPint   = '( qvenP  / Ainner ) d:x'
                # self.ns.qvenP0int  = '( qvenP0 / Ainner ) d:x'
                self.ns.qvenPintLV   = '( qvenPopen  / Ainner ) d:x' # Only applicable to LV simulations
                self.ns.qvenP0intLV  = '( qvenP0open / Ainner ) d:x'
                
                self.ns.qperS  = '( qperSa  + qperSb  )'
                self.ns.qperS0 = '( qperS0a + qperS0b )'
                self.ns.qperSintLV  = '( qperS  / Ainner ) d:x'
                self.ns.qperS0intLV = '( qperS0 / Ainner ) d:x'
                

            elif m_type == 'lv':

                self.leftventricle()
                # self.ns.qvenPopen  = '( ( qvenPa  / Ainner ) + qvenPbdx  ) d:x' 
                # self.ns.qvenP0open = '( ( qvenP0a / Ainner ) + qvenP0bdx ) d:x' 
                # self.ns.qvenPint   = function.max(0, self.ns.qvenPopen  )
                # self.ns.qvenP0int  = function.max(0, self.ns.qvenP0open )
                self.ns.qvenPintLV  = '( ( qvenPa  / Ainner ) + qvenPbdx  ) d:x' 
                self.ns.qvenP0intLV = '( ( qvenP0a / Ainner ) + qvenP0bdx ) d:x' 
                
                self.ns.qvenP     = numpy.maximum(0, self.topo.boundary[inner_lv_bound].integral( self.ns.qvenPintLV  , degree=gaussdeg) )
                self.ns.qvenP0    = numpy.maximum(0, self.topo.boundary[inner_lv_bound].integral( self.ns.qvenP0intLV , degree=gaussdeg) )
                
                self.ns.qperSintLV  = '( ( qperSa  / Ainner ) + qperSbdx  ) d:x'
                self.ns.qperS0intLV = '( ( qperS0a / Ainner ) + qperS0bdx ) d:x'
                
                self.ns.qperS     = self.topo.boundary[inner_lv_bound].integral( self.ns.qperSintLV , degree=gaussdeg)
                self.ns.qperS0    = self.topo.boundary[inner_lv_bound].integral( self.ns.qperS0intLV, degree=gaussdeg)            
                
                self.ns.Vlv   = self.topo.boundary[inner_lv_bound].integral('Vlvdx d:x' @ self.ns, degree=gaussdeg)
                self.ns.VvenP = 'Vtot - VartS - Vlv'
                
                
                self.ns.pvenPb = self.topo.boundary[inner_lv_bound].integral('qvenPbdx RvenP d:x' @ self.ns, degree=gaussdeg) 
                self.ns.pvenP  = self.ns.pvenPa + self.ns.pvenPb 
                #self.ns.pvenP  = '( VvenP - VvenPi0 ) CvenP '
                
            elif m_type == 'bv':
                ## Load additional constants
                self.ns.CvenS    =  const['CvenS']
                self.ns.CartP    =  const['CartP']  
                self.ns.RperP    =  const['RperP']
                self.ns.RartP    =  const['RartP']
                self.ns.RvenS    =  const['RvenS']
                self.ns.VvenSi0  =  const['VvenS0Con']
                self.ns.VartPi0  =  const['VartP0Con']

                
                ## Define the new Lagrange multipliers
                self.ns.partPbasis  = [1.]
                self.ns.pvenSbasis  = [1.]
                self.ns.prvbasis    = [1.] 
    
                # Add the recirculation unknowns
                self.ns.prv     = 'prvbasis_n ?prvlhs_n'      # Right-ventricle pressure
                self.ns.prv0    = 'prvbasis_n ?prv0lhs_n'     # Right-ventricle pressure
                self.ns.partP   = 'partPbasis_n ?partPlhs_n'  # Pulmonary arterial volume
                self.ns.partP0  = 'partPbasis_n ?partP0lhs_n' # Pulmonary arterial volume
                self.ns.pvenS   = 'pvenSbasis_n ?pvenSlhs_n'  # Systemic venous volume
                self.ns.pvenS0  = 'pvenSbasis_n ?pvenS0lhs_n' # Systemic venous volume            
            
                self.ns.VartP  = 'CartP partP  + VartPi0' 
                self.ns.VartP0 = 'CartP partP0 + VartPi0' 
                self.ns.δVartP = '( VartP - VartP0 )  / ( δt Ainner )' 
                
                self.ns.VvenS  = 'CvenS pvenS  + VvenSi0' 
                self.ns.VvenS0 = 'CvenS pvenS0 + VvenSi0' 
                self.ns.δVvenS = '( VvenS - VvenS0 )  / ( δt Ainner )'  

                ## Load biventricle specifics
                self.biventricle()
                ## Venous part (pulmonary)
                self.ns.qvenPa2     = '- ( VartP  + VvenS   ) / ( RvenP CvenP )'
                self.ns.qvenPa20    = '- ( VartP0 + VvenS0  ) / ( RvenP CvenP )' 
                self.ns.qvenPintLV  = '( ( ( qvenPa  + qvenPa2  ) / Ainner ) + qvenPbdx  ) d:x' 
                self.ns.qvenP0intLV = '( ( ( qvenP0a + qvenPa20 ) / Ainner ) + qvenP0bdx ) d:x' 
                self.ns.qvenPintRV  = ' qvenPcdx  d:x' 
                self.ns.qvenP0intRV = ' qvenP0cdx d:x'
                
                self.ns.qvenP     = numpy.maximum(0, self.topo.boundary[inner_lv_bound].integral( self.ns.qvenPintLV  , degree=gaussdeg) +
                                                    self.topo.boundary[inner_rv_bound].integral( self.ns.qvenPintRV  , degree=gaussdeg))
                self.ns.qvenP0    = numpy.maximum(0, self.topo.boundary[inner_lv_bound].integral( self.ns.qvenP0intLV , degree=gaussdeg) +
                                                    self.topo.boundary[inner_rv_bound].integral( self.ns.qvenP0intRV  , degree=gaussdeg))

                ## Arterial part (pulmonary)            
                self.ns.qartPopen   = '( prv  - partP  ) / RartP'
                self.ns.qartP0open  = '( prv0 - partP0 ) / RartP'
                
                self.ns.qartPintLV   = numpy.maximum(0, self.ns.qartPopen  / self.ns.Ainner )
                self.ns.qartP0intLV  = numpy.maximum(0, self.ns.qartP0open / self.ns.Ainner)
                self.ns.qartP        = self.topo.boundary[inner_lv_bound].integral( 'qartPintLV d:x' @ self.ns, degree=gaussdeg)
                
                ## Peripherial part (pulmonary)
                self.ns.pvenPa2   = 'pvenPa  - ( ( VvenS   + VartP   ) / CvenP )'
                self.ns.pvenP0a2  = 'pvenP0a - ( ( VvenS0  + VartP0  ) / CvenP )'
                
                
                self.ns.qperPintLV   = '( 1 / RperP ) ( ( ( partP   - pvenPa2   ) / Ainner ) + ( Vlvdx  / CvenP ) ) d:x'
                self.ns.qperP0intLV  = '( 1 / RperP ) ( ( ( partP0  - pvenP0a2  ) / Ainner ) + ( Vlv0dx / CvenP ) ) d:x'               
                self.ns.qperPintRV   = '( Vrvdx   / ( CvenP RperP ) ) d:x'
                self.ns.qperP0intRV  = '( Vrv0dx  / ( CvenP RperP ) ) d:x'
                
                self.ns.qperP  = self.topo.boundary[inner_lv_bound].integral( self.ns.qperPintLV , degree=gaussdeg) + self.topo.boundary[inner_rv_bound].integral( self.ns.qperPintRV , degree=gaussdeg)
        
                ## Venous part (systemic)
                self.ns.qvenSopen   = '( pvenS  - prv  ) / RvenS'
                self.ns.qvenS0open  = '( pvenS0 - prv0 ) / RvenS'
                
                self.ns.qvenS  = numpy.maximum( 0, self.ns.qvenSopen )
                self.ns.qvenS0 = numpy.maximum( 0, self.ns.qvenS0open )
                
                self.ns.qvenSintLV  = '( qvenS  / Ainner ) d:x'
                self.ns.qvenS0intLV = '( qvenS0 / Ainner ) d:x'
                
                ## Arterial part (systemic)
                # - already done
                
                ## Peripherial part (systemic)
                self.ns.qperS   = '( partS  - pvenS  ) / RperS'
                self.ns.qperS0  = '( partS0 - pvenS0 ) / RperS'

                self.ns.qperSintLV   = '( qperS  / Ainner ) d:x'
                self.ns.qperS0intLV  = '( qperS0 / Ainner ) d:x'
                
                ## Define parameters for post-processing    
                self.ns.Vlv   = self.topo.boundary[inner_lv_bound].integral('Vlvdx d:x' @ self.ns, degree=gaussdeg)
                self.ns.Vrv   = self.topo.boundary[inner_rv_bound].integral('Vrvdx d:x' @ self.ns, degree=gaussdeg)
                self.ns.VvenP = 'Vtot - VartS - Vlv - Vrv - VartP - VvenS'
                
                self.ns.pvenPb = self.topo.boundary[inner_lv_bound].integral('qvenPbdx RvenP d:x' @ self.ns, degree=gaussdeg) 
                self.ns.pvenPc = self.topo.boundary[inner_rv_bound].integral('qvenPcdx RvenP d:x' @ self.ns, degree=gaussdeg) 
                self.ns.pvenP  = 'pvenPa - ( ( VvenS   + VartP   ) / CvenP ) + pvenPb + pvenPc'
                
                
            ## Residuals
            treelog.info("Constructing residual")
            resplv  = self.topo.boundary[inner_lv_bound].integral('          plvbasis_n δVlv          d:x' @ self.ns, degree=gaussdeg)
            resplv += self.topo.boundary[inner_lv_bound].integral('     α  ( plvbasis_n qartSintLV  ) d:x' @ self.ns, degree=gaussdeg)
            resplv += self.topo.boundary[inner_lv_bound].integral('(1 - α) ( plvbasis_n qartS0intLV ) d:x' @ self.ns, degree=gaussdeg)  
            #resplv -= self.topo.boundary[inner_boundary].integral('     α  ( plvbasis_n qvenPint  )    ' @ self.ns, degree=4)  # d:x is inside the qvenPint variable
            #resplv -= self.topo.boundary[inner_boundary].integral('(1 - α) ( plvbasis_n qvenP0int )    ' @ self.ns, degree=4)  # d:x is inside the qvenP0int variable
            

            resvartS  = self.topo.boundary[inner_lv_bound].integral('          partSbasis_n δVartS        d:x' @ self.ns, degree=gaussdeg)   # δVartS is divided by Ainner already
            resvartS -= self.topo.boundary[inner_lv_bound].integral('     α  ( partSbasis_n qartSintLV  ) d:x' @ self.ns, degree=gaussdeg)   # qartSint is divided by Ainner already
            resvartS -= self.topo.boundary[inner_lv_bound].integral('(1 - α) ( partSbasis_n qartS0intLV ) d:x' @ self.ns, degree=gaussdeg)   # qartS0int is divided by Ainner already
            resvartS += self.topo.boundary[inner_lv_bound].integral('     α  ( partSbasis_n qperSintLV  )    ' @ self.ns, degree=gaussdeg)   # d:x is inside the qperSint variable
            resvartS += self.topo.boundary[inner_lv_bound].integral('(1 - α) ( partSbasis_n qperS0intLV )    ' @ self.ns, degree=gaussdeg)   # d:x is inside the qperS0int variable
            
            
            resplvvenLV  = self.topo.boundary[inner_lv_bound].integral('     α  ( plvbasis_n qvenPintLV  )    ' @ self.ns, degree=gaussdeg)
            resplvvenLV0 = self.topo.boundary[inner_lv_bound].integral('(1 - α) ( plvbasis_n qvenP0intLV )    ' @ self.ns, degree=gaussdeg)
            
            if m_type == 'bv':
                resplvvenRV  = self.topo.boundary[inner_rv_bound].integral('     α  ( plvbasis_n qvenPintRV  )    ' @ self.ns, degree=gaussdeg)
                resplvvenRV0 = self.topo.boundary[inner_rv_bound].integral('(1 - α) ( plvbasis_n qvenP0intRV )    ' @ self.ns, degree=gaussdeg)

                resplv -= numpy.maximum( 0, resplvvenLV  + resplvvenRV )  # d:x is inside the qvenPint variable
                resplv -= numpy.maximum( 0, resplvvenLV0 + resplvvenRV0 ) # d:x is inside the qvenP0int variable 
                
                ## venous (systemic)
                resvenS  = self.topo.boundary[inner_lv_bound].integral('          pvenSbasis_n δVvenS        d:x' @ self.ns, degree=gaussdeg)   # δVartS is divided by Ainner already
                resvenS -= self.topo.boundary[inner_lv_bound].integral('     α  ( pvenSbasis_n qperSintLV  )    ' @ self.ns, degree=gaussdeg)   # qartSint is divided by Ainner already
                resvenS -= self.topo.boundary[inner_lv_bound].integral('(1 - α) ( pvenSbasis_n qperS0intLV )    ' @ self.ns, degree=gaussdeg)   # qartS0int is divided by Ainner already
                resvenS += self.topo.boundary[inner_lv_bound].integral('     α  ( pvenSbasis_n qvenSintLV  )    ' @ self.ns, degree=gaussdeg)   # d:x is inside the qperSint variable
                resvenS += self.topo.boundary[inner_lv_bound].integral('(1 - α) ( pvenSbasis_n qvenS0intLV )    ' @ self.ns, degree=gaussdeg)   # d:x is inside the qperS0int variable

                ## arterial (pulmonary)
                resartP  = self.topo.boundary[inner_lv_bound].integral('          partPbasis_n δVartP        d:x' @ self.ns, degree=gaussdeg)   # δVartS is divided by Ainner already
                resartP -= self.topo.boundary[inner_lv_bound].integral('     α  ( partPbasis_n qartPintLV  ) d:x' @ self.ns, degree=gaussdeg)   # qartSint is divided by Ainner already
                resartP -= self.topo.boundary[inner_lv_bound].integral('(1 - α) ( partPbasis_n qartP0intLV ) d:x' @ self.ns, degree=gaussdeg)   # qartS0int is divided by Ainner already
                resartP += self.topo.boundary[inner_lv_bound].integral('     α  ( partPbasis_n qperPintLV  )    ' @ self.ns, degree=gaussdeg)   # d:x is inside the qperSint variable
                resartP += self.topo.boundary[inner_lv_bound].integral('(1 - α) ( partPbasis_n qperP0intLV )    ' @ self.ns, degree=gaussdeg)   # d:x is inside the qperS0int variable
                resartP += self.topo.boundary[inner_rv_bound].integral('     α  ( partPbasis_n qperPintRV  )    ' @ self.ns, degree=gaussdeg)   # d:x is inside the qperSint variable
                resartP += self.topo.boundary[inner_rv_bound].integral('(1 - α) ( partPbasis_n qperP0intRV )    ' @ self.ns, degree=gaussdeg)   # d:x is inside the qperS0int variable
                
                ## Right ventricle pressure/volume
                resprv  = self.topo.boundary[inner_rv_bound].integral('          prvbasis_n δVrv          d:x' @ self.ns, degree=gaussdeg)
                resprv -= self.topo.boundary[inner_lv_bound].integral('     α  ( prvbasis_n qvenSintLV  )    ' @ self.ns, degree=gaussdeg)
                resprv -= self.topo.boundary[inner_lv_bound].integral('(1 - α) ( prvbasis_n qvenS0intLV )    ' @ self.ns, degree=gaussdeg)  
                resprv += self.topo.boundary[inner_lv_bound].integral('     α  ( prvbasis_n qartPintLV  ) d:x' @ self.ns, degree=gaussdeg)
                resprv += self.topo.boundary[inner_lv_bound].integral('(1 - α) ( prvbasis_n qartP0intLV ) d:x' @ self.ns, degree=gaussdeg)  
                    
                ## Store in dictionary
                res      = dict( resplv=resplv, resprv=resprv, resvartS=resvartS, resvenS=resvenS, resartP=resartP )
    
                targets   = ['plvlhs' , 'prvlhs' , 'partSlhs' , 'pvenSlhs' , 'partPlhs' ]
                targets0  = ['plv0lhs', 'prv0lhs', 'partS0lhs', 'pvenS0lhs', 'partP0lhs' ] 
            else:
                resplv -= numpy.maximum( 0, resplvvenLV )  # d:x is inside the qvenPint variable
                resplv -= numpy.maximum( 0, resplvvenLV0 ) # d:x is inside the qvenP0int variable 
                ## Store in dictionary
                res      = dict( resplv=resplv, resvart=resvartS )
                targets   = ['plvlhs' , 'partSlhs' ]
                targets0  = ['plv0lhs', 'partS0lhs'] 
            
            
            if m_type != '0D':
                #resu  = self.topo.boundary[inner_lv_bound].integral('plv n_i ubasis_ni d:x' @ self.ns, degree=4) ## <<--- linear part
                resu  =  self.topo.boundary[inner_lv_bound].integral('( ubasis_nj ( plv n_i ) Finvint_ij detFint ) d:x' @ self.ns, degree=gaussdeg) ## <<--- fung part
                 
                if m_type == 'bv':
                    #resu += self.topo.boundary[inner_rv_bound].integral('prv n_i ubasis_ni d:x' @ self.ns, degree=4) 
                    resu +=  self.topo.boundary[inner_rv_bound].integral('( ubasis_nj ( prv n_i ) Finvint_ij detFint ) d:x' @ self.ns, degree=gaussdeg) ## <<--- fung part
                #resu   = self.topo.boundary[inner_lv_bound].integral('plv n_i ubasis_ni detFint d:x' @ self.ns, degree=4) ## <<--- nonlinear part
                res['resu'] = resu
            
            #variables = ['?artvalve','?artvalve0','?venvalve','?venvalve0']
            return res, targets, targets0
        
        def initialize(self,m_type,p_type,const,inner_boundary):
            # Targets:
            # - plvlhs  , plslhs0
            # - VartSlhs, VartSlhs0
            # - VartPlhs, VartPlhs0
            # - VvenSlhs, VvenSlhs0
            # - VvenPlhs, VvenPlhs0
            
            # Give approximation of arterial pressure
            partS0 =   86.257094*self.mmHg #100.*self.mmHg
            partP0 =   2.*self.mmHg  # [mmHg]*[Pa/mmHg] no idea where to get a good init approx.. perhaps experience?
            ped    =   5.*self.mmHg  # end diastolic pressure [mmHg]*[Pa/mmHg]
            VartS0 = const['VartS0Con']+const['CartS']*partS0
            
            # Construct init dict with standard input
            #init0    = dict( VartSlhs=VartS0, VartS0lhs=VartS0 )
            init0    = dict( partSlhs=partS0, partS0lhs=partS0 )
            # Check closed open etc...
            if p_type == 'closed':
                if m_type == '0D':
                    Vlv0   = ped/const['Epas']#const['Vlv0Con']
                    VvenP0 = const['Vtotal'] - VartS0 - Vlv0   
                    init0['plvlhs']  = ped
                    init0['plv0lhs'] = ped
                    #init0['VvenP0lhs'] = VvenP0
                    #init0['VvenPlhs']  = VvenP0
                elif m_type == 'bv': # Else it is a bv
                
                    init0['partPlhs']  = 35.*self.mmHg #0.45*partS0
                    init0['partP0lhs'] = 35.*self.mmHg #0.45*partS0
                    init0['pvenSlhs']  = 2*self.mmHg
                    init0['pvenS0lhs'] = 2*self.mmHg  
                    init0['plvlhs']    = 0*self.mmHg
                    init0['plv0lhs']   = 0*self.mmHg
                    init0['prvlhs']    = 0*self.mmHg
                    init0['prv0lhs']   = 0*self.mmHg 
                    # self.ns.intC    = self.topo.ndims
                    # if m_type=='bv':
                    #     inner_lv_bound = inner_boundary[0]
                    #     inner_rv_bound = inner_boundary[1]
                    
                else: # Else it is a lv
                
                    init0['partPlhs']  = 0.45*partS0
                    init0['partP0lhs'] = 0.45*partS0
                    init0['pvenSlhs']  = 5*self.mmHg
                    init0['pvenS0lhs'] = 5*self.mmHg   
                    init0['plvlhs']    = 0*self.mmHg
                    init0['plv0lhs']   = 0*self.mmHg                      
                        
                        
                    # else:
                    #     inner_lv_bound = inner_boundary
                
                    # Vlv_int = self.topo.boundary[inner_lv_bound].integral('( 1 / intC ) ( x_i n_i ) d:x' @ self.ns, degree=4) # DetF is not required (there is no deformation at t=0)
                    # Vlv0    = numpy.abs(Vlv_int.eval())
                    # VvenP0  = const['Vtotal'] - VartS0 - Vlv0
                    # init0['plvlhs']  = 0.
                    # init0['plv0lhs'] = 0.
                    # #init0['VvenP0lhs'] = VvenP0
                    # #init0['VvenPlhs']  = VvenP0
                    # if m_type == 'bv': ### <<<------- Bi-Ventricle part still to be added, not sure how this works yet!!
                    #     VartP0 = const['VartP0Con']+const['CartP']*partP0
                    #     # VartP0 = const['VartP0Con']+const['CartP']*partP0
                    #     # VartP0 = const['VartP0Con']+const['CartP']*partP0                               
            #return init0
            return { ikey: numpy.array([ival]) for ikey, ival in init0.items()}
        
        def leftventricle(self,):
            self.ns.intC    = self.topo.ndims                # Dimension 2D or 3D (1D is a bit silly)
            # self.ns.Vlvdx   = '( X_i  n_i ) detF  / intC'    # If 2D, divide by 2
            # self.ns.Vlv0dx  = '( X0_i n_i ) detF0 / intC'    # If 3D, divide by 3  
            #self.ns.Vlvdx   = '- ( X_i  n_i ) detFint / intC'    # If 2D, divide by 2, note the - sign (normal direction should be inverted because we want cavity volume)
            #self.ns.Vlv0dx  = '- ( X0_i n_i ) detFint / intC'    # If 3D, divide by 3 
            
            # before GPA was applied
            #self.ns.Vlvdx   = '- ( ( X_j - H_j )  n_i Finv_ij ) detFint   / intC'    # If 2D, divide by 2, note the - sign (normal direction should be inverted because we want cavity volume)
            #self.ns.Vlv0dx  = '- ( ( X0_j - H_j ) n_i Finv0_ij ) detFint0 / intC'    # If 3D, divide by 3 
            
            # After GPA was applied
            self.ns.Vlvdx   = '- ( ( X_j - H_j )  n_i Finvint_ij ) detFint   / intC'    # If 2D, divide by 2, note the - sign (normal direction should be inverted because we want cavity volume)
            self.ns.Vlv0dx  = '- ( ( X0_j - H_j ) n_i Finvint0_ij ) detFint0 / intC'    # If 3D, divide by 3 
 
 
            self.ns.δVlv    = '( Vlvdx - Vlv0dx )  / δt' # Should still be integrated!

            self.ns.qvenPbdx   = '-Vlvdx / ( RvenP CvenP )'# dx, this is the part that should be integrated over the inner boundary
            self.ns.qvenP0bdx  = '-Vlv0dx / ( RvenP CvenP )'# dx, this is the part that should be integrated over the inner boundary

            self.ns.qperSbdx   = 'Vlvdx  / ( RperS CvenP )'# dx, this is the part that should be integrated over the inner boundary
            self.ns.qperS0bdx  = 'Vlv0dx / ( RperS CvenP )'# dx, this is the part that should be integrated over the inner boundary
            return 
        
        def biventricle(self,):    
            self.ns.intC    = self.topo.ndims                # Dimension 2D or 3D (1D is a bit silly)
            
            self.ns.Vlvdx   = '- ( ( X_j - H_j )  n_i Finvint_ij ) detFint / intC'    # If 2D, divide by 2, note the - sign (normal direction should be inverted because we want cavity volume)
            self.ns.Vlv0dx  = '- ( ( X0_j - H_j ) n_i Finvint0_ij ) detFint0 / intC'    # If 3D, divide by 3 
            self.ns.δVlv    = '( Vlvdx - Vlv0dx )  / δt' # Should still be integrated!

            self.ns.Vrvdx   = '- ( ( X_j - H_j )  n_i Finvint_ij ) detFint / intC'    # If 2D, divide by 2, note the - sign (normal direction should be inverted because we want cavity volume)
            self.ns.Vrv0dx  = '- ( ( X0_j - H_j ) n_i Finvint0_ij ) detFint0 / intC'    # If 3D, divide by 3 
            self.ns.δVrv    = '( Vrvdx - Vrv0dx )  / δt' # Should still be integrated!

            # a is the first part (Vtot-VvenS-VartS-VartP), b the lv and c the rv part
            self.ns.qvenPbdx   = '-Vlvdx / ( RvenP CvenP )'# dx, this is the part that should be integrated over the inner boundary
            self.ns.qvenP0bdx  = '-Vlv0dx / ( RvenP CvenP )'# dx, this is the part that should be integrated over the inner boundary

            self.ns.qvenPcdx   = '-Vrvdx / ( RvenP CvenP )'# dx, this is the part that should be integrated over the inner boundary
            self.ns.qvenP0cdx  = '-Vrv0dx / ( RvenP CvenP )'# dx, this is the part that should be integrated over the inner boundary


            # self.ns.qperPbdx   = 'Vlvdx  / ( RperS CvenP )'# dx, this is the part that should be integrated over the inner boundary
            # self.ns.qperP0bdx  = 'Vlv0dx / ( RperS CvenP )'# dx, this is the part that should be integrated over the inner boundary
            
            # self.ns.qperPcdx   = 'Vrvdx  / ( RperS CvenP )'# dx, this is the part that should be integrated over the inner boundary
            # self.ns.qperP0cdx  = 'Vrv0dx / ( RperS CvenP )'# dx, this is the part that should be integrated over the inner boundary
           
            return 
        
        def leftventricle0D(self,const):
            self.ns.Epas    = const['Epas']
            self.ns.Emax    = const['Emax']
            self.ns.tact    = const['tact']
            self.ns.Vlv0Con = const['Vlv0Con']
            self.ns.pi = numpy.pi
            self.ns.at  = '?tactive ( sin(  ( pi ?t / tact) ) )^2'
            self.ns.Ct  = 'Epas + at ( Emax - Epas )'
            self.ns.at0 = '?tactive ( sin(  ( pi ?t0 / tact) ) )^2'
            self.ns.Ct0 = 'Epas + at0 ( Emax - Epas )'
        
            
            self.ns.Vlv   = '( plv  / Ct  ) + Vlv0Con'
            self.ns.Vlv0  = '( plv0 / Ct0 ) + Vlv0Con'            
            self.ns.δVlv  = '( Vlv  - Vlv0 )  / ( δt Ainner )' # backward euler
            self.ns.VvenP = 'Vtot - VartS - Vlv'

            self.ns.qvenPb   = '-Vlv  / ( RvenP CvenP )' # Second part of the venous flow
            self.ns.qvenP0b  = '-Vlv0 / ( RvenP CvenP )'

            self.ns.qperSb   = 'Vlv  / ( RperS CvenP )' # Second part of the peripherial flow
            self.ns.qperS0b  = 'Vlv0 / ( RperS CvenP )'
            
            # self.ns.qvenP = self.ns.qvenPa + self.ns.qvenPb
            # self.ns.qperS = self.ns.qperSa + self.ns.qperSb
            
            self.ns.pvenPb = 'qvenPb RvenP'
            self.ns.pvenP  =  self.ns.pvenPa + self.ns.pvenPb               
        
            return 
        
        def constants0D(self,):
            constants = {}
            constants['Epas']    = 0.05*self.mmHg/self.ml # converted to correct unit [Pa/m3] 
            constants['Emax']    = 2.24*self.mmHg/self.ml
            constants['tact']    = 400.*self.ms
            constants['Vlv0Con'] =   0.*self.ml         
            constants['ped']     =   5.*self.mmHg
            return constants

        def constants(self,m_type,p_type):
            ##-----------------------
            # S: Systemic part
            # P: Pulmonary part
            # Art: Arterial section
            # Ven: Venous section
            ##-----------------------
            
            # Constance for the Systemic Artial part
            constants = {'CartS'    : 1.53*1e1 * self.ml/self.kPa       , # Compliance
                         'RartS'    : 4.46*1e-3* self.kPa*self.s/self.ml, # Resistance 
                         'VartS0Con': 7.04*1e2 * self.ml                , # Unloaded volume 
                         'RvenP'    : 2.18*1e-3* self.kPa*self.s/self.ml} # Venous resistance
            if p_type == 'open':
                constants['PvenP']     = 8.       * self.mmHg # Prescribed preload
            else: # It is a closed circulation    
                constants['RperS']     = 1.49*1e-1* self.kPa*self.s/self.ml
                constants['Vtotal']    = 5.0 *1e3 * self.ml  # Only in systemic?!
                constants['CvenP']     = 1.53*1e1 * self.ml/self.kPa
                constants['VvenP0Con'] = 5.13*1e2 * self.ml
            if m_type == 'bv':           
                constants['CvenS']     = 4.59*1e1 * self.ml/self.kPa
                constants['CartP']     = 4.59*1e1 * self.ml/self.kPa            
                constants['RvenS']     = 1.10*1e-3* self.kPa*self.s/self.ml
                constants['RartP']     = 2.48*1e-3* self.kPa*self.s/self.ml
                constants['RperP']     = 1.78*1e-2* self.kPa*self.s/self.ml
                constants['VartP0Con'] = 7.83*1e1 * self.ml
                constants['VvenS0Con'] = 3.16*1e3 * self.ml  
 
            return constants