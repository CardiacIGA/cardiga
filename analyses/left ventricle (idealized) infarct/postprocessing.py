import numpy, treelog, os
from nutils import cli, export, function
from cardiga.dataproc import save_pickle, load_pickle

                
def main(nrefine : int, scar_reflvl: int, scar_degr: int, Usample : int, VTKboundaryOnly : bool, convert2Uniform: bool, saveTemporalVTK: bool, saveVideoVTK: bool, saveScarVTK : bool, savePointvalues: bool):
    '''
    .. arguments::

       nrefine [0]
         Number of uniform refinements.  
         
       scar_reflvl [4]
         Refinement of the scar location/topology.

       scar_degr [3]
         Polynomial degree of the scar stiffness. 
           
       Usample [20]
         Number of uniform sampling of the base topology.   
         
       VTKboundaryOnly [False]
         Save only boundary not internal/solid.   
         
       convert2Uniform [True]
         Convert the hierarchical mesh to a uniform one, which is used for post processing tasks (save vtk, computations etc.)
         
       saveTemporalVTK [False]
         Save temporal statistics (minima and maxima of the logarithmic strain) to a vtk file.
       
       saveVideoVTK [False]
         Save the results to a vtk file at every time-step (can be used for a video).
       
       saveScarVTK [False]
         Save the distribution of the scar tissue to a vtk. 
                 
       savePointvalues [False]
         Save the traces of the logarithmic strain at specific locations to a pickle file.
           
    '''
    folder = 'results/'
    dt     = 2    # [ms]
    tstart = 1557*2 #0    # [ms]
    tend   = 1956*2 #712  # [ms]
    tref   = 1750*2  #298  # [ms] reference instance for green-lagrange strain 
    time   = numpy.linspace(tstart,tend,int(tend/2)) 
    
    

    ## Load topo, geom, ns, and lhs 
#    topo, geom, ns = load_pickle(folder+'LeftventricleInfarct_spline{}_nref{}_reflvl{}_space'.format(scar_degr,nrefine,scar_reflvl))
#    LHS            = load_pickle(folder+'LeftventricleInfarct_spline{}_nref{}_reflvl{}_result'.format(scar_degr,nrefine,scar_reflvl))  
   
    topo, geom, ns = load_pickle(folder+'LeftventricleInfarct_ncycle6_spline{}_nref{}_reflvl{}_Nutilsv8_space'.format(scar_degr,nrefine,scar_reflvl))
    LHS            = load_pickle(folder+'LeftventricleInfarct_ncycle6_spline{}_nref{}_reflvl{}_Nutilsv8_result'.format(scar_degr,nrefine,scar_reflvl))  
    
    ## Print number of elements in topology after refinement
    #treelog.info('Number of elements: {}'.format(len(topo.integrate_elementwise(geom, degree=0))))
    #treelog.info(topo.integrate_elementwise(function.J(geom), degree=8)**(1/3))
    
    # Set referentie deformation field  
    ns.uref_i    = 'ubasis_ni ?lhsuref_n'
    
    # Method 1
    #ns.Fpost_ij  = 'd( x_i + u_i, x_j + uref_j )' # Deformation gradient tensor, wrt new reference config
    ns.Fpost_ij  = 'd( x_i + uref_i, x_j )'
    ns.Ffpost_i  = 'Fpost_ij ef_j'
    ns.lsref     = 'lsc0 ( Ffpost_i Ffpost_i )^0.5'
    #ns.FFpost_ij = 'Fpost_ki Fpost_kj'
    #ns.Epost_ij  = '0.5 ( FFpost_ij - δ_ij )' 
    
    # Method 2
#    ns.FpostC_ij = 'd( x_i + uref_i, x_j )'
#    ns.FpostCinv = function.inverse(ns.FpostC)
#    ns.FpostT_ij = 'F_ik FpostCinv_kj'
#    ns.FFpostT_ij = 'FpostT_ki FpostT_kj'
#    ns.EpostT_ij  = '0.5 ( FFpostT_ij - δ_ij )'
    
    # Convert sample to a uniform sampling if specified (based on hierarchical sample)
    if convert2Uniform:
      if VTKboundaryOnly:
         Ubezierbase = topo.basetopo.boundary.sample('bezier', Usample)
      else:
         Ubezierbase = topo.basetopo.sample('bezier', Usample)
      Ubezier     = topo.locate(ns.x, Ubezierbase.eval(ns.x), eps=1e-10)
    else:
      if VTKboundaryOnly:
        Ubezier = topo.boundary.sample('bezier', Usample)
      else:
        Ubezier = topo.sample('bezier', Usample)
      Ubezierbase = Ubezier
      
      
      
    # Save temporal max- and minima as a vtk---------------------------------------------------------------------------------------------------------
    if saveTemporalVTK:
      idx    = range(int(tstart/dt),int(tend/dt)+1)
      X      = Ubezier.eval(geom) # Evaluate geometry ones
      for i in idx: 
        treelog.info("Index value: {}/{}, physical time: {}".format(i,int(tend/dt)-1,dt*i))
        LHS[i].update(dict(lhsuref=LHS[int(tref/dt)]['ulhs']))
        #Eff = Ubezier.eval('ef_i Epost_ij ef_j' @ ns, **LHS[i])
        Myofiber_strain = Ubezier.eval('ln( ls / lsref )' @ ns, **LHS[i])
        
        if i == idx[0]:
          maxStrain = Myofiber_strain
          minStrain = Myofiber_strain
        else: # Add exception for Nan values
          maxStrain = numpy.fmax(maxStrain, Myofiber_strain) # Ignores NaN values #numpy.maximum(maxStrain, Myofiber_strain)
          minStrain = numpy.fmin(minStrain, Myofiber_strain) # Ignores NaN values #numpy.minimum(minStrain, Myofiber_strain)
      
      # Save result to vtk file  
      export.vtk('IGA_MyoFib_FINE_result_scarref{}_scardegr{}'.format(scar_reflvl,scar_degr), Ubezierbase.tri, X, maxStrain=maxStrain, minStrain=minStrain)      
    #------------------------------------------------------------------------------------------------------------------------------------------------
    
    
    
    
    
    
    
    ## Some post-processing: generating vtk's for a video--------------------------------------------------------------------------------------------
    if saveVideoVTK:
      X      = Ubezier.eval(geom)
      idx    = range(int(tstart/dt),int(tend/dt)+1,10)
      for i in idx: 
        treelog.info("Index value: {}/{}, physical time: {}".format(i,len(idx)-1,dt*i))
        
        LHS[i].update(dict(lhsuref=LHS[int(tref/dt)]['ulhs']))
        # U, Eff = bezier.eval(['u_i','ef_i Epost_ij ef_j'] @ ns, **LHS[i])
        # export.vtk('IGA_result_scarref{}_scardegr{}_{}'.format(scar_reflvl,scar_degr,str(i*2).zfill(4)), bezier.tri, X, U=U, Eff=Eff) 
        
        U, σa = Ubezier.eval(['u_i','σa'] @ ns, **LHS[i])
        export.vtk('IGA_result_scarref{}_scardegr{}_{}'.format(scar_reflvl,scar_degr,str(i*2).zfill(4)), Ubezierbase.tri, X, U=U, Active_stress=σa)      
    #------------------------------------------------------------------------------------------------------------------------------------------------
    
    
    
    
    ## Save the distribution of the scar tissue------------------------------------------------------------------------------------------------------
    if saveScarVTK:
      X, a0 = Ubezier.eval(['x_i','a0'] @ ns, **LHS[0])
      export.vtk('IGA_ScarResult_scarref{}_scardegr{}'.format(scar_reflvl,scar_degr), Ubezierbase.tri, X, a0=a0)
    #------------------------------------------------------------------------------------------------------------------------------------------------
    
    
        
    
    
    ## Save values at specific locations-------------------------------------------------------------------------------------------------------------
    if savePointvalues:
      Coordinates = {'A': {'0.15': numpy.array([ 1.42463031, -2.46753208, -1.        ]),
                           '0.5' : numpy.array([ 1.1561023 , -2.00242792, -1.        ]),
                           '0.85': numpy.array([ 0.90226144, -1.56276266, -1.        ])},
                     'B': {'0.15': numpy.array([ 2.75217442, -0.73744291, -1.        ]),
                           '0.5' : numpy.array([ 2.23341814, -0.59844259, -1.        ]),
                           '0.85': numpy.array([ 1.74303525, -0.46704489, -1.        ])},
                     'C': {'0.15': numpy.array([-2.84926062e+00,  3.48933790e-16, -1.00000000e+00]),
                           '0.5' : numpy.array([-2.31220460e+00,  2.83163397e-16, -1.00000000e+00]),
                           '0.85': numpy.array([-1.80452288e+00,  2.20990317e-16, -1.00000000e+00])},
                     'D': {'0.15': numpy.array([ 1.53434247,  1.53434247, -3.44784467]),
                           '0.5' : numpy.array([ 1.18805601,  1.18805601, -3.44784467]),
                           '0.85': numpy.array([ 0.88321464,  0.88321464, -3.44784467])}}
                           
      
      idx    = range(int(tstart/dt),int(tend/dt)+1)
      #idx    = range(0, len(LHS)) 
      folder = os.path.join(folder,'point values (6 ncycle) new strain/')
       
      
      for ikey, ivalue in Coordinates.items():
         for j, (jkey, jvalue) in enumerate(ivalue.items()):
            pointBezier = topo.locate(ns.x, [jvalue*1e-2], eps=1e-10)  # Decrease by factor 1e-2, different scales from [cm] to [m]
            
            #Eff = numpy.zeros(len(idx))  
            Myofiber_strain = numpy.zeros(len(idx))  
            for k, i in enumerate(idx): 
              treelog.info("Index value: {}/{}, physical time: {}".format(i,len(idx)-1,dt*i))
             
              LHS[i].update(dict(lhsuref=LHS[int(tref/dt)]['ulhs']))
              #Eff[i] = pointBezier.eval('ef_i Epost_ij ef_j' @ ns, **LHS[i])    
              Myofiber_strain[k] = pointBezier.eval('ln( ls / lsref )' @ ns, **LHS[i])    
            save_pickle(time, Myofiber_strain ,filename=folder + 'Location {} Reflvl {} Point {}'.format(ikey,scar_reflvl,j+1))
      #------------------------------------------------------------------------------------------------------------------------------------------------
      
      
      
              
    
    ## Test the Lagrange strain methods:
#    ttest = int(450/dt)
#    LHS[ttest].update(dict(lhsuref=LHS[int(tref/dt)]['ulhs']))
#    X, EffM1, EffM2 = bezier.eval(['x_i', 'ef_i Epost_ij ef_j', 'ef_i EpostT_ij ef_j'] @ ns, **LHS[ttest])
#    export.vtk('IGA_testing_GLS_methods'.format(scar_reflvl,scar_degr), bezier.tri, X, EffM1=EffM1, EffM2=EffM2)

#    bezier = topo.sample('bezier', Usample)
#    ttest = int(450/dt)
#    LHS[ttest].update(dict(lhsuref=LHS[int(tref/dt)]['ulhs']))
#    X, ls, ls0 = bezier.eval(['x_i', 'ls', 'ls0ref'] @ ns, **LHS[ttest])
#    export.vtk('IGA_testing_GLS_methods'.format(scar_reflvl,scar_degr), bezier.tri, X, ls=ls, ls0=ls0)
    return
           
if __name__ == "__main__":
   cli.run(main)     