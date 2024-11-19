# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 15:31:15 2021

@author: R. Willems
"""

from cardiga import multimodels as multm #import Models
from cardiga.solvers import my_solver
from nutils import solver, cli, export
from cardiga.geometry import Geometry
from cardiga.dataproc import save_pickle
from cardiga.postprocess import Graph
import numpy, treelog, math, os


## Bi-ventricle (idealized) simulation
def main(geomName: str, nrefine: int, dt: float, ncycle: int, TimeIntScheme: float, btype: str, bdegree: int, qdegree: int, saveVTK: bool, savePost: bool):
    '''
    .. arguments::

       geomName [REF]
         Geometry filename (extension) to be used, can be: 'REF', 'LONG', 'THK'.

       nrefine [1]
         Number of uniform refinements.

       dt [2]
         Physical time step of the simulation in milliseconds.

       ncycle [8]
         Number of cardiac cycles to be simulated.

       TimeIntScheme [0.5]
         Integrationscheme used for time-dependent problem, i.e. 0 = Explicit, 0.5 = Crank-Nicolson, 1 = Implicit.

       btype [spline]
         Basis function type, i.e. splines, std, th-spline, ...

       bdegree [3]
         Basis function degree to be used.

       qdegree [5]
         Quadrature degree to be used.

       saveVTK [False]
         Save data to vtk files.

       savePost [False]
         Save data to .pickle files used for postprocessing.

    '''
    ml=1e-6;mm=1e-3;mmHg=133.3223684;ms=1e-3;s=1;kPa=1e3 
    
    ## Load the geometry and topology
    directC    = os.path.realpath(os.path.dirname(__file__)) # Current directory
    directI    = os.path.split(os.path.split(directC)[0])[0]            # Remove the last 2 folders to get in the main working directory 
    direct     = 'geometries'  
    filename   = f'BV_GEOMETRY_{geomName}.pickle' # or .txt
    Ventricle  = Geometry(filename, direct=os.path.join(directI, direct))
    topo,geom  = Ventricle.get_topo_geom(nrefine=nrefine)
    treelog.info('Number of elements: {}'.format(len(topo.integrate_elementwise(geom, degree=0)))) # Print number of elements in topology after refinement
    
    ## Time-settings
    dt    *= ms            # Convert time step from [ms] to [s]
    tcycle = 800*ms        # Total cycle duration
    tactiv = 300*ms        # Time at which activation is initiated
    trelax = 294*ms        # Time to wait after tcycle has ended trelax < tactiv
    Tmax   = ncycle*tcycle # Total simulation duration
    nT     = Tmax/dt       # Number of time increments
    nTin   = math.ceil(nT) # Integer
    time   = numpy.linspace(0,nTin*dt,nTin+1)  
    

    ## Specify the boundary conditions
    boundary_cond = {'fixed (normal)'     : ('base_l',   0.     ), # Fix normal displacement at base
                     'fixed (normal)'     : ('base_r',   0.     ), # Fix normal displacement at base
                     'spring (alldirect)' : ('epi'   ,  1e4     )} # Add spring for additional (pericardium) stiffness in all directions 
    

    ## Set hymodynamic parameters (tuned such that a physical PV-loop is obtained)
    constants_hemo = {'RartS'     : 4.29*1e-3*2 * kPa*s/ml,
                      'RvenS'     : 8.57*1e-4   * kPa*s/ml,
                      'RperS'     : 1.39*1e-1   * kPa*s/ml,
                      
                      'RartP'     : 8.57*1e-4 * kPa*s/ml,
                      'RvenP'     : 8.57*1e-4 * kPa*s/ml,
                      'RperP'     : 2.77*1e-2 * kPa*s/ml,
                      
                      'CartS'     : 1.76*1e1      * ml/kPa  ,
                      'CvenS'     : (1.06/30)*1e3 * ml/kPa  ,
                      'CartP'     : 5.06*1e1      * ml/kPa  ,
                      'CvenP'     : 1.26*1e2      * ml/kPa  ,
                      
                      'VartS0Con' : 6.30*1e2  * ml      , 
                      'VvenS0Con' : 2.52*1e3  * ml      ,
                      'VartP0Con' : 7.00*1e2  * ml      ,
                      'VvenP0Con' : 2.80*1e2  * ml      ,
                      
                      # Time-model parameters
                      'α'         : TimeIntScheme       , 
                      'δt'        : '?dt'               , 
                      'H'         : Ventricle.height()  }
    
    ## Sarcomere dynamics parameter values
    constants_sarc = {'alphat'    : TimeIntScheme,
                      'T0'        : 185e3}

    ## Assemble complete model from its individual submodel components
    submodel = multm.Models(topo, geom, boundary_cond) # ..,fiber=Rossi), if not selected, isotropic behavior is used and or x-direction for activation stress 
    fibers   = submodel.fiber('rossi','bv')         # dict with fiber solution, constants/variables are stored in namespace 
    multi    = submodel.passive('bovendeerd-quasi', btype=btype, bdegree=bdegree, quad_degree=qdegree)
    multi   += submodel.active( 'kerckhoffs',       btype=btype, bdegree=bdegree, quad_degree=qdegree, const_input=constants_sarc) 
    multi   += submodel.lumpedparameter('bv', 'closed', ('endo_l','endo_r'), const_input=constants_hemo)
    
    
    ## Unpack for clarity
    res       = list(multi.res.values()) # Residuals:  list[ resu, resp, .. ]
    args      = multi.arguments          # Arguments:  list[ 'ulhs', 'plhs', '..']
    cons      = multi.cons               # Constrains: dict( argument = cons_array )
    ns        = multi.ns                 # Namespace
    initial   = multi.initial            # Initial vals: dict( 'ulhs' = array(...), .. )  
    # variables = multi.variables        # list[ ?c, ?t , .. ]

    
    ## Scale the residual array accordingly (all residuals of the coupled problem have values of similar order)
    ScD   = dict(m=mm, dt=ms, k=kPa)
    scale = [     1    / (ScD['m']**2*ScD['k']), 
             ScD['dt'] /  ScD['m']**3          , 
             ScD['dt'] /  ScD['m']**3          ,
             ScD['dt'] /  ScD['m']**3          ,
             ScD['dt'] /  ScD['m']**3          ,
             ScD['dt'] /  ScD['m']**3          ,
             ScD['dt'] /  ScD['m']**3          ] # [u, lc, plv, prv, partS, pvenS, partP]
    assert len(scale) == len(res), "Scaling array has different lenth than residual, perhaps you added an additional argument (boundary condition, ..)"                  
    res = [ires*iscale for ires, iscale in zip(res, scale)] # Rescale the residual array


    if savePost:
      fileName   = "Biventricle_{}_spline{}_nref{}".format(geomName,bdegree,nrefine)
      fileSpace  = fileName + "_space" 
      postpSpace = os.path.join(directC,"results",fileSpace)
      save_pickle(topo,geom,ns,filename=postpSpace)
      
    ## Initialize arrays, values, dictionaries, etc.
    QVENLV = numpy.zeros(len(time))
    QARTLV = QVENLV.copy()
    QPERLV = QVENLV.copy()
            
    VLV    = QVENLV.copy()
    VARTLV = QVENLV.copy()
    VVENLV = QVENLV.copy()
    VTOT   = QVENLV.copy()

    PLV    = QVENLV.copy()
    PARTLV = QVENLV.copy()
    PVENLV = QVENLV.copy()    

    QVENRV = QVENLV.copy()
    QARTRV = QVENLV.copy()
    QPERRV = QVENLV.copy()
            
    VRV    = QVENLV.copy()
    VARTRV = QVENLV.copy()
    VVENRV = QVENLV.copy()
    
    PRV    = QVENLV.copy()
    PARTRV = QVENLV.copy()
    PVENRV = QVENLV.copy()     


    ## Initialize for-loop parameters
    if saveVTK:
      bezier  = topo.boundary.sample('bezier', 15)
    argkeys = {"Current"  : ('plvlhs' ,'partSlhs' ,'prvlhs' ,'partPlhs' ,'pvenSlhs' ,'ulhs' ,'lhslc' ) , # Current time-step
               "Previous" : ('plv0lhs','partS0lhs','prv0lhs','partP0lhs','pvenS0lhs','ulhs0','lhslc0') } # Previous time-step
    
    arguments     = {}   # Empty arguments dict which is filled in the for-loop
    resnorm       = []   # Empty list to which the residual norm is appended
    startcycleLV  = True # Indicates if new cycle is started for left ventricle 
    startcycleRV  = True # Indicates if new cycle is started for right ventricle
    ncycle_old    = 0    # Old cycle index (starts at 0)
    saveID        = 0    # Save ID used to for filenaming
    save_everynth = 100  # Save every n-th time the results to a vtk-file
    constrain_lc  = True # Constrain the contractile part of the sarcomere during filling
    Newtontol     = 1e-6 #1e-8 # Newton tolerance
    graph         = Graph()            
    with treelog.iter.plain('timestep', time) as timesteps:
      for timestep, t in enumerate(timesteps):
            
            # Convert time to specific cycle time ts
            ncycle_new =  math.floor( t / tcycle )
            if ncycle_new > ncycle_old: # we entered a new cycle
               if t - ncycle_old*tcycle < tcycle + trelax:
                  ts = t - ncycle_old*tcycle
               else:
                  ncycle_old = ncycle_new
                  ts = t - ncycle_new*tcycle
            else:
               ts = t - ncycle_new*tcycle
               
            if timestep == 0:
                arguments['t']         = numpy.array(ts) 
                arguments['ta']        = numpy.array(ts-tactiv)
                arguments['dt']        = numpy.array(dt)
                arguments['ulhs']      = initial['ulhs0']
                arguments['lhslc']     = initial['lhslc0']
                arguments.update(initial)
                arguments.update(fibers)
                lhs  = arguments.copy()
                lhs0 = lhs
                #Vlv[timestep] = topo.boundary['inner'].integral('- 0.5 Hi n_i x_i d:x' @ ns, degree=4).eval()
            else:
                arguments['t']         = numpy.array(ts)
                arguments['ta']        = numpy.array(ts-tactiv)
                arguments['dt']        = numpy.array(dt) # Reset dt in case of adaptive dt                  
                for current_arg, previous_arg in zip(argkeys["Current"],argkeys["Previous"]):
                    arguments[previous_arg] = lhs0[current_arg]  # Assign current argument of previous solution to previous argument of current step
                    arguments[current_arg]  = lhs0[current_arg]  # Assign current argument of previous solution to current argument to be solved (initial guess)              
                

                # if the elasticity problem is solved using a mixed-formulation
                if timestep > 1 and 'plhs' in lhs0: 
                   arguments['plhs']      = lhs0['plhs']
                   arguments['plhs0']     = lhs0['plhs']
                   
                  
                # During filling, the sarcomere contractile length (lc) should not be computed but kept constant to ls  
                if (ts-tactiv < 0 or numpy.isclose(ts-tactiv,0,ms*1e-3)):
                  arguments.pop('lhslc', None) # Remove from arguments before minimizing this target 
                  sqr_lc  = topo.integral('( lc - ls )^2 d:x' @ multi.ns, degree=4)                
                  init_lc = solver.optimize('lhslc', sqr_lc, arguments=arguments, tol=1e-6, droptol=1e-15)  
                  lhs['lhslc']         = init_lc # Simply update the solution, this will not affect other results, only used for post-processing step (during passive filling the lhslc = contrained
                  cons['lhslc']        = init_lc # Add to constrains
                  arguments['lhslc0']  = init_lc # Add to arguments as previous time-step value
                  constrain_lc = True
                elif ts-tactiv > 0 and constrain_lc:  
                  treelog.info('Entered lhslc pop')
                  cons.pop('lhslc', None) 
                  constrain_lc = False  
                

                #TODO Increment pressure or get a better initial displacement (first timestep only)
                # Nincrem = 10
                # if timestep == 1:
                #   consI      = cons.copy()
                #   argumentsI = arguments.copy()
                #   PvenP   = 230 #ns(**lhs).pvenP.eval()
                #   PvenS   = 14 #ns(**lhs).pvenS.eval()
                #   print(PvenP,PvenS)
                #   p_left  = numpy.linspace(PvenP*1e-3, PvenP,Nincrem+1)
                #   p_right = numpy.linspace(PvenS*1e-3, PvenS,Nincrem+1)
                #   for i, (Pl, Pr) in enumerate(zip(p_left, p_right)):
                #       treelog.info('Pressure increment: {}/{}'.format(i+1,Nincrem))
                #       #consI['plvlhs'] = numpy.array(Pl)
                #       #consI['prvlhs'] = numpy.array(Pr)
                #       argumentsI['plvlhs'] = numpy.array([Pl])
                #       argumentsI['prvlhs'] = numpy.array([Pr])
                #       #argumentsI.pop('plvlhs')
                #       #argumentsI.pop('prvlhs')
                #       lhsI, resnorm  = my_newton_solver(('ulhs','lhslc'), res[:2], constrain=consI, arguments=argumentsI, tol=1e-8, nstep=25, elaborate_res=False)
                      
                #       argumentsI['ulhs']     = lhsI['ulhs']
                #       #argumentsI['ulhs0']    = lhsI['ulhs'] 
                #       #my_newton_solver(('ulhs','lhslc'), res[:-1], constrain=consI, arguments=arguments, tol=1e-8, nstep=25)
                #   arguments['ulhs']     = lhsI['ulhs']
                #   #arguments['ulhs0']    = lhsI['ulhs']               
                ##-----------------------------------------------------------------------------
                #linesearch = solver.NormBased(minscale=0.85, acceptscale=0.9, maxscale=10)  # or None
                #lhs  = solver.newton(tuple(args), res, constrain=cons, arguments=arguments, linesearch=linesearch).solve(tol=1e-8)   
                
                ## Solve the system of (nonlinear) equations
                lhs, resnorm  = my_solver(tuple(args), res, cons, arguments, tol=Newtontol, nstep=15, adaptive=True, elaborate_res=True)
                lhs0 = lhs.copy()
                                
                         
                
        


                 
            ##---------------------------------------------##    
            ##              Post processing                ##    
            ##---------------------------------------------##    
                

            ## Store left ventricle hemodynamics data--------------------------------------------------------------# 
            QVENLV[timestep] = ns(**lhs).qvenP.eval()
            QARTLV[timestep] = ns(**lhs).qartS.eval()
            QPERLV[timestep] = ns(**lhs).qperS.eval()  
            
            VLV[timestep]    = ns(**lhs).Vlv.eval()
            VARTLV[timestep] = ns(**lhs).VartS.eval()
            VVENLV[timestep] = ns(**lhs).VvenP.eval()

            PLV[timestep]    = ns(**lhs).plv.eval()
            PARTLV[timestep] = ns(**lhs).partS.eval()
            PVENLV[timestep] = ns(**lhs).pvenP.eval()

            # Store right ventricle hemodynamics data
            QVENRV[timestep] = ns(**lhs).qvenS.eval()
            QARTRV[timestep] = ns(**lhs).qartP.eval()
            QPERRV[timestep] = ns(**lhs).qperP.eval()  
            
            VRV[timestep]    = ns(**lhs).Vrv.eval()
            VARTRV[timestep] = ns(**lhs).VartP.eval()
            VVENRV[timestep] = ns(**lhs).VvenS.eval()

            PRV[timestep]    = ns(**lhs).prv.eval()
            PARTRV[timestep] = ns(**lhs).partP.eval()
            PVENRV[timestep] = ns(**lhs).pvenS.eval()

            # Total volume (should be preserved)
            VTOT[timestep]   =  VLV[timestep] + VARTLV[timestep] + VVENLV[timestep] + VRV[timestep] + VARTRV[timestep] + VVENRV[timestep]

            # Print initial volumes
            if timestep == 0:
               treelog.info('Initial volume LV: {}'.format(VLV[0]/ml))
               treelog.info('Initial volume RV: {}'.format(VRV[0]/ml))

            treelog.info('Venous pressure (Systemic):  {} [mmHg]'.format(PVENRV[timestep]/mmHg))
            treelog.info('Venous pressure (Pulmonary): {} [mmHg]'.format(PVENLV[timestep]/mmHg))
            
            treelog.info("Pressure left {}\nPressure right: {}".format(PLV[timestep]/mmHg,PRV[timestep]/mmHg))
            treelog.info("Isovolumetric (left) when > 0 {}".format(PLV[timestep]/mmHg - PVENLV[timestep]/mmHg ) )
            treelog.info("Isovolumetric (right) when > 0 {}".format(PRV[timestep]/mmHg - PVENRV[timestep]/mmHg ) )
            treelog.info("Ejection (left) when < 0 {}".format(PARTLV[timestep]/mmHg - PLV[timestep]/mmHg ) )
            treelog.info("Ejection (right) when < 0 {}".format(PARTRV[timestep]/mmHg - PRV[timestep]/mmHg ) )
            
            treelog.info('Pressure difference (Partp - Pvenp) = {}, volumeflow: {}'.format(PARTRV[timestep] - PVENLV[timestep], QPERRV[timestep]) )
            
            ##------------------------------------------------------------------------------------------------------#
            

            
            ## If we enter new cycle, print previous stroke volume (SV)---------------------------------------------#
            # Left side
            if PLV[timestep] > PVENLV[timestep] and startcycleLV:
               istart       = timestep
               startcycleLV = False
            if PLV[timestep] < PVENLV[timestep] and not startcycleLV:
               iend         = timestep
               startcycleLV = True
               treelog.info('Left SV: {:.4f}'.format( (VLV[istart] - VLV[iend])/ml )) 
                
            # Right side
            if PRV[timestep] > PVENRV[timestep] and startcycleRV:
               istart       = timestep
               startcycleRV = False
            if PRV[timestep] < PVENRV[timestep] and not startcycleRV:
               iend         = timestep
               startcycleRV = True
               treelog.info('Right SV: {:.4f}'.format( (VRV[istart] - VRV[iend])/ml ))             
            ##-----------------------------------------------------------------------------------------------------#  
               


            ## Visualize hemodymics curve  Postprocessing----------------------------------------------------------#  
            if timestep in range(0,len(time))[::save_everynth] or timestep == len(time):  
              graph.overview(time, P=[PLV,PARTLV,PVENLV], V=[VLV,], Q=[QPERLV,QARTLV,QVENLV])
              graph.overview(time, P=[PRV,PARTRV,PVENRV], V=[VRV,], Q=[QPERRV,QARTRV,QVENRV], plabel=("RV","Art","Ven"), qlabel=("RV","Art","Ven"), vlabel=("RV",), filename='Rightventricle')
              graph.pressure_volume(V=[VLV,VRV], P=[PLV,PRV])

              graph.residual(resnorm)
              graph.total_volume(time,VTOT)
            ##-----------------------------------------------------------------------------------------------------#



            ## Save hemodynamics and results-array (lhs) in .pickle files every timestep---------------------------#
            if savePost:
              # Save hemodynamics 
              fileHemo  = fileName + "_Hemodynamics"
              postpHemo = os.path.join(directC,"results",fileHemo)
              save_pickle([ VLV, [PLV, PARTLV, PVENLV], [QVENLV, QARTLV, QPERLV] ], [ VRV, [PRV, PARTRV, PVENRV], [QVENRV, QARTRV, QPERRV] ], filename=postpHemo)
              append = False if timestep == 0 else True

              # Save spatial results
              fileResult  = fileName + "_result"
              postpResult = os.path.join(directC,"results",fileResult)
              save_pickle(lhs,filename=postpResult,append=append)
            ##-----------------------------------------------------------------------------------------------------#
            

            
            ## Save spatial results to .vtk-file-------------------------------------------------------------------#
            if saveVTK:
              if timestep ==0:
                 continue
              elif timestep in range(0,len(time))[::save_everynth]:# and timestep > 800:       
                saveID += 1
                name = 'Biventricle_result_{}_dt{}ms'.format(geomName,dt) + '_' + str(saveID).zfill(3)
                X, U, stressff, Eff, sigma_a, lc = bezier.eval(['x_i','u_i','ef_i stress_ij ef_j', 'ef_i E_ij ef_j', 'σa','lc'] @ ns, **lhs0)
                #strain  = bezier.eval(ns.eval_ij('fiberbase_ti E_tk fiberbase_kj'), **lhs0)
                #stress  = bezier.eval(ns.eval_ij('stress_ij'), **lhs0)
                export.vtk(name, bezier.tri, X, U=U, stressff=stressff, Eff=Eff, sigma_a=sigma_a, lc=lc)
            ##----------------------------------------------------------------------------------------------------#  


    
                                                 
if __name__ == "__main__":
   cli.run(main)                