# -*- coding: utf-8 -*-
"""
Created on Wed Dec 8 2021

@author: R. Willems
"""

from cardiga import multimodels as multm
from nutils import solver, cli, export
import numpy, treelog, math, os
from cardiga.solvers import my_solver
from cardiga.dataproc import save_pickle, load_pickle
from cardiga.geometry import Geometry
from cardiga.postprocess import Graph
from cardiga.infarct import Infarct 
            
## Left ventricle (idealized) with infarct simulation
def main(nrefine: int, dt: float, ncycle: int, TimeIntScheme: float, btype: str, bdegree: int, qdegree: int, scar_reflvl : int, scar_degr : int, saveVTK : bool, savePost : bool, surpressJump : bool):
    '''
    .. arguments::

       nrefine [0]
         Number of uniform refinements.

       dt [2]
         Physical time step of the simulation in milliseconds.

       ncycle [1]
         Number of cardiac cycles to be simulated.

       TimeIntScheme [0.]
         Integrationscheme used for time-dependent problem, i.e. 0 = Explicit, 0.5 = Crank-Nicolson, 1 = Implicit.

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

       saveVTK [False]
         Save results to vtk file.

       savePost [False]
         Save data for post-processing tasks.
         
       surpressJump [False]
         Option to surpress interface discontinuity jump.

    '''
    ml=1e-6;mm=1e-3;mmHg=133.3223684;ms=1e-3;s=1;kPa=1e3;cm=1e-2;   

    directC    = os.path.realpath(os.path.dirname(__file__)) # Current directory
    directI    = os.path.split(os.path.split(directC)[0])[0]            # Remove the last 2 folders to get in the main working directory 
    direct     = 'geometries'  
    filename   = 'LV_GEOMETRY.pickle' # or .txt
    Ventricle  = Geometry(filename, direct=os.path.join(directI, direct))
    topo,geom  = Ventricle.get_topo_geom(nrefine=nrefine)
    
    # Load scar topology from separate file if applicable
    # directInfarct   = "data infarct"
    # filenameInfarct = "ScarTissue_ref{}_degree{}_gaussdeg5".format(scar_ref, scar_degr)
    # topoHierarch, scarValues, gauss = load_pickle(os.path.join(directC, directInfarct, filenameInfarct))
    # infarct = Infarct(topoHierarch, geom) # Construct scar 
    # topoInf, scar = infarct.project(scarValues, gauss, bdegree=scar_degr, gdegree=5)   # Project data onto basis 
    # topoInf, scar = infarct.convolute() # Convolute data onto basis
    
    infarct = Infarct(topo, geom) # Construct scar
    θscar   = (1.75, 2.50)
    φscar   = (-0.350,0.350)#(-0.175, 0.175)
    ξscar   = (0.371296808757, 0.678355651828)#(1.35*0.371296808757, 0.8*0.678355651828)
    topoInf, scar, scarlhs = infarct.analytic(θscar, φscar, ξscar, improve=True, btype=btype, bdegree=scar_degr, gaussdegr=qdegree, reflvl=scar_reflvl, constrainBase=True, saveVTK=True)  # Construct based on analytic expression/input

    topo = topoInf
    treelog.info('Number of elements: {}'.format(len(topo.integrate_elementwise(geom, degree=0)))) # Print number of elements in topology after refinement


    ## Time-settings
    dt    *= ms   # time-step
    tcycle = 800*ms
    tactiv = 300*ms # Time at which activation is initiated
    trelax = 292*ms # time to wait after tcycle has ended trelax < tactiv
    Tmax   = ncycle*tcycle #800.*ms
    nT     = Tmax/dt # Number of time increments
    nTin   = math.ceil(nT) #Integer
    time   = numpy.linspace(0,nTin*dt,nTin+1)  
      
      
    ## Boundary conditions
    boundary_cond = {'fixed (normal)'  : ('base_l',     0.     ),  # Fix normal displacement at base
                     'rigid body'      : ('base_l', 'nullspace')}  # Fix 'nodal' points or nullspace to prevent rigid body motion
      
    # Circulatory system parameters
    constants_hemo = {'CartS'    : 25       * ml/kPa  , # Compliance
                      'RartS'    : 0.010    * kPa*s/ml, # Resistance 
                      'VartS0Con': 500      * ml      , # Unloaded volume 
                      'RvenP'    : 0.002    * kPa*s/ml, # Venous resistance 
                      'RperS'    : 0.120    * kPa*s/ml,
                      'Vtotal'   : 5.0*1e3  * ml      ,  
                      'CvenP'    : 600.     * ml/kPa  ,
                      'VvenP0Con': 3000     * ml      ,
                      
                      # Time-model parameters
                      'α'        : TimeIntScheme      ,
                      'δt'       : '?dt'              ,
                      'H'        : Ventricle.height() }
    
    # Passive model parameters
    a0 = (scar*3.6 + 0.4)*1e3 # passive stiffness relation
    constants_pass = {'a0'       :   a0     , # [-],
                      'a1'       :  3.0     , # [-]
                      'a2'       :  6.0     , # [-]
                      'a3'       :  3.0     , # [-]
                      'a4'       :  0.      , # [-]
                      'a5'       : 55.*1e3  } # [kPa]
        
    # Active model parameters
    T0 = (1 - scar)*140e3 # Active tension relation
    constants_act = {'alphat': TimeIntScheme, # [-], Crank-Nicholson
                     'Ea'    :     20.      , # [1/μm]
                     'T0'    :     T0       , # [Pa]
                     'v0'    :     7.5      , # [μm/s]
                     'lcc0'  :     1.5      , # [μm]
                     'lsc0'  :     1.9      , # [μm]
                     'tr'    :     0.075    , # [s]
                     'td'    :     0.150    , # [s]
                     'b'     :     0.16     , # [s/μm]
                     'ld'    :    -1.0      , # [μm]
                     'a6'    :     2.0      , # [1/μm]
                     'a7'    :     1.5      } # [μm]    
    
    
    submodel = multm.Models(topo, geom, boundaries=boundary_cond)     # ..,fiber=Rossi), if not selected, isotropic behavior is used and or x-direction for activation stress  
    fibers   = submodel.fiber('analytic', 'lv'    , btype=btype, quad_degree=qdegree) #TODO check how to elevate fiber qdegree!
    multi    = submodel.passive('bovendeerd-quasi', btype=btype, bdegree=bdegree, quad_degree=qdegree, const_input=constants_pass, surpressJump=surpressJump) #, rigidb_cons=True)
    multi   += submodel.active('kerckhoffs'       , btype=btype, bdegree=bdegree, quad_degree=qdegree, const_input=constants_act )
    multi   += submodel.lumpedparameter('lv','closed',('endo_l'), const_input=constants_hemo)

    
    
    ## Unpack for clarity
    ScD       = dict(m=cm, dt=dt, k=kPa)
    scale     = [     1    / (ScD['m']**2*ScD['k']), 
                 ScD['dt'] /  ScD['m']**3          , 
                 ScD['dt'] /  ScD['m']**3          ,
                 ScD['dt'] /  ScD['m']**3          ,
                      1    /  ScD['m']**3          ,
                      1    /  ScD['m']**3          ,
                      1    /  ScD['m']**4          ] # [u, lc, plv, partS, lx, ly]                 
    res       = [ires*iscale for ires, iscale in zip(list(multi.res.values()), scale)] # Scale the individual residual components accordingly
    args      = multi.arguments   # list[ 'ulhs', 'plhs', '..']
    cons      = multi.cons        # dict( argument = cons_array )
    ns        = multi.ns          # namespace
    initial   = multi.initial     # Initial vals: dict( 'ulhs' = array(...), .. )
    # variables = multi.variables   # list[ ?c, ?t , .. ]





   
    #postfilename = "LV_infarct_degree{}_ref{}".format(scar_degr,scar_ref)
    if savePost:
      fileName   = f"LeftventricleInfarct_ncycle{ncycle}_spline{bdegree}_nref{nrefine}_reflvl{scar_reflvl}_Nutilsv8"
      fileSpace  = fileName + "_space" 
      postpSpace = os.path.join(directC,"results",fileSpace)
      save_pickle(topo,geom,ns,filename=postpSpace)

    ## Initialize arrays, values, dictionaries, etc.
    QVENLV = numpy.zeros( len(time) )
    QARTLV = QVENLV.copy()
    QPERLV = QVENLV.copy()
            
    VLV    = QVENLV.copy()
    VARTLV = QVENLV.copy()
    VVENLV = QVENLV.copy()
    VTOT   = QVENLV.copy()

    PLV    = QVENLV.copy()
    PARTLV = QVENLV.copy()
    PVENLV = QVENLV.copy()    
            
    ## Initialize for-loop parameters
    if saveVTK:
      bezier  = topo.boundary.sample('bezier', 15)
    argkeys = {"Current"  : ('plvlhs' ,'partSlhs' ,'ulhs' ,'lhslc' ) , # Current time-step
               "Previous" : ('plv0lhs','partS0lhs','ulhs0','lhslc0') } # Previous time-step
    
    saveID        = 0
    save_everynth = 50
    constrain_lc  = True
    Newtontol     = 1e-6
    ncycle_old    = 0 
    resnorm       = []   
    arguments     = {}        
    graph         = Graph()  
    with treelog.iter.plain('timestep', time) as timesteps:
      for timestep, t in enumerate(timesteps):
            treelog.info("Physical time {}".format(t))

            
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
                arguments['dt']        = numpy.array(dt)
                arguments['ta']        = numpy.array(ts-tactiv) #numpy.array(ts-tactiv) # Time after activation
                arguments['ulhs']      = initial['ulhs0']
                arguments['lhslc']     = initial['lhslc0'] 
                if surpressJump:
                  arguments['elem']      = multi.Ielem.length_normal(arguments)  # Get the length of the element at the patch interfaces in normal direction to it (require for jump)
                arguments.update(initial)                
                arguments.update(fibers)   
                arguments.update(scarlhs)          
                lhs  = arguments.copy()
                lhs0 = lhs
            else:   
                arguments['t']  = numpy.array(ts)
                arguments['ta'] = numpy.array(ts-tactiv)
                arguments['dt'] = numpy.array(dt)
                if surpressJump:
                  arguments['elem'] = multi.Ielem.length_normal(lhs0)
                  
                for current_arg, previous_arg in zip(argkeys["Current"],argkeys["Previous"]):
                    arguments[previous_arg] = lhs0[current_arg]  # Assign current argument of previous solution to previous argument of current step
                    arguments[current_arg]  = lhs0[current_arg]  # Assign current argument of previous solution to current argument to be solved (initial guess)              
                
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


                ## TODO Increment pressure or get a better initial displacement (first timestep only)
#                Nincrem = 10
#                if timestep == 1:
#                  argumentsI = arguments.copy()
#                  PlvI    = 45 #[Pa]
#                  p_left  = numpy.linspace(PlvI*1e-3, PlvI,Nincrem+1)
#                  for i, Pl in enumerate(p_left):
#                      treelog.info('Pressure increment: {}/{}'.format(i+1,Nincrem))
#                      argumentsI['plvlhs'] = numpy.array([Pl])
#                      lhsI, resnorm  = my_newton_solver(('ulhs','lhslc'), res[:2], constrain=consI, arguments=argumentsI, tol=1e-8, nstep=25)
#                      argumentsI['ulhs']     = lhsI['ulhs']
#                      
#                  arguments['ulhs']     = lhsI['ulhs']              
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
            #                 
            # Left ventricle data 
            QVENLV[timestep] = ns(**lhs).qvenP.eval()
            QARTLV[timestep] = ns(**lhs).qartS.eval()
            QPERLV[timestep] = ns(**lhs).qperS.eval()  
            
            VLV[timestep]    = ns(**lhs).Vlv.eval()
            VARTLV[timestep] = ns(**lhs).VartS.eval()
            VVENLV[timestep] = ns(**lhs).VvenP.eval()

            PLV[timestep]    = ns(**lhs).plv.eval()
            PARTLV[timestep] = ns(**lhs).partS.eval()
            PVENLV[timestep] = ns(**lhs).pvenP.eval()
            
            # Total volume (should be preserved)
            VTOT[timestep]  =  VLV[timestep] + VARTLV[timestep] + VVENLV[timestep] 

            # Print initial volumes
            if timestep == 0:
               treelog.info("Initial volume LV: {}".format(VLV[0]/ml))
            treelog.info("Arterial pressure: {} [mmHg]".format(PARTLV[timestep]/mmHg))
            treelog.info("Venous pressure  : {} [mmHg]".format(PVENLV[timestep]/mmHg))
            treelog.info("Pressure left    : {} [mmHg]".format(PLV[timestep]/mmHg))
            ##-----------------------------------------------------------------------------------------------------#

            ## Save hemodynamics and results-array (lhs) in .pickle files every timestep---------------------------#
            if savePost:
              # Save hemodynamics 
              fileHemo  = fileName + "_Hemodynamics"
              postpHemo = os.path.join(directC,"results",fileHemo)
              save_pickle(VLV, [PLV, PARTLV, PVENLV], [QVENLV, QARTLV, QPERLV], filename=postpHemo)
              append = False if timestep == 0 else True

              # Save spatial results
              fileResult  = fileName + "_result"
              postpResult = os.path.join(directC,"results",fileResult)
              save_pickle(lhs,filename=postpResult,append=append)
            ##-----------------------------------------------------------------------------------------------------#
            


            ## Save spatial results to .vtk-file-------------------------------------------------------------------#
            if saveVTK:
              if timestep in range(0,len(time))[::save_everynth]:# and timestep > 800:       
                saveID += 1
                name = f'Leftventricle_result_dt{dt}ms'+ '_' + str(saveID).zfill(3)
                X, a0_passive, T0_active = bezier.eval(['x_i','a0','T0'] @ ns, **lhs0)
                #X, U, stressff, Eff, sigma_a, lc = bezier.eval(['x_i','u_i','ef_i stress_ij ef_j', 'ef_i E_ij ef_j', 'σa','lc'] @ ns, **lhs0)
                #strain  = bezier.eval(ns.eval_ij('fiberbase_ti E_tk fiberbase_kj'), **lhs0)
                #stress  = bezier.eval(ns.eval_ij('stress_ij'), **lhs0)
                export.vtk(name, bezier.tri, X, a0=a0_passive, T0=T0_active)
            ##----------------------------------------------------------------------------------------------------#  


            ## Visualize hemodymics curve  Postprocessing----------------------------------------------------------#  
            if timestep in range(0,len(time))[::save_everynth] or timestep == len(time):  
              graph.overview(time, P=[PLV,PARTLV,PVENLV], V=[VLV,], Q=[QPERLV,QARTLV,QVENLV])
              graph.pressure_volume(V=[VLV,], P=[PLV,])

              graph.residual(resnorm)
              graph.total_volume(time,VTOT)
            ##-----------------------------------------------------------------------------------------------------#

                                                 
if __name__ == "__main__":
   cli.run(main)                