import numpy as np
from nutils import solver, export
import csv, treelog

def GPA_algorithm(multi, fibers, p_img, qdegree, scales=[1., 1., 1., 1.], nload=10, niter=20, targetiter=7, maxiter=15):
    
    # Unpack the displacement/passive part only
    res     = [scales[0]*multi.res["resu"], scales[1]*multi.res["reslx"], scales[2]*multi.res["resly"], scales[3]*multi.res["reslxy"]]
    targets = ['ulhs', 'lhslx', 'lhsly', 'lhslxy']
    cons    = multi.cons               # Constrains: dict( argument = cons_array )
    ns      = multi.ns                 # Namespace
    initial = multi.initial            # Initial vals: dict( 'ulhs' = array(...), .. )
    topo    = multi.topo
    
    # Set up the arguments dict for inflation
    arguments = {}
    arguments.update(fibers)  # Store fiber results
    arguments.update(initial) # Store initial value results (lc, lc0, ulhs, ulhs0, ...)
    arguments['t']       = np.array(0) # Set time = 0, makes sure no active stress is present
    arguments['ta']      = np.array(0) # Set time = 0, makes sure no active stress is present
    arguments['dt']      = np.array(1) # Set to an arbitrary value
    arguments["plvlhs"]  = np.array([0]) # Set lv pressure = 0, this is the simulation pressure, not p_image
    arguments["plv0lhs"] = np.array([0]) # Set lv0 pressure = 0
    arguments["lhslc"]   = arguments["lhslc0"]
    
    bezier  = topo.boundary.sample('bezier', 15)
    
    uprelhs  = np.zeros(ns.ubasis.shape[0])
    scale0   = 0
    Δscale   = 1./nload
    postiter = 0

    plist = []
    Vcavlist = []
    Vwalllist = []
    elist = []

    with treelog.iter.plain('GPA step', range(1000)) as GPAsteps:
      for istep in GPAsteps:

        scale = scale0 + Δscale  

        try:

          # Load scale
          scale = min(1.,scale)
          arguments["Pimg"]    = scale*p_img # Set P_img
          arguments["uprelhs"] = uprelhs

          treelog.user(f'GPA load scale : {scale}')

          # Inflated simulation      
          lhs, resnorm = solver.newton(targets, res, constrain=cons, arguments=arguments, linesearch=None).solve_withinfo(tol=1e-8, maxiter=maxiter)

          if np.isnan(resnorm.resnorm):
            treelog.user('GPA step resulted in NaN residual.')
            raise solver.SolverError('GPA step resulted in NaN residual.')
          else:
            treelog.user(f'GPA step converged in {resnorm.niter} iterations.') 

          # F projection
          ns.Δupre      = 'ubasis_ni ?Δuprelhs_n'
          ns.uprenew_i  = 'upre_i + Δupre_i'

          ns.ΔF_ij = 'δ_ij + Δupre_i,j'
          ns.detΔF = np.linalg.det(ns.ΔF)
          ns.ΔFinv = np.linalg.inv(ns.ΔF)

          sqr  = topo.integral('(δ_ij - uprenew_i,j - Finv_ij) (δ_ij - uprenew_i,j - Finv_ij) d:x' @ ns, degree=qdegree)
          sqr += topo.boundary['base_l'].integral('( lx uprenew_0 + ly uprenew_1 ) detΔF d:x' @ ns, degree=qdegree)
          sqr += topo.boundary['base_l'].integral('lxy ( ΔFinv_i0 uprenew_1,i - ΔFinv_j1 uprenew_0,j ) detΔF d:x' @ ns, degree=qdegree)
          sol  = solver.optimize(('Δuprelhs','lhslx','lhsly','lhslxy'), sqr, constrain={"Δuprelhs":cons["ulhs"]}, arguments=lhs, tol=1e-8, linesearch=None)

          if abs(scale-1.) < 1e-12:
              postiter += 1
              uprelhs = uprelhs + sol['Δuprelhs']
              scale0 = 1.
              Δscale = 0.
          else:
              uprelhs = uprelhs + sol['Δuprelhs']
              scale0 = scale
              Δscale *= (targetiter / resnorm.niter)
              Δscale = min(1./nload,Δscale)

          lhs.update({'Δuprelhs':sol['Δuprelhs']})

          ns.Fprenewinv_ij = 'δ_ij - uprenew_i,j'
          ns.Fprenew       = np.linalg.inv(ns.Fprenewinv)
          ns.detFprenew    = np.linalg.det(ns.Fprenew)

          Vcav  = topo.boundary['endo_l'].integrate('- ( ( ( x_j - uprenew_j - H_j ) n_i Fprenew_ij ) ( 1 / detFprenew ) / intC ) d:x' @ ns, degree=qdegree, arguments=lhs)
          Vwall, L2upre, L2Δupre = topo.integrate(['( 1 / detFprenew ) d:x', 'uprenew_i uprenew_i d:x', '(uprenew_i - upre_i) (uprenew_i - upre_i) d:x'] @ ns, degree=qdegree, arguments=lhs)

          error = np.sqrt(L2Δupre / L2upre)
          treelog.user(f'increment = {error}') 
          treelog.user(f'V cavity = {Vcav}') 
          treelog.user(f'V wall = {Vwall}')

          X, U, UPRE, UPRENEW, EF, EF0, STRESSFF = bezier.eval(['x_i','u_i', 'upre_i', 'uprenew_i', 'ef_i', 'ef0_i', 'stress_ij ef_i ef_j'] @ ns, **lhs)
          export.vtk('GPA', bezier.tri, X, U=U, UPRE=UPRE, UPRENEW=UPRENEW, EF=EF, EF0=EF0, STFF=STRESSFF)

          plist.append(scale*p_img)
          Vcavlist.append(Vcav)
          Vwalllist.append(Vwall)
          elist.append(error)

          with treelog.userfile('iterations.csv', 'w') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow(['iteration', 'pressure', 'Vcav', 'Vwall', 'incremenentL2'])
            for i, (a,b,c, d) in enumerate(zip(plist, Vcavlist, Vwalllist, elist)):
              writer.writerow([i, a, b, c, d])
    
        except (solver.SolverError, OverflowError, ValueError):
          
          Δscale *= (targetiter/maxiter)
          
        treelog.user(f'Load scale increment set to {Δscale}')
        if postiter == niter:
           break

    return {"Pimg" : np.array(0), "uprelhs":uprelhs}