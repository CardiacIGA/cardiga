from nutils import solver, function, cache, types
import numpy, treelog


def my_solver(target, res, cons, arguments_in, tol=1e-8, nstep=25, adaptive=False, elaborate_res=False): #topo, ns, qdegree=5, tasync_field=0
    ms=1e-3
    
    arguments = arguments_in.copy()
    treelog.info("Solver time-instance: {}".format(arguments['t']))
    
     
    t_2bsolved   = arguments['t'].copy()   # The time-instance we want to solve for
    dt_initial   = arguments['dt'].copy()  # The time-step we initially have
    t_previous   = t_2bsolved - dt_initial # We start from this time-instance and try to calc the next one += dt
    tactiv       = arguments['t'].copy() - arguments['ta'].copy()
    
    Converged   = False  
    adapt_level = 0
    
         
    while not Converged:
     #sol, resnorm  = my_newton_solver(target, res, constrain=cons, arguments=arguments, tol=tol, nstep=nstep, elaborate_res=elaborate_res)
     #linesearch = None #solver.NormBased(minscale=0.85, acceptscale=0.9, maxscale=10)
     #sol, resnorm  = solver.newton(target, res, constrain=cons, arguments=arguments, linesearch=linesearch).solve_withinfo(tol=tol, maxiter=nstep)
     #treelog.info(print(resnorm.resnorm)) 
     
#     sol, resnorm  = solver.newton(target, res, constrain=cons, arguments=arguments, linesearch=linesearch).solve_withinfo(tol=tol, maxiter=nstep)  
#     resnorm = [resnorm.resnorm]
     
#     linesearch=None
#     sol, resnorm  = solver.newton(target, res, constrain=cons, arguments=arguments, linesearch=linesearch).solve_withinfo(tol=tol, maxiter=nstep)  
#     resnorm = [resnorm.resnorm] 
#
#     
#     linesearch = None
#     resnorm = []
#     with treelog.context('iter {}', 0) as recontext:
#          for iiter, (sol, info) in enumerate(solver.newton(target, res, constrain=cons, arguments=arguments, linesearch=linesearch)):
#              if iiter == 0:
#                 resnorm0 = info.resnorm 
#              recontext(f'{iiter+1} ({100 * numpy.log(resnorm0 / max(info.resnorm, tol)) / numpy.log(resnorm0 / tol):.0f}%)')
#              resnorm.append(info.resnorm) 
#              if (info.resnorm < tol) or (iiter >= nstep):
#                 break


#     linesearch = None #solver.NormBased(minscale=0.85, acceptscale=0.9, maxscale=10)
#     sol, resnorm  = solver.newton(target, res, constrain=cons, arguments=arguments, linesearch=linesearch).solve_withinfo(tol=tol, maxiter=nstep)  
#     resnorm = [resnorm.resnorm] 
                  
     try:
      linesearch = None #solver.NormBased(minscale=0.85, acceptscale=0.9, maxscale=10)
      
      sol, resnorm  = solver.newton(target, res, constrain=cons, arguments=arguments, linesearch=linesearch).solve_withinfo(tol=tol, maxiter=nstep)  
      resnorm = [resnorm.resnorm] 
      
#      resnorm = []
#      for i, (sol, info) in enumerate(solver.newton(target, res, constrain=cons, arguments=arguments, linesearch=linesearch)):
#          resnorm.append(info.resnorm) 
#          if info.resnorm < tol or i > maxiter:
#            break
            
      
     except: # Something happend (crash or not able to find minimum)  
      resnorm = [1e5] # arbitrary high number  
     
       
     if (resnorm[-1] > tol or numpy.isnan(resnorm[-1])) and adaptive:
        t_prev = arguments['t'].copy() - arguments['dt'].copy()
        arguments['dt'] *= 0.5 # Decrease time-step 
        arguments['t']   = numpy.array(t_prev + arguments['dt']) # Update new time-instance we solve for, since time-step has decreased
        arguments['ta']  = numpy.array(t_prev + arguments['dt'] - tactiv) # Idem
        adapt_level     += 1 # We enter an adaptivity level
        increase         = False
        
        #arguments, cons, lhs0 = sarcomere_length(topo, ns, cons, arguments, arguments, qdegree=qdegree, tasync_field=tasync_field)
        
        treelog.info("Adaptivity level: {}".format(adapt_level))
        treelog.info("Solver adaptive time-instance (solving for): {}".format(arguments['t']))
        treelog.info("Solver adaptive time-step:                   {}".format(arguments['dt']))
        
     elif numpy.isclose( arguments['t'], t_2bsolved ): # we reachd to time-instance that we wanted (can be if in adaptive time stepping)
        Converged    = True    
        lhs = sol.copy()
        
        #assert numpy.isclose( arguments['dt'], dt_initial ), "Final time-step deviates from initial"
        #assert , "Final time-instance deviates from time-instance to be solved"
                    
     elif adapt_level > 0:
     
         
        lhs0 = sol.copy()
        
        t_prev = arguments['t'].copy()
        if increase:
          arguments['dt'] /= 0.5 # Increase time-step, this is done at the end. We could do it at the beginning, but this makes the time a bit more complicaed + we expect the next-iteration to also have some trouble, so best to use the same dt a second time
          adapt_level -= 1 # Remove an adaptivity level
          increase = False
        else:
          increase = True
                    
        # TODO Make this more general
        arguments['t']   = numpy.array(t_prev + arguments['dt']) # Update new time-instance we solve for, since time-step has decreased
        arguments['ta']  = numpy.array(t_prev + arguments['dt'] - tactiv) # Idem
        
        arguments['plv0lhs']   = lhs0['plvlhs']   
        arguments['partS0lhs'] = lhs0['partSlhs']
        arguments['ulhs0']     = lhs0['ulhs']
        arguments['lhslc0']    = lhs0['lhslc']  
        arguments['lhslc']     = lhs0['lhslc']
        # Initial gues value
        arguments['plvlhs']    = lhs0['plvlhs']   
        arguments['partSlhs']  = lhs0['partSlhs']
        arguments['ulhs']      = lhs0['ulhs']
        
        
        # Set Lagrange multipliers initial guess
        if "lhslx" in lhs0:
          arguments['lhslx']  = lhs0['lhslx']
        if "lhsly" in lhs0:
          arguments['lhsly']  = lhs0['lhsly']
        if "lhslxy" in lhs0:
          arguments['lhslxy'] = lhs0['lhslxy']
        
        
        
        if 'prvlhs' in lhs0: # It is a biventricle simulation
          arguments['prv0lhs']   = lhs0['prvlhs']   
          arguments['partP0lhs'] = lhs0['partPlhs']   
          arguments['pvenS0lhs'] = lhs0['pvenSlhs']
  
          # Initial gues value
          arguments['prvlhs']    = lhs0['prvlhs']   
          arguments['partPlhs']  = lhs0['partPlhs']   
          arguments['pvenSlhs']  = lhs0['pvenSlhs']      
        
        #arguments, cons, lhs0 = sarcomere_length(topo, ns, cons, lhs0, arguments, qdegree=qdegree, tasync_field=tasync_field)
        
#        if (arguments['ta'] < 0 or numpy.isclose(arguments['ta'],0,ms*1e-3)):
#          arguments.pop('lhslc', None) # Remove from arguments before minimizing this target
#          sqr_lc  = topo.integral('( lc - ls )^2 d:x' @ multi.ns, degree=4)                
#          init_lc = solver.optimize('lhslc', sqr_lc, arguments=arguments, tol=1e-6, droptol=1e-15)
#          cons['lhslc']        = init_lc
#          arguments['lhslc0']  = init_lc
#          lhs['lhslc']         = init_lc # Simply update the solution, this will not affect other results, only used for post-processing step (during passive filling the lhslc = contrained
#          lcinit = 1
#        elif arguments['ta'] > 0 and lcinit == 1:  
#          treelog.info('Entered lhslc pop')
#          lcinit = 0
#          #cons.pop('lhslc0', None)
#          cons.pop('lhslc', None)  
                          
        # update cons
#        if (arguments['ta'] < 0 or numpy.isclose(arguments['ta'],0,ms*1e-3)):
#          arguments.pop('lhslc', None) # Remove from arguments before minimizing this target
#          sqr_lc  = topo.integral('( lc - ls )^2 d:x' @ multi.ns, degree=4)                
#          init_lc = solver.optimize('lhslc', sqr_lc, arguments=arguments, tol=1e-6, droptol=1e-15)
#          cons['lhslc']        = init_lc
#          arguments['lhslc0']  = init_lc
#          lhs['lhslc']         = init_lc # Simply update the solution, this will not affect other results, only used for post-processing step (during passive filling the lhslc = contrained
#          lcinit = 1
#        elif ts-tactiv > 0 and lcinit == 1:  
#          treelog.info('Entered lhslc pop')
#          lcinit = 0
#          cons.pop('lhslc', None)  
                  
        
        treelog.info("Adaptivity level: {}".format(adapt_level))
        treelog.info("Solver adaptive time-instance (solving for): {}".format(arguments['t']))
        treelog.info("Solver adaptive time-step:                   {}".format(arguments['dt']))
        #increase     = True
        #adapt_level -= 1 # Remove an adaptivity level
        
        
        
     else: # This should not occur, else we have a problem
        Converged    = True    
        lhs = sol.copy()
        
        #assert numpy.isclose( arguments['dt'], dt_initial ), "Final time-step deviates from initial"
        assert numpy.isclose( arguments['t'], t_2bsolved ), "Final time-instance deviates from time-instance to be solved"
                  
    return sol, resnorm                     

#def sarcomere_length(topo, ns, cons, lhs, arguments, qdegree=5, tconstant_field=1, tasync_field=0):
#    
#    # During filling, the sarcomere contractile length (lc) should not be computed but kept constant to ls  
#    tmask = numpy.less_equal( arguments["ta"]*tconstant_field - tasync_field, 0)
#    if tmask.any(): # If any of the tmask values if True (i.e. below the threshold)
#      treelog.info("Sarcomere length kept constant somewhere in domain")
#      arguments.pop('lhslc', None) # Remove from arguments before minimizing this target 
#      
#      sqr_lc  = topo.integral('( lc - ls )^2 d:x' @ ns, degree=qdegree)                
#      init_lc = solver.optimize('lhslc', sqr_lc, arguments=arguments, tol=1e-6, droptol=1e-15)  
#      
#      # lhs is read only, so we have to use the copied lhs0
#      lc_prev = lhs['lhslc'].copy()# Simply update the solution, this will not affect other results, only used for post-processing step (during passive filling the lhslc = contrained
#      lc_prev[tmask]      = init_lc[tmask]
#      lhs['lhslc']        = lc_prev
#      arguments['lhslc0'] = lc_prev # Add to arguments as previous time-step value
#      
#      # initialize the cons array
#      lccons = numpy.full(len(init_lc), numpy.nan)
#      lccons[tmask] = init_lc[tmask]
#      cons['lhslc'] = lccons # Add to constrains
#      
#      treelog.info(cons['lhslc'])
#      #treelog.info(tmask)
#      #constrain_lc = True
#    elif not tmask.any():  
#      treelog.info('Entered lhslc pop')
#      cons.pop('lhslc', None) 
#      #constrain_lc = False
#    return arguments, cons, lhs





#@solver.single_or_multiple
#@types.apply_annotations
#@cache.function
#def my_newton_solver(target, residual:solver.integraltuple, *, constrain:solver.arrayordict=None, lhs0:types.arraydata=None, arguments:solver.argdict={}, tol=1e-6, nstep=20, elaborate_res=False,**kwargs):
#  '''solve linear problem
#
#  Parameters
#  ----------
#  target : :class:`str`
#      Name of the target: a :class:`nutils.function.Argument` in ``residual``.
#  residual : :class:`nutils.evaluable.AsEvaluableArray`
#      Residual integral, depends on ``target``
#  constrain : :class:`numpy.ndarray` with dtype :class:`float`
#      Defines the fixed entries of the coefficient vector
#  arguments : :class:`collections.abc.Mapping`
#      Defines the values for :class:`nutils.function.Argument` objects in
#      `residual`.  The ``target`` should not be present in ``arguments``.
#      Optional.
#
#  Returns
#  -------
#  :class:`numpy.ndarray`
#      Array of ``target`` values for which ``residual == 0``'''
#
#  solveargs = solver._strip(kwargs, 'lin')
#  if kwargs:
#    raise TypeError('unexpected keyword arguments: {}'.format(', '.join(kwargs)))
#
#  resnorm = []
#  ## Initialize lhs and constrain (True/False) 
#  lhs0, constrain = solver._parse_lhs_cons(lhs0, constrain, target, solver._argobjs(residual), arguments)
#  jacobian = solver._derivative(residual, target)
#  
#  lhs, vlhs = solver._redict(lhs0, target) # This function couples lhs and vlhs (pointers)
#  mask, vmask = solver._invert(constrain, target)
#  indx = numpy.cumsum([numpy.count_nonzero(imask) for imask in mask])
#  #zeropivot = False
#  for i in range(0,nstep):  
#      
#
#      res, jac = solver._integrate_blocks(residual, jacobian, arguments=lhs, mask=mask)
#      try:
#        vlhs[vmask] -= jac.solve(res, **solveargs) # Solve linear system
#      except: # If we have zero-pivot
#        treelog.info('Zero-Pivot Error detected!')
#
#          
#         
#    
#      L2res    = numpy.linalg.norm(res)
#      resnorm += [L2res]
#      treelog.info('{} Newton iteration, residual: {}'.format(i,L2res))
#
#      if elaborate_res: # Does not work yet, the length of res is different from residual, because res is masked (shows only the solved dofs)
#        resn  = numpy.split(res,indx) # list of arrays that correspond to te target-residual arrays
#        space = max([len(itarg) for itarg in target])
#        for targ_name, normarr in zip(target,resn):
#          targ_name = targ_name.ljust(space, ' ')
#          if len(normarr) == 0:
#            treelog.info(f'{targ_name} residual: Constrained')
#          else:
#            treelog.info(f'{targ_name} residual: {numpy.linalg.norm(normarr)}')
#            
#
##      if L2res > 10: # If we have zero-pivot
##         break
#         
#      if L2res < tol:
#        break
#      elif numpy.isnan(L2res):
#        treelog.info('NaN detected, iteration terminated')
#        break  
#        
#        
##  if zeropivot:
##     return 1, [numpy.nan,] 
##  else:     
#  return lhs, [resnorm,] if len(resnorm)==1 else resnorm
#  
  
#@solver.single_or_multiple
#@types.apply_annotations
#@cache.function
#def my_newton_solver(target, residual, *, constrain=None, lhs0:types.arraydata=None, arguments={}, tol=1e-6, nstep=20, elaborate_res=False,**kwargs):
#  '''solve linear problem
#
#  Parameters
#  ----------
#  target : :class:`str`
#      Name of the target: a :class:`nutils.function.Argument` in ``residual``.
#  residual : :class:`nutils.evaluable.AsEvaluableArray`
#      Residual integral, depends on ``target``
#  constrain : :class:`numpy.ndarray` with dtype :class:`float`
#      Defines the fixed entries of the coefficient vector
#  arguments : :class:`collections.abc.Mapping`
#      Defines the values for :class:`nutils.function.Argument` objects in
#      `residual`.  The ``target`` should not be present in ``arguments``.
#      Optional.
#
#  Returns
#  -------
#  :class:`numpy.ndarray`
#      Array of ``target`` values for which ``residual == 0``'''
#
#  solveargs = solver._strip(kwargs, 'lin')
#  target, residual = _target_helper(target, residual)
#  if kwargs:
#    raise TypeError('unexpected keyword arguments: {}'.format(', '.join(kwargs)))
#
#  resnorm = []
#  
#  ## Add these when using new nutils version
#  constraints    = types.frozendict((k, types.arraydata(v)) for k, v in (constrain or {}).items())
#  argumentsfd    = types.frozendict((k, types.arraydata(v)) for k, v in (arguments or {}).items())
#  solvearguments = types.frozendict(solveargs)
#  
#  ## Initialize lhs and constrain (True/False) 
#  lhs0, constrain = solver._parse_lhs_cons(lhs0, constraints, target, solver._argobjs(residual), argumentsfd)
#  jacobian = solver._derivative(residual, target)
#  
#  dtype = solver._determine_dtype(target, residual, argumentsfd, constraints)
#  lhs, vlhs = solver._redict(lhs0, target, dtype) # This function couples lhs and vlhs (pointers)
#  mask, vmask = solver._invert(constraints, target)
#  indx = numpy.cumsum([numpy.count_nonzero(imask) for imask in mask])
#  #zeropivot = False
#  for i in range(0,nstep):  
#      
#
#      res, jac = solver._integrate_blocks(residual, jacobian, arguments=lhs, mask=mask)
#      try:
#        vlhs[vmask] -= jac.solve(res, **solvearguments) # Solve linear system
#      except: # If we have zero-pivot
#        treelog.info('Zero-Pivot Error detected!')
#
#          
#         
#    
#      L2res    = numpy.linalg.norm(res)
#      resnorm += [L2res]
#      treelog.info('{} Newton iteration, residual: {}'.format(i,L2res))
#
#      if elaborate_res: # Does not work yet, the length of res is different from residual, because res is masked (shows only the solved dofs)
#        resn  = numpy.split(res,indx) # list of arrays that correspond to te target-residual arrays
#        space = max([len(itarg) for itarg in target])
#        for targ_name, normarr in zip(target,resn):
#          targ_name = targ_name.ljust(space, ' ')
#          if len(normarr) == 0:
#            treelog.info(f'{targ_name} residual: Constrained')
#          else:
#            treelog.info(f'{targ_name} residual: {numpy.linalg.norm(normarr)}')
#            
#
##      if L2res > 10: # If we have zero-pivot
##         break
#         
#      if L2res < tol:
#        break
#      elif numpy.isnan(L2res):
#        treelog.info('NaN detected, iteration terminated')
#        break  
#        
#        
##  if zeropivot:
##     return 1, [numpy.nan,] 
##  else:     
#  return lhs, [resnorm,] if len(resnorm)==1 else resnorm
#  
  
  
#def solve_linear(target, residual, *, constrain = None, lhs0: types.arraydata = None, arguments = {}, **kwargs):
#    '''solve linear problem
#
#    Parameters
#    ----------
#    target : :class:`str`
#        Name of the target: a :class:`nutils.function.Argument` in ``residual``.
#    residual : :class:`nutils.evaluable.AsEvaluableArray`
#        Residual integral, depends on ``target``
#    constrain : :class:`numpy.ndarray` with dtype :class:`float`
#        Defines the fixed entries of the coefficient vector
#    arguments : :class:`collections.abc.Mapping`
#        Defines the values for :class:`nutils.function.Argument` objects in
#        `residual`.  The ``target`` should not be present in ``arguments``.
#        Optional.
#
#    Returns
#    -------
#    :class:`numpy.ndarray`
#        Array of ``target`` values for which ``residual == 0``'''
#
#    if isinstance(target, str) and ',' not in target and ':' not in target:
#        return solve_linear([target], [residual], constrain={} if constrain is None else {target: constrain},
#            lhs0=lhs0, arguments=arguments if lhs0 is None else {**arguments, target: lhs0}, **kwargs)[target]
#    if lhs0 is not None:
#        raise ValueError('lhs0 argument is invalid for a non-string target; define the initial guess via arguments instead')
#    target, residual = _target_helper(target, residual)
#    solveargs = _strip(kwargs, 'lin')
#    if kwargs:
#        raise TypeError('unexpected keyword arguments: {}'.format(', '.join(kwargs)))
#    return _solve_linear(target, residual,
#        types.frozendict((k, types.arraydata(v)) for k, v in (constrain or {}).items()),
#        types.frozendict((k, types.arraydata(v)) for k, v in (arguments or {}).items()),
#        types.frozendict(solveargs))
#
#
#@cache.function
#def _solve_linear(target, residual: tuple, constraints: dict, arguments: dict, solveargs: dict):
#    arguments, constraints = _parse_lhs_cons(constraints, target, _argobjs(residual), arguments)
#    jacobians = _derivative(residual, target)
#    if not set(target).isdisjoint(_argobjs(jacobians)):
#        raise SolverError('problem is not linear')
#    dtype = _determine_dtype(target, residual, arguments, constraints)
#    lhs, vlhs = _redict(arguments, target, dtype)
#    mask, vmask = _invert(constraints, target)
#    res, jac = _integrate_blocks(residual, jacobians, arguments=lhs, mask=mask)
#    vlhs[vmask] -= jac.solve(res, **solveargs)
#    return lhs  