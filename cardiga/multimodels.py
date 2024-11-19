# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 14:32:10 2021

@author: s146407
"""

import numpy
from nutils import mesh, export, function, solver, matrix, topology
import pickle
import treelog

import matplotlib.pyplot as plt
import matplotlib.collections

## Import relevant files
from .passive import Passive_models
from .active import Active_models, async_activation
from .fiberfield import Fiber_models
from .lumpedparameter import Lumped_models
from .infarct import Infarct
from .geometry import InterfaceElem

# def save_pickle(lhs,filename):
#       with open(filename+'.pickle', 'wb') as f:
#           pickle.dump(lhs,f)
#       return  
# TODO Check method of the Models() class      
class Models():
    
    def __init__(self, topo, geom, boundaries={}):
        # Assign to self
        self.topo = topo
        self.geom = geom
        self.init_ns()
        self.ftype      = False
        self.boundaries = self.init_bound(boundaries) # Dictionary containing specific names, if a name is missing, it will neglect it
        return
    
    def __add__(self,other): # This function adds the different objects and consequently constructs the matrix!
        # 1. Chain the namespace and basis functions
        # self.ns = self.pmodel.ns_model(self.m_type , ns)
        # 2. add the residuals
        # res = self.pmodel.res + self.amodel.res + self.lmodel.res # Add all residuals
        
        res = 1
        return res
    def __str__(self,):
        Str = 'The follow'
        return 'testing'
        
    ## Initialize functions
    def init_ns(self):
        self.ns    = function.Namespace()
        self.ns.x  = self.geom
        return
    
    def init_bound(self,boundaries):
        return boundaries
    
    
    ## Add or append functions
    # def add2chain(self): # Add basis to chain
    #     return
    # def add2ns(self,): # Add to namespace
    #     return
    
    ## Pre-processing: The fiber field, in case this step is skipped an orthotrpoic passive material is used 
    #                  + an x-direction oriented fiber for active tension
    def fiber(self,m_type,g_type='lv', btype='th-spline', bdegree=3, quad_degree=None, fiber_dir=None, angles_input={}): # m_type: 'default' or 'rossi', g_type: 'lv' or 'bv'
        self.fmodel = Fiber_models(self.ns, self.topo, self.geom, angles_input)
        self.g_type = g_type
        # used for printing and checking the active/passive stress component
        self.ftype    = True
        self.fdefault = True if m_type == 'default' else False # True because the default fiber orientaion is used 

        lhs, const       = self.fmodel.construct(m_type, g_type, fiber_dir, btype, bdegree, quad_degree)
        self.fiberangles = const

        return lhs   

    
    ## Model functions
    def passive(self, m_type, btype='spline', bdegree=3, quad_degree=None, const_input={}, prestress={}, surpressJump=False):
        if self.ftype == False:
            treelog.info("No fiber field specified, switching to isotropic behavior")
        elif self.ftype == True and self.fdefault == True:
            treelog.info("Default fiber field specified, switching to isotropic behavior")
        
        
        # Initialize the following clas. Required when calling interfaces (between patches), especially for hierarchical topologies
        if surpressJump:
          self.InterfaceElems = InterfaceElem(self.ns, self.topo)
          classes = {"Ielem" : self.InterfaceElems}
        else:
          self.InterfaceElems = None
          classes = {}
          
        ## Construct the passive model  
        self.pmodel = Passive_models(self.ns, self.topo, self.geom, self.g_type, m_type, self.boundaries) 
        res, targets, constants = self.pmodel.construct(const_input, btype, bdegree, quad_degree, 
                                                        surpressJump   = surpressJump, 
                                                        interfaceClass = self.InterfaceElems,
                                                        prestress      = prestress) # get residual, updated namespace and argument + constants used in model
        
        ## Extract useful dicts
        cons     = self.pmodel.get_cons() # Get constrain dict
        initial0 = self.pmodel.get_initial()
        if self.ftype:
            constants.update(self.fiberangles)
    
        return model(self.topo, self.ns, res, constants, targets, cons=cons, initial=initial0, classes=classes) # Return as a model class
    
    def active(self, m_type, btype='spline', bdegree=2, quad_degree=None, const_input={}, async_activ=None):
        if self.ftype == False:
            raise Exception('No fiber field has been specified!')
        elif self.ftype == True and self.fdefault == True:
            treelog.info("Default fiber field used, not recommended")
            
        initialTasync = {}
        if type(async_activ)==str:
           initialTasync = self.async_act(async_activ, quad_degree=quad_degree, return_field=True, btype=btype, bdegree=bdegree) # Add tasync to namespace (in background), the loaded file is initialized at 0
            
        self.amodel = Active_models(self.ns,self.topo)
        res, targ, const = self.amodel.construct(m_type, btype, bdegree, quad_degree, const_input=const_input)
        initial0        = self.amodel.get_initial()
        initial0.update(initialTasync)
        if self.ftype != False:
            const.update(self.fiberangles)
        return model(self.topo, self.ns, res, const, targ, initial=initial0)


    def lumpedparameter(self, m_type, p_type, inner_boundary, quad_degree=None, const_input={}): # 1, 2, or 3 element windkessel model + nr of cavities
        self.lmodel = Lumped_models(self.ns,self.topo) #<< -- Do I want to store this lmodel!?
        ##<<<------- Add a check to see whether material law is defined (raise error if this is required, i.e., when m_type='lv' or 'bv')
        ##<<<------- Add a check to see whether active part is defined (raise error if no 'dt' is specified in the constants and m_type='0D', otherwise always yield error)
        res, targ, targ0, const, init0 = self.lmodel.construct(m_type, p_type, inner_boundary, const_input=const_input, quad_degree=quad_degree)
        # targ0 contains the targets, of the previous time-step, useful to look-up if the number of variables increase...
        
        # Error if LV and no material law
        # Error if no active model at all (or specify time-step variables)
        return model(self.topo, self.ns, res, {'lumped-parameter':const}, targ, initial=init0)

    def async_act(self, filename, quad_degree=None, return_field=False, btype="spline", bdegree=3):
        quad_degree = 5 if quad_degree == None else quad_degree
        
        gauss_sample = self.topo.sample("gauss", quad_degree)
        tact, Xgauss = async_activation(filename, gauss_sample, return_coords=True)
        
        # Check if the file has same number of gauss points as used in the simulation (early error/warning)
        assert len(gauss_sample.eval(self.geom)) == len(Xgauss), "Mismatch between provided async. activation times gauss points and the evaluated gauss degree (incorrect topology/geometry or specified gaussdeg)."
        #self.ns.tasync = tact # Add to namespace
        
        if return_field:
           treelog.info("Projecting asynchronous activation field")
           self.ns.aSyncbasis = self.topo.basis(btype, degree=bdegree)
           self.ns.TaSync = "aSyncbasis_n ?lhsTasync_n"
           self.ns.tasync = self.ns.TaSync
           # solve for T async
           lhsTasync = self.topo.project(tact, self.ns.aSyncbasis, self.geom, degree=quad_degree)
           # Also solve for constant value field of 1 (needed when comparing to different time-instances)
           #lhsTones = self.topo.project(gauss_sample.asfunction(numpy.ones(len(Xgauss))), self.ns.aSyncbasis, self.geom, degree=quad_degree)
           #treelog.info(lhsTones)
           return {"lhsTasync" : lhsTasync}#, "lhsTones" : lhsTones}
           
    # def infarct_input(self, gausspoints, gauss_sample, btype='th-spline', bdegree=3, gdegree=2, ptype='projection', limit=(400,4000) ):
    
    #     self.ns.Infarctsampled = gauss_sample.asfunction(gausspoints) 
    #     self.ns.a0basis  = self.topo.basis(btype, degree=bdegree)
    
    #     if ptype == 'convolute':
    #        self.a0 = 'a0basis_n ?lhsa0_n'
    #        lhsa0 = self.topo.project(self.ns.sampled, self.ns.a0basis, self.ns.x, degree=gdegree, ptype='convolute')
    #     else:
    #        ## Projection
    #        min_arg = '( a0T - {} )^2 d:x'.format(limit[0])
    #        self.ns.a0T  = 'a0basis_n ?lhsa0_n'
    #        sqra0C  = self.topo.boundary['base_l'].integral(min_arg @ self.ns, degree=8)                
    #        consa0  = solver.optimize('lhsa0', sqra0C, tol=1e-6, droptol=1e-15) 
    #        sqra0   = self.topo.integral('( Infarctsampled - a0T )^2 d:x' @ self.ns, degree=gdegree)                
    #        lhsa0   = solver.optimize('lhsa0', sqra0, tol=1e-6, droptol=1e-15, constrain=consa0)
    #        self.a0 = function.min( function.max(self.ns.a0T, limit[0]), limit[1] )
    #     return dict(lhsa0=lhsa0)

        
    ## Solve system of equations
    def solve(self,):
        
        return
    
    
    
class model():
    
    def __init__(self, topo, ns, res, constants, arguments, cons={}, initial={}, variables=None, functions={}, classes={}):
        self.topo = topo
        self.res  = res
        self.ns   = ns
        self.cons = cons
        if variables == None:
            self.variables = self.get_variables(constants)
        else:
            self.variables = variables
        self.arguments = arguments
        self.constants = constants
        self.initial   = initial
        self.__Functions = functions
        self.__Classes   = classes
        self.__set_attribute(functions)
        self.__set_attribute(classes)
        return
    
    def __add__(self, other):
        #ns = self.ns
        #ns1 = other.ns
        #ns._attributes.update( ns1._attributes)
        
        for i in self.arguments:
            if i in other.arguments:
                treelog.info("Identical arguments, '{}', are given in separate models!".format(i))
                
        for i in other.res.copy():
            if i in self.res:
                self.res[i] += other.res[i]
            else:
                self.res[i]  = other.res[i]
        #res       = self.res + other.res # automatically appends residual to existing list        
        # constants = {}
        # constants.update(self.constants)
        # constants.update(other.constants)
        # cons = {}
        # cons.update(self.cons)
        # cons.update(other.cons)
        
        
        arguments = self.arguments + other.arguments #self.arguments.append(other.arguments) 
        self.constants.update(other.constants)
        self.cons.update(other.cons)
        self.initial.update(other.initial)
        self.variables.update(other.variables)
        
        functions = {}
        functions.update(self.__Functions)
        functions.update(other.__Functions)
        
        classes = {}
        classes.update(self.__Classes)
        classes.update(other.__Classes)
        
        return model(self.topo, self.ns,self.res,self.constants,arguments, cons=self.cons, initial=self.initial, variables=self.variables, functions=functions, classes=classes)
    
    def __str__(self,):
        # Print the type of model (mechanical, active, etc.) and the constants
        #print('NotImplemented')
        Lines = 'The following models and corresponding constants are used:\n'
        Lines += '_____________________________________________________\n'
        for i, im in enumerate(self.constants):        
            Lines += "'{}' model constants\n".format(im)
            Lines += '-----------------------------------------------------\n'
            #Lines += 'Constants:--------------------\n'
            for c in self.constants[im]:
                Lines += '{:<8} = {}\n'.format(c,self.constants[im][c])
            Lines += '_____________________________________________________\n'
        return Lines
    
    def __set_attribute(self, Object):
        for objectName, object in Object.items():
                setattr(self, objectName, object)
        return
    
    def check(self,): # Run some checks to validate model input
        # 1. 
        assert len(self.res) == len(self.arguments), 'Number of residuals is not same as arguments'
        return
    
    
    
    
    def separate_constants(self, constants): # Split the constant array into real constants and nutils variables
        varlist = []
        for i in constants:
            const = constants[i]
            for c in const: 
                val = const[c]
                if type(val) == str:
                    varlist += self.read_str(val)
        variables = { iv : None for iv in varlist}
        variables.pop('', None)
        return variables
    
    def read_str(self,var): # Function that reads the variables '?c n_i' or '< ?c, ?d >_i', and returns 'c' (the name)
        assert type(var)==str, 'Variable is not given as string'
        for i, iv in enumerate(var):
            if iv == '<':
                icom1 = numpy.char.find(var,',',start=0)        # index of first comma
                icom2 = numpy.char.find(var,',',start=icom1+1)  # index of second comma (if there is none, -1 ir returned)
                
                igr = numpy.char.find(var,'>',start= icom2 if icom2!=-1 else icom1) # index of >, marking end of vector
                if icom2==-1:
                    c = [ self.get_const(var[i:icom1]) , self.get_const(var[icom1:icom2]) , self.get_const(var[icom2:igr]) ]
                else:
                    c = [ self.get_const(var[i:icom1]), self.get_const(var[icom1:igr]) ]
                break
            elif iv=='?':
                c = [ self.get_const(var) ] 
                break    
        return c

    def get_const(self,var):
        idx_start = numpy.char.find(var,'?')
        if idx_start==-1:
            return ''
        else:
            idx_end = numpy.char.find(var,' ',start=idx_start) 
            if idx_end== -1:
                return var[(idx_start+1):]
            else:
                return var[(idx_start+1):idx_end]
            
    def get_variables(self,constants):
        return self.separate_constants(constants)