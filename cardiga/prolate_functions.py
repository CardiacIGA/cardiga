# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 10:27:55 2021

@author: s146407
"""

## Prolate coordinates & Normalized wall coordinates = Transmural/Longitudinal (u, v)

import numpy
_ = numpy.newaxis

class Ellips_Functs:
    def cartesian_to_prolate(Xcart,C):
        """
        Expression for ellipsoidal coordinates (alternative definition)
        https://en.wikipedia.org/wiki/Prolate_spheroidal_coordinates
    
          Parameters
          ----------
            Xcart : Array containing Cartesian coordinates Xcart = array([ x, y, z ])
            C     : Float defined as the distance from the origin to the shared focus points
    
          Returns
          -------
            Array containing Prolate spheroidal coordinates Xprol = array([ξ, θ, φ])
        """
        if len(Xcart.shape) == 1: 
          sqr_add = numpy.linalg.norm( Xcart + numpy.array([ 0, 0, C ]) )
          sqr_min = numpy.linalg.norm( Xcart - numpy.array([ 0, 0, C ]) )
          sigma = ( sqr_add + sqr_min )/(2*C)
          tau   = ( sqr_add - sqr_min )/(2*C)
          return numpy.array([numpy.arccosh( sigma ), numpy.arccos( tau ), numpy.arctan2( Xcart[1], Xcart[0] )])
        else:
          sqr_add = numpy.linalg.norm( Xcart + numpy.array([ 0, 0, C ]) , axis=1)
          sqr_min = numpy.linalg.norm( Xcart - numpy.array([ 0, 0, C ]) , axis=1)
          sigma = ( sqr_add + sqr_min )/(2*C)
          tau   = ( sqr_add - sqr_min )/(2*C)
          return numpy.concatenate([numpy.arccosh( sigma )[:,_], numpy.arccos( tau )[:,_], numpy.arctan2( Xcart[:,1], Xcart[:,0] )[:,_]], axis=1)
        
        
    
    
    def prolate_to_cartesian(Xprol,C):
        """
          Parameters
          ----------
            Xprol : Array containing prolate spheroidal coordinates Xprol = array([ξ, θ, φ])
            C     : Float defined as the distance from the origin to the shared focus points
    
          Returns
          -------
            Array contianing Cartesian coordinates Xcart = array([ x, y, z ])
        """
        if len(Xprol.shape) == 1:
          return numpy.array([C*numpy.sinh(Xprol[0])*numpy.sin(Xprol[1])*numpy.cos(Xprol[2]),
                              C*numpy.sinh(Xprol[0])*numpy.sin(Xprol[1])*numpy.sin(Xprol[2]),
                              C*numpy.cosh(Xprol[0])*numpy.cos(Xprol[1])                    ])
        else:
          return numpy.array([C*numpy.sinh(Xprol[:,0])*numpy.sin(Xprol[:,1])*numpy.cos(Xprol[:,2]),
                              C*numpy.sinh(Xprol[:,0])*numpy.sin(Xprol[:,1])*numpy.sin(Xprol[:,2]),
                              C*numpy.cosh(Xprol[:,0])*numpy.cos(Xprol[:,1])                    ]).T
    
    
    def prolate_to_normalizedwall(ξ: float, θ: float, z:float , vbound,ubound,zmax, ξθbound, integral=False):
        """
          Parameters array([ξ, θ, φ])
          ---------- 
            ξ : Prolate spheroidal coordinate 
            θ : Prolate spheroidal coordinate 
            z : Cartesian z-coordinate
            
            zmax   : Float value of the maximum Cartesian value of z.
            vbound : List containing the min and max values of the transmural coordinate v
            ubound : List containing the min and max values of the longitudinal coordinate u
                
            ξθbound: List containing either:
                          - The integral boundaries in the following order: [ξstart, ξend], given that integral=False
                          - The start boundary of the ξ-integral and the result of the fixed integrals: [ξstart, ξint, θint], given that integral=True
            integral: Boolean which specifies the input of ξθbound   
            
          Returns
          -------
            Normalized wall coordinates u, v corresponding the prolate spheroidal coordinate given in the input
        """
        ξstart    = ξθbound[0]
        ξend      = ξθbound[1]
        v = (vbound[1] - vbound[0]) *Ellips_Functs.integrate_ξ_fraction(ξstart, ξend, ξ, θ)  + vbound[0]   
        if θ > 0.5*numpy.pi:
            u = ubound[0]*( 1 - Ellips_Functs.integrate_θ_fraction(numpy.pi, 0.5*numpy.pi, θ, ξ)  ) 
        else:
            u = ubound[1]*z/zmax      
        return u, v
        
        # if integral: # This option is preferred when you try to compute multiple u,v values (reduces comp. speed)
        #     ξintegral = ξθbound[1]
        #     θintegral = ξθbound[2]
        # elif integral:
        #     ξintegral = integrate_ξ(  ξstart,   ξθbound[1], θ)
        #     θintegral = integrate_θ(numpy.pi, 0.5*numpy.pi, ξ)
        
        # v = (vbound[1] - vbound[0]) * ( integrate_ξ( ξstart, ξ, θ ) / integrate_ξ( ξstart, ξend, θ )  )  + vbound[0]   
        # if θ > 0.5*numpy.pi:
        #     u = ubound[0]*( 1 - integrate_θ( numpy.pi, θ, ξ ) / integrate_θ( numpy.pi, numpy.pi/2, ξ ) ) 
        # else:
        #     u = ubound[1]*z/zmax  
        
    
    
    def integrate_ξ_fraction(ξstart, ξend, ξ, θ): # Integrate over Xi, given the integral boundaries Xi_start, Xi_end
        """
        Solve the ξ-integral fraction
          Parameters
          ---------- 
            ξstart : Starting boundary integral
            ξend   : Ending boundary integral
            ξ      : Prolate spheroidal coordinate
            θ      : Prolate spheroidal coordinate 
    
          Returns
          -------
            Result of the fraction: integral( start - x ) / integral( start - end )
        """
        nr = int( ( 1 - (ξ / ξend) ) * 2001 )
        ξ1 = numpy.linspace(ξstart,    ξ, max(2001 - nr, 0) )
        ξ2 = numpy.linspace(     ξ, ξend,  max(nr, 0) )
        int1 = numpy.trapz( numpy.sqrt( numpy.sinh( ξ1 )**2 + numpy.sin( θ ) ), ξ1)  # trapz(y,x)
        int2 = numpy.trapz( numpy.sqrt( numpy.sinh( ξ2 )**2 + numpy.sin( θ ) ), ξ2)  # trapz(y,x)
        return int1 / (int1 + int2)
    
    
    def integrate_θ_fraction(θstart, θend, θ, ξ): # Integrate over Xi, given the integral boundaries Xi_start, Xi_end
        """
        Solve the θ-integral fraction
          Parameters
          ---------- 
            θstart : Starting boundary integral
            θend   : Ending boundary integral
            θ      : Prolate spheroidal coordinate
            ξ      : Prolate spheroidal coordinate
            
          Returns
          -------
            Result of the fraction: integral( start - x ) / integral( start - end )
        """
        nr = int( ( (θ / θstart) ) * 2001 )
        θ1 = numpy.linspace(θstart,    θ, max(2001 - nr, 0) )
        θ2 = numpy.linspace(     θ, θend, max(nr, 0) )
        int1 = numpy.trapz( numpy.sqrt( numpy.sinh( ξ )**2 + numpy.sin( θ1 ) ), θ1)  # trapz(y,x)
        int2 = numpy.trapz( numpy.sqrt( numpy.sinh( ξ )**2 + numpy.sin( θ2 ) ), θ2)  # trapz(y,x)
        return int1 / (int1 + int2)
    
    
    
    
    ## Seprate integrals (take more time)
    def integrate_ξ(ξstart, ξend, θ): # Integrate over Xi, given the integral boundaries Xi_start, Xi_end
        """
        Solve the ξ-integral
          Parameters
          ---------- 
            ξstart : Starting boundary integral
            ξend   : Ending boundary integral
            θ      : Prolate spheroidal coordinate 
    
          Returns
          -------
            Result of the integral
        """
        ξ = numpy.linspace(ξstart, ξend, 2001)
        return numpy.trapz( numpy.sqrt( numpy.sinh( ξ )**2 + numpy.sin( θ ) ), ξ)  # trapz(y,x)
    
    def integrate_θ(θstart, θend, ξ):
        """
        Solve the θ-integral
          Parameters
          ---------- 
            θstart : Starting boundary integral
            θend   : Ending boundary integral
            ξ      : Prolate spheroidal coordinate 
    
          Returns
          -------
            Result of the integral
        """
        θ = numpy.linspace(θstart, θend, 2001)
        return numpy.trapz( numpy.sqrt( numpy.sinh( ξ )**2 + numpy.sin( θ ) ), θ)  # trapz(y,x)

    ## Local unit vectors:
    def vec_transmural(φ,θ):
        frac = 1 / numpy.sqrt( numpy.sinh(ξ)**2 + numpy.sin(θ)**2 )
        return numpy.array([ numpy.cosh(ξ)*numpy.sin(θ)*numpy.cos(φ),
                             numpy.cosh(ξ)*numpy.sin(θ)*numpy.sin(φ),
                             numpy.sinh(ξ)*numpy.cos(θ)             ]) / frac
    def vec_longitudinal(ξ,φ,θ):
        frac = 1 / numpy.sqrt( numpy.sinh(ξ)**2 + numpy.sin(θ)**2 )
        return numpy.array([ numpy.sinh(ξ)*numpy.cos(θ)*numpy.cos(φ),
                             numpy.sinh(ξ)*numpy.cos(θ)*numpy.sin(φ),
                            -numpy.cosh(ξ)*numpy.sin(θ)             ]) / frac
    def vec_circumferential(φ,θ):
        return numpy.array([ -numpy.sin(φ), numpy.cos(φ), 0 ])    



## Run this script and test the functions
if __name__ == "__main__":
   
    ## Validate coordinate conversion functions
    C = 0.75  # Focal length
    ξ = 0.63  # Random
    φ = 1.    # Random
    θ = numpy.linspace(1e-3,numpy.pi-1e-3,3)
    
    for θi in θ:
        Xprol     = numpy.array([ξ,θi,φ])
        Xcart     = Ellips_Functs.prolate_to_cartesian(Xprol, C)
        Xprol_new = Ellips_Functs.cartesian_to_prolate(Xcart, C)
        assert numpy.linalg.norm( Xprol_new - Xprol ) < 1e-10 # Check whether these are the same 
     
        
     
        
    ## Plot u & v coordinates
    import matplotlib.pyplot as plt
    C = 0.043
    H = 0.024
    ξ = numpy.linspace(0.37, 0.68, 26)
    φ = numpy.pi/2
    θ = numpy.linspace(numpy.pi/4,numpy.pi-1e-3,51)
    
    vmin = -1; vmax = 1;
    umin = -1; umax = 0.5;
    zmax = H
    
    X = numpy.zeros((len(θ),len(ξ),2))
    U = numpy.zeros((len(θ),len(ξ)))
    V = numpy.zeros((len(θ),len(ξ)))
    
    
    for i, ξi in enumerate(ξ):
        for j, θj in enumerate(θ):
            Xprol = numpy.array([ξi,θj,φ])
            X[j,i,:] = Ellips_Functs.prolate_to_cartesian(Xprol, C)[1:]
            u, v     = Ellips_Functs.prolate_to_normalizedwall( ξi,θj,X[j,i,1], (vmin,vmax), (umin,umax), zmax, (ξ[0], ξ[-1])) # Note H is not related to θ but it should
            U[j,i] = u
            V[j,i] = v
    
    
    fig, ax = plt.subplots()
    cont = ax.contourf(X[:,:,0],X[:,:,1], U, cmap=plt.get_cmap('rainbow'), levels=100)
    fig.colorbar(cont)
    ax.set_ylabel('Z-coordinate')
    ax.set_xlabel('X-coordinate')
    ax.set_title('Longitudinal coordinate: U')
    #plt.show()
    
    fig, ax = plt.subplots()
    cont = ax.contourf(X[:,:,0],X[:,:,1], V, cmap=plt.get_cmap('rainbow'), levels=100)
    fig.colorbar(cont)
    ax.set_ylabel('Z-coordinate')
    ax.set_xlabel('X-coordinate')
    ax.set_title('Transmural coordinate: V')
    plt.show()

        
    