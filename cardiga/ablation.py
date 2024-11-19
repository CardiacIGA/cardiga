from nutils.topology import Topology
import numpy as np
from nutils import function
import treelog


_ = np.newaxis

class Ablation:
    """
        Class that facilitates ablation of the left ventricle. It allows the user to define points in the domain 
            at which a local change in material property is applied (loss of contractility or increase in stiffness).
            Ablation is currently assumed to have a spherical shape.

    """
    
    def __init__(self, topo: Topology, geom: function.Array):
        if topo ==None and geom == None:
            return None
        
        self.topo = topo
        self.geom = geom
        self.ns   = function.Namespace()
        self.ns.x = geom
        return
    
    def probe(self, X, R, δR=0.):
        """ 
            Function that models the ablation probe in terms of location and intensity (radius of spherical beam).

            Input:
                -------- 
                X  : Location of the probe,
                R  : Radius of the probe
                δR : Radius of the border zone of the probe (R < δR)

            Returns:
                --------
                Spatial function (dependent on x) that yields following values:
                    1     for <=  R
                    [1-0] for <  δR  <  
                    0     for >= δR    
        """
        dR     = - ( R**2 - (R + δR)**2)
        fprobe = np.dot(self.ns.x - X, self.ns.x - X) - (R + δR)**2 # Unbounded and nonscaled probe function

        # Bound and scale the probe function such that 1 values are in the center, and 0 everywhere else. 
        # Spherical gradient in borderzone.
        return np.minimum( -np.minimum( fprobe, 0 ), dR )/(dR) 

    def ablate_sites(self, Xsites, Rsites, δRsites=0.):
        """ 
            Function that combines different probe functions (ablation sites) into a single field bounded between [0,1].

            Input:
                -------- 
                Xsites  : Location of the probe,
                Rsites  : Radius of the probe
                δRsites : Radius increment (thickness) of the border zone of the probe

            Returns:
                --------
                self.ablated_sites : 
                Spatial function (dependent on x) that yields following values:
                    1     if dead tissue (ablated)
                    [1-0] if in border-zone of (dead - healthy)
                    0     if healthy tissue    

        """
        δRsites = np.zeros(len(Rsites)) + δRsites
        AsiteName = lambda i: f"Asite{i}"

        for i, (X, R, δR) in enumerate(zip(Xsites, Rsites, δRsites)):

            asite = AsiteName(i) # Name of the current ablation site
            setattr(self.ns, asite,  self.probe(X, R, δR=δR) )

            if i > 0:
                Asites = getattr(self.ns, asite)*( 1 - Asites ) + Asites
                #Asites = np.max( Asites, getattr(ns, asite) ) #ns._attributes.update( ns1._attributes)??
            else:
                Asites = getattr(self.ns, AsiteName(0))

        treelog.info(f"Ablation performed at {len(Rsites)} different sites.")

        self.ablated_sites = Asites
        return Asites
    



