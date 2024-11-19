import numpy as np
from .dataproc import load_pickle
from nutils import mesh, function  
import os
import sys


def is_64bit() -> bool:
    return sys.maxsize > 2**32

class Geometry():

    def __init__(self, filename, direct=""):
        self.filepath = os.path.join(direct,filename)
        self.filetype = filename.split(".")[-1]
        assert self.filetype in ["txt","pickle"], f"Unknown filetype {self.filetype}, use '.txt' or '.pickle' instead."
        return
   
    def height(self,):
        """
        Determine the height of the basal plane (assuming it is a plane). 
        The value is required to correctly determine the cavity volumes.
            Returns
            -------
            float height
        """
        return self.topo.boundary['base_l'].sample('bezier',2).eval(self.geom)[0,-1] # base_l should always be a boundary of the ventricle geometry
   
    
    def get_topo_geom(self, nrefine=0):
        """
        Get the topology and geometry given an input .pickle file
            Parameters
            ---------- 
            filename : str   
                Filename of the .pickle file containing all the geometrical information
            direct   : str 
                Directory of the .pickle file
            nrefine  : int 
                Number of uniform refinements desired 

            Returns
            -------
            nutils topology (topo) and geometry (geom)
        """
        if self.filetype == "pickle":
            cps, w, patchverts, patchesn, nelems, knotval, knotmult, boundaries = load_pickle(self.filepath)  
            patches = patchesn.astype(np.int64) #if is_64bit() else patchesn.astype(np.int32) 
        elif self.filetype == "txt":
            cps, w, patchverts, patches, nelems, knotval, knotmult, boundaries = self.load_txt(self.filepath) 
        
        topo, lingeom = mesh.multipatch(patches=patches, patchverts=patchverts, nelems=nelems)
        topo     = topo.withboundary(**boundaries)
        bsplines = topo.basis('spline', degree=2, patchcontinuous=False, knotvalues=knotval, knotmultiplicities=knotmult)
        weight   = bsplines @ w
        nurbsbasis = bsplines * w / weight
        geom = nurbsbasis @ cps

        if nrefine:
            topo = topo.refine(nrefine)

        # Store as attribute
        self.topo = topo
        self.geom = geom

        return topo, geom

    @staticmethod
    def load_txt(filename : str = 'filename', returnDict=False):
        import ast

        if filename.split(".")[-1] != "txt":
            filename += ".txt"

        with open(filename, 'r') as f:
            lines = f.readlines()

        Headers = "Control points", "Weights", "Patch vertices", "Patch connectivity", \
                "Number of elements per boundary", "Knot values per boundary", \
                "Knot multiplicity per boundary", "Boundary names" # Ordering is of importance! Should be same as in the save_txt() function
        DictKeys = "control points", "weights", "patch vertices", "patch connectivity", \
                "nelems", "knot values", "knot multiplicity", "boundary names"
        
        # Strip empty spaces and remove '\n'
        lines = [idat.strip("\n") for idat in lines if len(idat.strip("\n")) != 0]   
        catch_line = "View"
        idx = []   
        for i, line in enumerate(lines):
            if line in Headers:
                idx += [i]
        idx += [None]
        loadedData = []
        for k, (i,j) in enumerate(zip(idx[:-1],idx[1:])):
            if k < 4: # We encounter np.arrays
                loadedData += [np.array([ast.literal_eval(iline) for iline in lines[(i+1):j]])]  
            # elif k == 3: # We have the weights list (special case)
            #     loadedData += [np.array([ast.literal_eval(', '.join(lines[(i+1):j]))])]  
            else: # We encounter dicts
                d = {}
                for b in lines[(i+1):j]:
                    i = b.split(':')
                    if k != len(Headers)-1:
                        d[ast.literal_eval(i[0])] = ast.literal_eval(i[1])
                    else: # Else we have the boundaries dict
                        d[i[0]] = i[1]      
                loadedData += [d]

        if returnDict:
            return {key : value for key, value in zip(DictKeys, loadedData)}
        else:
            return loadedData


class InterfaceElem:
    """
        This class facilitates calculates (~approximates) the length of the element normal to the interface 
             of different patches in a multipatch topology. The class also supports Hierarchical topologies.
             Initialize the class first after which the function 'self.length_normal()' should be used to get
             the lengths per element.
        """
    def __init__(self, ns, reftopo, target='elem'): # Hierarchical refined topology
        self.ns = ns
        self.reftopo = reftopo
        self.ns.elembasis = reftopo.basis('discont',0)
        self.target   = target # Column with unknown element lengths 
        self.ns.elemf = f'elembasis_n ?{target}_n' 


        ref_topo = reftopo
        while True:
            try:
                self.nrpatches = len(ref_topo.patches)
                assert type(self.nrpatches) == int, "incorrect nrpatches type"
                break
            except:
                ref_topo = ref_topo.basetopo

        self.refifaces = reftopo.basetopo.interfaces['interpatch'] & reftopo.interfaces
        self.interface_elems = self.refifaces.sample('gauss', 0).eval(np.stack([reftopo.f_index, function.opposite(reftopo.f_index)]))
        self.interface_elems_unique = np.unique(self.interface_elems) 
        self.total_elems = reftopo.sample('gauss', 0).eval(np.stack([reftopo.f_index]))

        self.extr_elem, self.extr_celem = self.__find_extraordinary_element(reftopo)
        return

    def __find_extraordinary_element(self, reftopo): 
        
        # find index which has 2 or more patches attached to its element
        u, count =np.unique(self.interface_elems, return_counts=True)
        dupl_elems = u[count>1]

        
        elems_patches = []
        for ip in range(self.nrpatches):
            elems_patches.append(reftopo[f'patch{ip}'].sample('gauss', 0).eval(np.stack([reftopo.f_index]))) # Append indices of all elements in a patch

        # Check if an interface element has 2 shared patches
        extr_element = []
        c_patches = []
        for dupl_elem in dupl_elems:
            indices = np.argwhere( dupl_elem == self.interface_elems )
            col = 0 if indices[0,1] == 1 else 1
            other_elem_indices = self.interface_elems[ indices[:,0], col ]

            # Check if other elem_indices are part of atleast 2 difference patches
            within = np.array([False]*len(elems_patches))
            
            for index in other_elem_indices:
                for ip, elems_patch in enumerate(elems_patches):
                    if within[ip] == False: # Make sure we are not overwritting a previous True value
                        within[ip] = index in elems_patch 
                if sum(within) > 1: # We have an extraordinary patch element (sum -> more than 1 True in the list)  
                    extr_element.append(dupl_elem) # Index of the extraordinary element
                    c_patches.append(sum(within)) # Count which is used to divide the integration of the total interface             
        return np.array(extr_element), np.array(c_patches)

    # Get the length of the interface in normal direction to it
    def length_normal(self, value):
        Ainterface = function.eval( self.refifaces.integral('elemf detFint d:x' @ self.ns, degree=5).derivative(self.target), **value)
        if len(self.extr_elem) != 0:
            Ainterface[self.extr_elem] /= self.extr_celem
        Ainterface_opposite = function.eval( self.refifaces.integral('opposite(elemf detFint) d:x' @ self.ns, degree=5).derivative(self.target), **value)
        # make sure Ainterface is corretcly filled with opposite values (these are currently 0's)
        Ainterface[self.interface_elems[:,1]] = Ainterface_opposite[self.interface_elems[:,1]]
        Vdomain = function.eval( self.reftopo.integral('elemf detFint d:x' @ self.ns, degree=5).derivative(self.target), **value)
        Ainterface[np.isclose(Ainterface,0)] = 1
        h_interface = Vdomain/Ainterface
        return h_interface 
        
         
# Function implementing the mesh-regularization algorithm.
def regularize_mesh(topo, difference):
    # Refine the hierarhical mesh until satisfying the requirement on the
    # size difference between neighboring elements.
    while True:
        elem_indices = get_elements_to_be_refined(topo, difference)
        if not elem_indices:
            break
        topo = topo.refined_by(elem_indices)

    return topo

# Function returning the indices of elements that have a neighbour which
# is more than `difference` times refined compared to itself.
def get_elements_to_be_refined(topo, difference):
    assert difference > 0, 'difference must be positive'

    # Initiate the list of elements to be refined.
    elem_indices = []

    for transforms in topo.interfaces.transforms, topo.interfaces.opposites:
        for trans in transforms:
            index, tail = topo.transforms.index_with_tail(trans)
            if len(tail) > difference + 1:
                # Mark the element for refinement if the transformation from
                # interface to topology (consisting of one edge-to-volume
                # transformation and any number of coarsening transformations)
                # is longer than the specified allowable difference plus 1.
                elem_indices.append(index)

    return elem_indices
            