## To  be written
# Might be useful to have a human-readable file unlike the current .pickle files

from cardiga.dataproc import load_pickle
import os
import numpy as np

def main(file : str, name : str):
    cps, w, patchverts, patches, nelems, knotval, knotmult, boundaries = load_pickle(file)
    Title   = f"{name} multipatch geometry data"
    Data    = (cps, w, patchverts, patches, nelems, knotval, knotmult, boundaries)
    Headers = "Control points", "Weights", "Patch vertices", "Patch connectivity", \
              "Number of elements per boundary", "Knot values per boundary", \
              "Knot multiplicity per boundary", "Boundary names"

    with open(file+".txt", 'w') as f:
       f.write(Title+"\n\n")
       for head, data in zip(Headers,Data):
           f.write(head+"\n")
           if type(data) == np.ndarray:
            lines = "\n".join( [ str(row.tolist()) for row in data ] )
            f.write(lines) 
 
           if type(data) == dict:
              for key, value in data.items(): 
                  f.write('%s:%s\n' % (key, value))
           f.write("\n\n")        
    return

if __name__ == "__main__":
   files = "BV_GEOMETRY_LONG", "BV_GEOMETRY_REF", "BV_GEOMETRY_THK", "LV_GEOMETRY_PS", "LV_GEOMETRY"
   names = "Bi-ventricle REF (idealized)", "Bi-ventricle LONG (idealized)", "Bi-ventricle THK (idealized)", "left ventricle (patient specific)", "Left ventricle healthy (idealized)" 
   for file, name in zip(files,names):
    main(os.path.join("geometries",file),name)                