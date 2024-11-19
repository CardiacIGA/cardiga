## Test script to compute distance between an arbitrary manifold (given as a convex hull in 3D) and a point
from pygel3d import hmesh
import numpy as np
from scipy.spatial import ConvexHull

def dist(hull, points):
    # Construct PyGEL Manifold from the convex hull
    m = hmesh.Manifold()
    for s in hull.simplices:
        m.add_face(hull.points[s])

    dist = hmesh.MeshDistance(m)
    res = []
    for p in points:
        # Get the distance to the point
        # But don't trust its sign, because of possible
        # wrong orientation of mesh face
        d = dist.signed_distance(p)

        # Correct the sign with ray inside test
        if dist.ray_inside_test(p):
            if d > 0:
                d *= -1
        else:
            if d < 0:
                d *= -1
        res.append(d)
    return np.array(res)

plane = np.array([ [0,0,0], [1,0,0], [1,1,0], [0,1,0], [0,0,1], [1,0,1], [1,1,1], [0,1,1]  ]) # Unit length cube
hull  = ConvexHull(plane)
point = np.array([[0.5, 0.5, 3],
                  [ -1, -1 , 2]])

distance = dist(hull, point)
print(distance)