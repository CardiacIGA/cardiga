import numpy as np
import matplotlib.pyplot as plt
_ = np.newaxis

def random_sample_triangle(P1 : np.ndarray , P2 : np.ndarray, P3 : np.ndarray, nsamples : int = 200):
    v1 = P2-P1
    v2 = P3-P1

    u = np.random.random_sample((nsamples, 2))
    mask = (u.sum(axis=1) > 1)
    u[mask] *= -1
    u[mask] +=  1

    p = u[:,0]*v1[:,_] + u[:,1]*v2[:,_]
    randpoints = p.T + P1
    return randpoints

def remove_near_indentical_rows(arr, threshold):
        
    row, column = arr.shape
    arg = arr.argsort(axis=0)[:, 0]
    arr=arr[arg]

    arr_mask = np.zeros(row, dtype=bool)
    cur_row = arr[0]
    arr_mask[0] = True
    for i in range(1, row):
        if np.sum(np.abs(arr[i] - cur_row))/column > threshold:
            arr_mask[i] = True
            cur_row = arr[i]

    arg = arg[arr_mask]
    return arr[arg]  

def uniform_sample_2Dtriangle(P1, P2, P3, nsample : int = 11):
    v1 = P2-P1
    v2 = P3-P1
    A  = np.concatenate([v1[:,_],v2[:,_]],axis=1) # Linear tranformation matrix

    x   = np.linspace(0,1,nsample)
    for i, ix in enumerate(x):
        if i == 0:
           Xsub = np.concatenate([x,np.ones(len(x))*ix]).reshape(2,-1).T 
           X    = Xsub.copy()   
        else:
           Xsub = np.concatenate([x[:-i],np.ones(len(x[:-i]))*ix]).reshape(2,-1).T   
           X = np.concatenate([X, Xsub])
    return np.dot(A,X.T).T+P1 


def cluster_based_on_either_xydist(a, dist_thresh=10):
    c0 = np.abs(a[:,0,None]-a[:,0])<dist_thresh
    c1 = np.abs(a[:,1,None]-a[:,1])<dist_thresh
    c01 = c0 & c1
    return a[~np.triu(c01,1).any(0)]

P1 = np.array([-0.35,-2.5])
P2 = np.array([0.35,-2.5])
P3 = np.array([0,1.75])

points = np.concatenate([P1[:,_],P2[:,_],P3[:,_],P1[:,_]],axis=1)
# randpoints = random_sample_triangle(P1,P2,P3)
# print(len(randpoints))
# randpoints = cluster_based_on_either_xydist(randpoints, dist_thresh=5e-1) # Threshold is based on radians
# print(len(randpoints))

randpoints = uniform_sample_2Dtriangle(P1, P2, P3, nsample=50)

fig = plt.figure(figsize=(7.5, 5))
ax = fig.add_subplot(111)
ax.plot(points[0],points[1], color='r',marker='o', linestyle='solid')
ax.plot(randpoints[:,0],randpoints[:,1], color='b', marker='x', linestyle='None')
ax.grid()
plt.tight_layout()
plt.show()