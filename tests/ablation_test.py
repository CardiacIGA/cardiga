import numpy as np
import matplotlib.pyplot as plt
from matplotlib import collections
from nutils import export, mesh, function, solver
from mpl_toolkits.axes_grid1 import make_axes_locatable

## Make a 2D slab for ablation test-case



## Input

# Geometric input
L = 1       # length of domain
H = 0.5     # Height of the domain

# Probe dimensions
Rprobe = 0.1 # Radius of the probe
Xprob  = np.array([0.5,0.5]) # Location of the probe
δprobe = 0.5*Rprobe # Synthetic borderzone size, used for hierarchical refinement marking

# Discretization input
nelemsL = 10 # Nr of elements in length
nelemsH = 10 # Nr of elements in height
bdegree = 3
gdegree = 5
btype   = 'th-spline'  
image   = True

## Domain and Topology
domain, geom =  mesh.rectilinear([np.linspace(0,L,nelemsL+1), np.linspace(0,H,nelemsH+1)])

# Define the function space
ns = function.Namespace()
ns.x         = geom 

# Probe functionals
def probe(X, R):
    """ X : Location of the probe,
        R : Radius of the probe
    """
    return function.dot(ns.x - X, ns.x - X) - R**2 #f'( x_0 - {X[0]} )^2 + ( x_1 - {X[1]} )^2 - {R}^2'

def ablate_sites(ns, Xsites, Rsites, δRsites=0., Hierarchical=False):
    δRsites = np.zeros(len(Rsites)) + δRsites
    AsiteName = lambda i: ( f"AsiteH{i}" if Hierarchical else f"Asite{i}")
    for i, (X, R, δR) in enumerate(zip(Xsites, Rsites, δRsites)):
        asite = AsiteName(i) # Name of the current ablation site
        dR = - ( R**2 - (R + δR)**2)
        setattr(ns, asite,  np.min( -np.min( probe(X, R + δR), 0 ), dR )/(dR))
        if i > 0:
            Asites = getattr(ns, asite)*( 1 - Asites ) + Asites
            #Asites = np.max( Asites, getattr(ns, asite) ) #ns._attributes.update( ns1._attributes)??
        else:
            Asites = getattr(ns, AsiteName(0))
    return Asites


Xlocations = np.array([[0.4,0.5],
                       [0.5,0.5],
                       [0.75,0.5]])
Rlocations = np.array([0.3,0.05,0.1])

#ns.fprobeHref  = ablate_sites(ns, Xlocations, Rlocations, δRsites=δprobe)
ns.fprobeHref = ablate_sites(ns, Xlocations, Rlocations, δRsites=δprobe, Hierarchical=True)
ns.fprobe = ablate_sites(ns, Xlocations, Rlocations, δRsites=δprobe, Hierarchical=True)


def Hrefinement(topo, ns, Hlevels=2, btype="th-spline", bdegree=3):
    

    for i in range(Hlevels):
        print(f"H-ref: {i}")
        reftopo     = topo.refined
        ns.refbasis = reftopo.basis(btype, degree=bdegree)

    
        indicator   = reftopo.integrate_elementwise('sqrt( fprobeHref_,k fprobeHref_,k ) d:x' @ ns(), degree=bdegree*2)
        indicator  /= reftopo.integrate_elementwise('d:x' @ ns(), degree=bdegree*2)
        treshold    = 0.1 #np.mean(indicator)
        supp        = np.where( indicator > treshold )[0]
        print(supp)
        #treelog.info(np.mean(indicator / elem_volume))
        #indicator   = reftopo.integral('refbasis_n ( scar_,k scar_,k ) d:x' @ ns, degree=bdegree*2).eval(**lhs)
        #supp        = ns.refbasis.get_support(indicator > np.mean(indicator))

        reftopo   = topo.refined_by(reftopo.transforms[supp])

        topo = reftopo
    return reftopo

domain = Hrefinement(domain, ns, Hlevels=2, btype=btype, bdegree=bdegree)


## Project a basis
# ns.basisproj = domain.basis(btype, degree=bdegree)
# ns.proj      = 'basisproj_n ?plhs_n'
# ns.fproj      = np.max( np.min( ns.proj, 1 ), 0)
# sqr  = domain.integral('( proj - fprobe )^2 d:x' @ ns, degree=gdegree)
# proj = solver.optimize('plhs', sqr, droptol=1e-15, tol=1e-6)

# bezier = domain.sample('bezier', 20)
# x, fprobe, fproj = bezier.eval(['x_i', 'fprobeHref', 'fproj'] @ ns, plhs=proj)


# fig = plt.figure(figsize=(10, 10))
# ax = fig.add_subplot(111, aspect='equal', xlabel=r'$x$ [mm]', ylabel=r'$y$ [mm]')#
# im = ax.tripcolor(x[:,0], x[:,1], bezier.tri, fprobe, shading='gouraud', cmap='jet')
# ax.add_collection(collections.LineCollection(x[bezier.hull,:2], colors='w', linewidths=.3))
# ax.autoscale(enable=True, axis='both', tight=True)
# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="5%", pad=0.05)
# #plt.colorbar(im, cax=cax)
# cb = fig.colorbar(im, cax=cax)
# cb.set_label(r'Ablation site functional $f_{asites}(x)$ [-]')

# fig = plt.figure(figsize=(10, 10))
# ax = fig.add_subplot(111, aspect='equal', xlabel=r'$x$ [mm]', ylabel=r'$y$ [mm]')#
# im = ax.tripcolor(x[:,0], x[:,1], bezier.tri, fproj, shading='gouraud', cmap='jet')
# ax.add_collection(collections.LineCollection(x[bezier.hull,:2], colors='w', linewidths=.3))
# ax.autoscale(enable=True, axis='both', tight=True)
# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="5%", pad=0.05)
# #plt.colorbar(im, cax=cax)
# cb = fig.colorbar(im, cax=cax)
# cb.set_label(r'Projected Abl. site functional $f_{asites}(x)$ [-]')
# plt.show()




ns.δ = function.eye(domain.ndims)

ns.ubasis    = domain.basis(btype, degree=bdegree).vector(domain.ndims)

ns.loadv_i = '< -1, 0 >_i' # Load on right/bottom boundary/point
ns.tload   = 0.3
poisson = 0.2 
ns.u_i  = 'ubasis_ni ?ulhs_n'
ns.X_i  = 'x_i + u_i'

ns.ν = poisson
ns.λ = 2 * poisson * ( 5*ns.fprobe + 1 )
ns.μ = 1 - 2 * poisson
ns.E = 'λ ( 1 + ν ) ( 1 - 2 ν ) / ν '
ns.ε_ij = '( u_i,j + u_i,j ) / 2' #'( u_i,j + u_i,j ) / 2'
ns.σ_ij = 'λ ε_kk δ_ij + 2 μ ε_ij'
ns.gradN_nij = 'ubasis_ni,j'


# sqr = domain.boundary['left'].integral('u_k u_k d:x' @ ns, degree=gdegree)
sqr = domain.boundary['left'].integral('u_0 u_0 d:x' @ ns, degree=gdegree)
sqr += domain.boundary['bottom'].integral('u_1 u_1 d:x' @ ns, degree=gdegree)
#sqr += domain.boundary['right'].integral('(u_0 - .5)^2 d:x' @ ns, degree=gdegree)
cons = solver.optimize('ulhs', sqr, droptol=1e-15, tol=1e-6)

res = domain.integral('(ubasis_ni,j) σ_ij d:x' @ ns, degree=gdegree)
res -= domain.boundary['bottom'].integral('tload ubasis_ni loadv_i d:x' @ ns, degree=gdegree)
lhs = solver.solve_linear('ulhs', res, constrain=cons)



## Adding several ablation points
t  = np.linspace(0,6,1000)
g1 = np.sqrt( np.maximum( -(t-2)**2 + 1, 0 ) )
g2 = np.sqrt( np.maximum( -(t-3.5)**2 + 1, 0 ) )
g3 = g1*(1-g2) + g2#np.maximum(g1,g2)

# fig = plt.figure(figsize=(10, 10))
# ax = fig.add_subplot(111, aspect='equal', xlabel=r'$x$ [mm]', ylabel=r'$y$ [mm]')#
# ax.plot(t, g1, "r")
# ax.plot(t, g2, "b")
# ax.plot(t, g3, "g")
# ax.autoscale(enable=True, axis='both', tight=True)
# plt.show()


if image:
    bezier = domain.sample('bezier', 20)
    X, x, ux, strain_xx, fprobe, fprobeH, fmark = bezier.eval(['X_i', 'x_i', 'u_1', 'ε_00', 'fprobe', 'fprobeHref', 'sqrt( fprobeHref_,k fprobeHref_,k )'] @ ns, ulhs=lhs)
    #x, fprobe = bezier.eval(['x_i', 'fprobeHref'] @ ns, ulhs=lhs)

    # Retrieve the Gauss Quadrature points (for indication purposes)
    gauss = domain.sample('gauss', gdegree)
    x_gauss, fprobe_gauss = gauss.eval(['x_i', 'fprobe'] @ ns, ulhs=lhs)


    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, aspect='equal', xlabel=r'$x$ [mm]', ylabel=r'$y$ [mm]')#
    #im = ax.tripcolor(X[:,0], X[:,1], bezier.tri, ux, shading='gouraud', cmap='jet')
    im = ax.tripcolor(x[:,0], x[:,1], bezier.tri, fprobeH, shading='gouraud', cmap='jet')
    # ax.scatter(x_gauss[:,0], x_gauss[:,1], fprobe_gauss.astype(int))
    ax.add_collection(collections.LineCollection(x[bezier.hull,:2], colors='w', linewidths=.3))
    ax.autoscale(enable=True, axis='both', tight=True)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    #plt.colorbar(im, cax=cax)
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(r'Ablation site functional $f_{asites}(x)$ [-]')
    plt.show()

    # # Stiffness distribution
    # fig = plt.figure(figsize=(10, 10))
    # ax = fig.add_subplot(111, aspect='equal', xlabel=r'$x$ [mm]', ylabel=r'$y$ [mm]')#
    # #im = ax.tripcolor(X[:,0], X[:,1], bezier.tri, ux, shading='gouraud', cmap='jet')
    # im = ax.tripcolor(x[:,0], x[:,1], bezier.tri, fprobe, shading='gouraud', cmap='jet')
    # ax.scatter(x_gauss[:,0], x_gauss[:,1], 3*fprobe_gauss.astype(int))
    # ax.add_collection(collections.LineCollection(x[bezier.hull,:2], colors='w', linewidths=.3))
    # ax.autoscale(enable=True, axis='both', tight=True)
    # cb = fig.colorbar(im)
    # cb.set_label(r'Probe functional fprobe(x) [m]')
    # plt.show()

    # Deformation
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, aspect='equal', xlabel=r'$x$ [mm]', ylabel=r'$y$ [mm]')#
#im = ax.tripcolor(X[:,0], X[:,1], bezier.tri, ux, shading='gouraud', cmap='jet')
im = ax.tripcolor(X[:,0], X[:,1], bezier.tri, strain_xx, shading='gouraud', cmap='jet')
ax.add_collection(collections.LineCollection(X[bezier.hull,:2], colors='k', linewidths=.1))
ax.autoscale(enable=True, axis='both', tight=True)
cb = fig.colorbar(im)
cb.set_label(r'$y-Displacement$ [m]')
plt.show()