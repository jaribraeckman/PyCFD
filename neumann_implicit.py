from __future__ import division
import numpy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 3d images
from matplotlib import cm                # colourmap

from scipy import linalg                # for implicit solution

def implicitDiffusion(Nt,Nx, T, L, D):
    dt = T/Nt
    dx = L/Nx
    alpha = D*dt/(dx*dx)

    x = numpy.linspace(0,L,Nx)
    t = numpy.linspace(0,T,Nt)
    u = numpy.zeros((Nx,Nt))

    # initial condition
    u[:,0] = numpy.sin(numpy.pi*x)

    aa = -alpha*numpy.ones(Nx-3)
    bb = (1+2*alpha)*numpy.ones(Nx-2)
    cc = -alpha*numpy.ones(Nx-3)

    M = numpy.diag(aa,-1) + numpy.diag(bb,0) + numpy.diag(cc,1)

    for k in range(1,Nt):
        u[1:-1,k] = linalg.solve(M, u[1:-1, k-1])

    return u, x, t, alpha

fig = plt.figure(figsize=(12,6))
plt.rcParams['font.size'] = 15

ax = fig.add_subplot(121, projection='3d')
ui, xi, ti, alphai = implicitDiffusion(Nt = 2500, Nx = 50, L= 1., T = 0.1, D = 5)
Ti, Xi = numpy.meshgrid(ti,xi)
N = ui/ui.max()
ax.plot_surface(Ti, Xi, ui, linewidth=0, facecolors=cm.jet(N), rstride=1, cstride=50 )
ax.set_xlabel('Time $t$')
ax.set_ylabel('Distance $x$')
ax.set_zlabel('Concentration $u$')
ax.set_title('$\\alpha = $' + str(alphai))

ax = fig.add_subplot(122, projection='3d')
ui1, xi1, ti1, alphai1 = implicitDiffusion(Nt = 2500, Nx = 50, L= 1., T = 1., D = 0.25)
Ti1, Xi1 = numpy.meshgrid(ti1,xi1)
N = ui1/ui1.max()
ax.plot_surface(Ti1, Xi1, ui1, linewidth=0, facecolors=cm.jet(N), rstride=1, cstride=50 )
ax.set_xlabel('Time $t$')
ax.set_ylabel('Distance $x8$')
ax.set_zlabel('Concentration $u$')
ax.set_title('$\\alpha = $' + str(alphai1))
plt.tight_layout()
plt.show()