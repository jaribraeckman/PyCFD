from __future__ import division
import numpy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 3d images
from matplotlib import cm                # colourmap

from scipy import linalg                # for implicit solution


def explicitDiffusion(Nt, Nx, T, L, D):
    dt = T/Nt
    dx = L/Nx
    u = numpy.zeros((Nx,Nt))
    x = numpy.linspace(0, L, Nx)
    t = numpy.linspace(0, T, Nt)
    alpha = D*dt/(dx*dx)

    # boundary conditions
    u[0,:] = 0
    u[-1,:] = 0

    # initial conditions
    u[:,0] = numpy.sin(numpy.pi*x)
    #un = numpy.vstack((u[-1,:],u[:-1,:]))
    #u[:-1,1:] = u[:-1,:-1] + alpha*(u[1:,:-1] - 2*u[:-1,:-1] + un[:-1,:-1])

    for j in range(Nt-1):
        for i in range(Nx-1):
            u[i,j+1] = u[i,j] + alpha*(u[i+1,j] - 2*u[i,j] + u[i-1,j])

    return u, x, t, alpha


fig = plt.figure(figsize=(12,6))
plt.rcParams['font.size'] = 15

ax = fig.add_subplot(121, projection='3d')
D = 0.495
u, x, t, alpha = explicitDiffusion(Nt = 2500, Nx = 50, L= 1., T = 1., D = D)
T, X = numpy.meshgrid(t,x)
N = u/u.max()
ax.plot_surface(T, X, u, linewidth=0, facecolors=cm.jet(N), rstride=1, cstride=50 )
ax.set_xlabel('Time $t$')
ax.set_ylabel('Distance $x$')
ax.set_zlabel('Concentration $u$')
ax.set_title('$\\alpha$ = ' + str(D))

ax = fig.add_subplot(122, projection='3d')
D1 = 0.505
u1, x1, t1, alpha1 = explicitDiffusion(Nt = 2500, Nx = 50, L= 1., T = 1., D = D1)
T1, X1 = numpy.meshgrid(t1,x1)
N1 = u1/1.
ax.plot_surface(T1, X1, u1, linewidth=0, facecolors=cm.jet(N1), rstride=1, cstride=50 )
ax.set_xlabel('Time $t$')
ax.set_ylabel('Distance $x$')
ax.set_zlabel('Concentration $u$')
ax.set_title('$\\alpha$ = ' + str(D1))

plt.tight_layout() # optimises the layout
plt.show()

plt.figure(figsize=(12, 6))
plt.subplot(121)
Nt = 2500
for i in range(Nt):
    if i % 300 == 0:
        plt.plot(x, u[:, i])
plt.xlabel('$x$')
plt.ylabel('$u$')
plt.title('$\\alpha < 0.5$')

plt.subplot(122)
for i in range(Nt):
    if i % 300 == 0:
        plt.plot(x1, u1[:, i])
plt.xlabel('$x$')
plt.ylabel('$u$')
plt.title('$\\alpha > 0.5$')
plt.rcParams['font.size'] = 15
plt.tight_layout()
plt.show()