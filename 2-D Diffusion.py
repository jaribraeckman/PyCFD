from mpl_toolkits.mplot3d import Axes3D

import numpy
from matplotlib import pyplot, cm

nx = 31
ny = 31
nu = 0.05
dx = 2 / (nx - 1)
dy = 2 / (ny - 1)
sigma = 0.25
dt = sigma * dx * dy / nu

x = numpy.linspace(0, 2, nx)
y = numpy.linspace(0, 2, ny)
X, Y = numpy.meshgrid(x,y)
u = numpy.ones((ny, nx))
un = numpy.ones((ny, nx))

def diffuse(nt):
    u[int(.5/dy):int(1/dy+1),int(.5/dx):int(1/dx+1)] = 2

    for n in range(nt+1):
        un = u.copy()
        u[1:-1,1:-1] = un[1:-1,1:-1] + nu*(dt/dx**2)*(un[1:-1,2:]-2*un[1:-1,1:-1]+un[1:-1,0:-2]) \
                       + nu*(dt/dy**2)*(un[2:,1:-1]-2*un[1:-1,1:-1]+un[0:-2,1:-1])
        u[0,:] = 1
        u[:,0] = 1
        u[-1,:] = 1
        u[:,-1] = 1

    fig = pyplot.figure(figsize=(11,7))
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, u[:], rstride=1, cstride=1, cmap=cm.viridis, linewidth=0, antialiased=True)
    ax.set_zlim(1,2.5)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    pyplot.show()

diffuse(0)
diffuse(10)
diffuse(20)
diffuse(30)
diffuse(40)