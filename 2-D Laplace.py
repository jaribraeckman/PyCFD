import numpy
from matplotlib import pyplot, cm
from mpl_toolkits.mplot3d import Axes3D
from plot import plot2D


def laplace2d(p, y, dx, dy, l1norm_target):
    l1norm = 1
    pn = numpy.empty_like(p)
    it = 0
    while l1norm>l1norm_target:
        pn = p.copy()
        p[1:-1,1:-1] = ((dy**2)*(pn[1:-1,2:]+pn[1:-1,0:-2]) + (dx**2)*(pn[2:,1:-1]+pn[0:-2,1:-1]))/(2*((dx**2)+(dy**2)))
        p[:,0] = 0
        p[:,-1] = y
        p[0,:] = p[1,:]
        p[-1,:] = p[-2,:]
        l1norm = (numpy.sum(numpy.abs(p[:])-numpy.abs(pn[:]))/numpy.sum(numpy.abs(pn[:])))
        it = it+1
    return p,it

nx = 31
ny = 31
c = 1
dx = 2/(nx-1)
dy = 1/(ny-1)

#initial conditions
p = numpy.zeros((ny,nx))

x = numpy.linspace(0,2,nx)
y = numpy.linspace(0,1,ny)

#boundary conditions
p[:,0] = 0
p[:,-1] = y
p[0,:] = p[1,:] #dp/dy = 0 @ y = 0
p[-1,:] = p[-2,:] #dp/dy = 0 @ y = 1
plot2D(x,y,p)
p, it = laplace2d(p, y, dx, dy, 1e-4)
plot2D(x,y,p)