import numpy
from plot import plot2D

nx = 50
ny = 50
#nt = 100
xmin = 0
xmax = 2
ymin = 0
ymax = 1

dx = (xmax-xmin)/(nx-1)
dy = (ymax-ymin)/(ny-1)

p = numpy.zeros((ny,nx))
pd = numpy.zeros((ny,nx))
b = numpy.zeros((ny,nx))
x = numpy.linspace(xmin,xmax,nx)
y = numpy.linspace(ymin,ymax,ny)

b[int(ny/4),int(nx/4)] = 100
b[int(3*ny/4),int(3*nx/4)] = -100

plot2D(x, y, b)

def poisson(l1norm_target):
    l1norm = 1
    it = 0
    while l1norm>l1norm_target:
        pd = p.copy()
        p[1:-1,1:-1] = ((pd[1:-1,2:]+pd[1:-1,0:-2])*(dy**2) + (pd[2:,1:-1]+pd[0:-2,1:-1])*(dx**2) -
                        b[1:-1,1:-1]*(dx**2)*(dy**2))/(2*((dx**2)+(dy**2)))
        p[0,:] = 0
        p[-1,:] = 0
        p[:,0] = 0
        p[:,-1] = 0
        if numpy.sum(numpy.abs(pd[:])) != 0:
            l1norm = numpy.sum(numpy.abs(p[:])-numpy.abs(pd[:]))/numpy.sum(numpy.abs(pd[:]))
        it = it + 1
    return p,it
p,it = poisson(1e-4)
print(it)
plot2D(x,y,p)

