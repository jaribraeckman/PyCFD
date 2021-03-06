import numpy # import array library
from matplotlib import pyplot # import plotting library
import time, sys

X = 2
nx = 41
dx = X/(nx-1)
nt = 15
c = 1
dt = dx/2
u = numpy.ones(nx) # initialize u
u[int(0.5/dx):int(1/dx+1)] = 2
##pyplot.plot(numpy.linspace(0,2,nx),u)
##pyplot.show()
#print(u)
un = numpy.ones(nx)
for n in range(nt):
    un = u.copy()
    for i in range(1, nx):
        u[i] = un[i] - c*(dt/dx)*(un[i] - un[i-1])
pyplot.plot(numpy.linspace(0, 2, nx), u)
pyplot.show()
