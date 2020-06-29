import numpy # import array library
from matplotlib import pyplot # import plotting library
import time, sys

X = 2
nx = 20
dx = X/(nx-1)
nt = 50
dt = 0.025
c = 1
u = numpy.ones(nx) # initialize u
u[int(0.5/dx):int(1/dx+1)] = 2

un = numpy.ones(nx)
for n in range(nt):
    print(n)
    un = u.copy()
    for i in range(1, nx):
        u[i] = un[i] - un[i] * (dt / dx) * (un[i] - un[i - 1])
pyplot.plot(numpy.linspace(0, 2, nx), u)
pyplot.show()
