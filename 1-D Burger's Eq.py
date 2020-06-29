import numpy
import sympy
from sympy import init_printing
from matplotlib import pyplot
from sympy.utilities.lambdify import lambdify

init_printing(use_latex=True)

x, nu, t = sympy.symbols('x nu t')
phi = (sympy.exp(-(x**2)/(4*nu)) +
       sympy.exp((-(x-2*sympy.pi)**2)/(4*nu)))

phiprime = phi.diff(x)
print(phiprime)

u = -2*nu*(phiprime/phi)+4

ufunc = lambdify((x,nu),u)

nx = 101
nt = 100
dx = 2*numpy.pi/(nx-1)
nu = .07
dt = dx*nu
x = numpy.linspace(0,2*numpy.pi,nx)
un = numpy.empty(nx)


u = numpy.asarray([ufunc(x0,nu) for x0 in x]) # u at t = 0
u0 = u.copy()

for n in range(nt):
    un = u.copy()
    for i in range(1, nx-1):
        # Calculate u at each position i at time step n+1
        u[i] = un[i] - un[i]*(dt/dx)*(un[i] - un[i-1]) + nu*(dt/(dx)**2)*(un[i+1] - 2*un[i] + un[i-1])
    u[0] = un[0] - un[0]*(dt/dx)*(un[0]-un[-2]) + nu*(dt/(dx)**2)*(un[1] - 2*un[0] + un[-2])
    u[-1] = u[0] # u[-1] = u[nx-1]

pyplot.figure(figsize=(11,7),dpi=100)
pyplot.xlim([0,2*numpy.pi])
pyplot.ylim([0,10])
pyplot.plot(x,u,label='t = 100 s')
pyplot.plot(x,u0,label='t = 0 s')
pyplot.legend()
pyplot.show()
print(u)