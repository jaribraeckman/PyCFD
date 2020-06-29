import numpy
from plot import plot_cavity
from poisson import pressure_poisson
from poisson import build_up_b

nx = 41
ny = 41
dx = 2/(nx-1)
dy = 2/(ny-1)
u = numpy.zeros((ny,nx))
v = numpy.zeros((ny,nx))
b = numpy.zeros((ny,nx))
p = numpy.zeros((ny,nx))

def cavity_flow(nt, u, v, dt, dx, dy, p, rho, nu):
    un = numpy.empty_like(u)
    vn = numpy.empty_like(v)
    b = numpy.zeros((ny,nx))

    for n in range(nt):
        un = u.copy()
        vn = v.copy()

        b = build_up_b(b, rho, dt, u, v, dx, dy)
        p = pressure_poisson(p, dx, dy, b, nit)

        u[1:-1,1:-1] = (un[1:-1,1:-1] -
                        un[1:-1,1:-1] * dt / dx *
                        (un[1:-1,1:-1] - un[1:-1,0:-2]) -
                        vn[1:-1,1:-1] * dt / dy *
                        (un[1:-1,1:-1] - un[0:-2,1:-1]) -
                        dt / (2 * rho * dx) * (p[1:-1,2:] - p[1:-1,0:-2]) +
                        nu * (dt/ dx**2 *
                        (un[1:-1,2:] - 2 * un[1:-1,1:-1] + un[1:-1,0:-2]) +
                        dt / dy**2 *
                        (un[2:,1:-1] - 2 * un[1:-1,1:-1] + un[0:-2,1:-1])))
        v[1:-1,1:-1] = (vn[1:-1,1:-1] -
                        un[1:-1,1:-1] * dt / dx *
                        (vn[1:-1,1:-1] - vn[1:-1,0:-2]) -
                        vn[1:-1,1:-1] * dt / dy *
                        (vn[1:-1,1:-1] - vn[0:-2,1:-1]) -
                        dt / (2 * rho * dy) * (p[2:,1:-1] - p[0:-2,1:-1]) +
                        nu * (dt / dx**2 *
                        (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, 0:-2]) +
                        dt / dy**2 *
                        (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[0:-2, 1:-1])))
        u[0,:] = 0
        u[:,0] = 0
        u[:,-1] = 0
        u[-1,:] = 1 # u = 1 at y = 2 for cavity flow // u = 0 at y = 2 for channel flow
        v[0, :] = 0 # u,v = 0 at other boundaries
        v[:, 0] = 0
        v[:, -1] = 0
        v[-1, :] = 0

    return u, v, p

nx = 41
ny = 41
nit = 50
x = numpy.linspace(0, 2, nx)
y = numpy.linspace(0, 2, ny)
rho = 0.1
nu = 0.1
dt = 0.001
u = numpy.zeros((ny,nx))
v = numpy.zeros((ny,nx))
b = numpy.zeros((ny,nx))
p = numpy.zeros((ny,nx))
nt = 500
u, v, p = cavity_flow(nt, u, v, dt, dx, dy, p, rho, nu)
plot_cavity(x,y,p,u,v)