import numpy
from plot import plot_channel
from poisson_channel import pressure_poisson_periodic
from poisson_channel import build_up_b

nx = 41
ny = 41
dx = 2/(nx-1)
dy = 2/(ny-1)
u = numpy.zeros((ny,nx))
v = numpy.zeros((ny,nx))
b = numpy.zeros((ny,nx))
p = numpy.zeros((ny,nx))

def channel_flow(nt, u, v, dt, dx, dy, p, rho, nu, F):
    un = numpy.empty_like(u)
    vn = numpy.empty_like(v)
    b = numpy.zeros((ny,nx))
    u_diff = 1
    iteration = 0
    while u_diff > 0.001:
        un = u.copy()
        vn = v.copy()

        b = build_up_b(b, rho, dt, u, v, dx, dy)
        p = pressure_poisson_periodic(p, dx, dy, b, nit)

        u[1:-1,1:-1] = (un[1:-1,1:-1] -
                        un[1:-1,1:-1] * dt / dx *
                        (un[1:-1,1:-1] - un[1:-1,0:-2]) -
                        vn[1:-1,1:-1] * dt / dy *
                        (un[1:-1,1:-1] - un[0:-2,1:-1]) -
                        dt / (2 * rho * dx) * (p[1:-1,2:] - p[1:-1,0:-2]) +
                        nu * (dt/ dx**2 *
                        (un[1:-1,2:] - 2 * un[1:-1,1:-1] + un[1:-1,0:-2]) +
                        dt / dy**2 *
                        (un[2:,1:-1] - 2 * un[1:-1,1:-1] + un[0:-2,1:-1])) +
                        F * dt)
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

        # Periodic BC u @ x = 2
        u[1:-1,-1] = (un[1:-1,-1] -
                        un[1:-1,-1] * dt / dx *
                        (un[1:-1,-1] - un[1:-1,-2]) -
                        vn[1:-1,-1] * dt / dy *
                        (un[1:-1,-1] - un[0:-2,-1]) -
                        dt / (2 * rho * dx) * (p[1:-1,0] - p[1:-1,-2]) +
                        nu * (dt/ dx**2 *
                        (un[1:-1,0] - 2 * un[1:-1,-1] + un[1:-1,-2]) +
                        dt / dy**2 *
                        (un[2:,-1] - 2 * un[1:-1,-1] + un[0:-2,-1])) +
                        F * dt)

        # Periodic BC u @ x = 0
        u[1:-1,0] = (un[1:-1,0] -
                        un[1:-1,0] * dt / dx *
                        (un[1:-1,0] - un[1:-1,-1]) -
                        vn[1:-1,0] * dt / dy *
                        (un[1:-1,0] - un[0:-2,0]) -
                        dt / (2 * rho * dx) * (p[1:-1,1] - p[1:-1,-1]) +
                        nu * (dt/ dx**2 *
                        (un[1:-1,1] - 2 * un[1:-1,0] + un[1:-1,-1]) +
                        dt / dy**2 *
                        (un[2:,0] - 2 * un[1:-1,0] + un[0:-2,0])) +
                        F * dt)

        # Periodic BC v @ x = 2
        v[1:-1,-1] = (vn[1:-1,-1] -
                        un[1:-1,-1] * dt / dx *
                        (vn[1:-1,-1] - vn[1:-1,-2]) -
                        vn[1:-1,-1] * dt / dy *
                        (vn[1:-1,-1] - vn[0:-2,-1]) -
                        dt / (2 * rho * dy) * (p[2:,-1] - p[0:-2,-1]) +
                        nu * (dt / dx**2 *
                        (vn[1:-1, 0] - 2 * vn[1:-1, -1] + vn[1:-1, -2]) +
                        dt / dy**2 *
                        (vn[2:, -1] - 2 * vn[1:-1, -1] + vn[0:-2, -1])))

        # Periodic BC v @ x = 0
        v[1:-1,0] = (vn[1:-1,0] -
                        un[1:-1,0] * dt / dx *
                        (vn[1:-1,0] - vn[1:-1,-1]) -
                        vn[1:-1,0] * dt / dy *
                        (vn[1:-1,0] - vn[0:-2,0]) -
                        dt / (2 * rho * dy) * (p[2:,0] - p[0:-2,0]) +
                        nu * (dt / dx**2 *
                        (vn[1:-1, 1] - 2 * vn[1:-1, 0] + vn[1:-1, -2]) +
                        dt / dy**2 *
                        (vn[2:, 0] - 2 * vn[1:-1, 0] + vn[0:-2, 0])))

        # Wall BC: u,v = 0 @ y = 0,2
        u[0,:] = 0
        u[-1,:] = 0
        v[0,:] = 0
        v[-1,:] = 0

        u_diff = (numpy.sum(u) - numpy.sum(un))/numpy.sum(u)
        iteration += 1
    print(iteration)
    return u, v, p

nx = 41
ny = 41
nit = 50
nt = 10
x = numpy.linspace(0, 2, nx)
y = numpy.linspace(0, 2, ny)
rho = 0.1
nu = 0.1
dt = 0.01
F = 1
u = numpy.zeros((ny,nx))
v = numpy.zeros((ny,nx))
b = numpy.zeros((ny,nx))
p = numpy.zeros((ny,nx))
nt = 10
u, v, p = channel_flow(nt, u, v, dt, dx, dy, p, rho, nu, F)
plot_channel(x,y,p,u,v)