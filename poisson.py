import numpy

def build_up_b(b, rho, dt, u, v, dx, dy):
    b[1:-1,1:-1] = (rho * (1 / dt *
                    ((u[1:-1,2:] - u[1:-1,0:-2]) /
                    (2 * dx) + (v[2:,1:-1] - v[0:-2,1:-1]) / (2 * dy)) -
                    ((u[1:-1,2:] - u[1:-1,0:-2]) / (2 * dx))**2 -
                    2 * ((u[2:,1:-1] - u[0:-2,1:-1]) / (2 * dy) *
                         (v[1:-1,2:] - v[1:-1,0:-2]) / (2 * dx)) -
                        ((v[2:,1:-1] - v[0:-2,1:-1]) / (2 * dy))**2))
    return b

def pressure_poisson(p, dx, dy, b, nit):
    pn = numpy.empty_like(p)
    pn = p.copy()

    for q in range(nit):
        pn = p.copy()
        p[1:-1,1:-1] = (((pn[1:-1,2:] + pn[1:-1,0:-2]) * dy**2 +
                         (pn[2:,1:-1] + pn[0:-2,1:-1]) * dx**2) /
                         (2 * (dx**2 + dy**2)) -
                         dx**2 * dy**2 / (2 * (dx**2 + dy**2)) *
                         b[1:-1,1:-1])

        p[:,-1] = p[:,-2] # dp/dx = 0 at x = 2
        p[0,:] = p[1,:] # dp/dy = 0 at y = 0
        p[:,0] = p[:,1] # dp/dx = 0 at y = 0
        p[-1,:] = 0 # p = 0 at y = 2
    return p