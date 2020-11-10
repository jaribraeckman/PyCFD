import numpy as np
import time
import iterative_help as help

# jacobi iteration method
def jacobi(u, u_a, f, h, max_err, max_it):
    t = time.time()
    conv = []
    it = 0
    u_n = u.copy()

    while True:
        it = it + 1
        u_n[1:-1,1:-1] = 0.25*(u_n[2:,1:-1] + u_n[:-2,1:-1] + u_n[1:-1,2:] + u_n[1:-1,:-2] - f[1:-1,1:-1]*h*h)

        err = help.error(u_n,u_a)

        conv = np.concatenate((conv,[err]))

        if err < max_err:
            break

        if it > max_it:
            break

    t = time.time() - t


    print('Computation time: ' + ('%.5f' %t) + ' s')
    print('Iterations: ', it)
    print('Maximum error: ' + ('%.4f' %err))
    help.plt2d(conv, it, 'Jacobi iteration method')
    return u_n, it, conv,t
