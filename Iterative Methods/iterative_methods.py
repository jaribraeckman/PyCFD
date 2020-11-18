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

        err = help.error(u_n, u_a)

        conv = np.concatenate((conv, [err]))

        if err < max_err:
            break

        if it >= max_it:
            break

    t = time.time() - t


    print('Computation time: ' + ('%.5f' %t) + ' s')
    print('Iterations: ', it)
    print('Maximum error: ' + ('%.4f' %err))
    help.plt2d(conv, it, 'Jacobi iteration method')
    return u_n, it, conv,t

def gauss_seidel(u, u_a, f, h, max_err, max_it):
    t = time.time()
    conv = []
    it = 0
    u_n = u.copy()
    nx, ny = np.shape(u)

    while True:
        it = it + 1
        for i in range(1,nx-1):
            for j in range(1,ny-1):
                u_n[i,j] = 0.25*(u_n[i+1,j] + u_n[i-1,j] + u_n[i,j+1] + u_n[i,j-1] - f[i,j]*h*h)

        err = help.error(u_n, u_a)

        conv = np.concatenate((conv, [err]))

        if err < max_err:
            break

        if it >= max_it:
            break

    t = time.time() - t

    print('Computation time: ' + ('%.5f' %t) + 's')
    print('Iterations: ', it)
    print('Maximum error: ' + ('%.4f' %err))
    help.plt2d(conv, it, 'Gauss-Seidel iteration method')

    return u_n, it, conv, t

def sor(u, u_a, f, h, max_err, max_it, w):
    t = time.time()
    conv = []
    it = 0
    u_n = u.copy()
    nx, ny = np.shape(u)

    while True:
        it = it + 1
        for i in range(1,nx-1):
            for j in range(1,ny-1):
                u_n[i,j] = (1-w)*u_n[i,j] + (w/4)*(u_n[i+1,j] + u_n[i-1,j] + u_n[i,j+1] + u_n[i,j-1] - f[i,j]*h*h)

        err = help.error(u_n, u_a)

        conv = np.concatenate((conv, [err]))

        if err < max_err:
            break

        if it >= max_it:
            break

    t = time.time() - t

    print('Computation time: ' + ('%.5f' %t) + 's')
    print('Iterations: ', it)
    print('Maximum error: ' + ('%.4f' %err))
    help.plt2d(conv, it, 'Successive over-relaxation iteration method ($\omega$ = '+str(w)+')')

    return u_n, it, conv, t

def line_gauss_seidel(u, u_a, f, h, max_err, max_it, method):
    t = time.time()
    conv = []
    it = 0
    u_n = u.copy()
    nx, ny = np.shape(u)
    l, d_n, up = help.CM(ny)
    B = np.zeros(nx)

    if method == 'column':
        while True:
            it = it + 1
            for j in range(1,ny-1):
                B[1:-1] = - u_n[1:-1,j+1] - u_n[1:-1,j-1] + f[1:-1,j]*h**2
                u_n[1:-1,j] = help.TDMA(B[1:-1], l, d_n, ny, up)

            err = help.error(u_n, u_a)

            conv = np.concatenate((conv, [err]))

            if err < max_err:
                break

            if it >= max_it:
                break

    if method == 'row':
        while True:
            it = it + 1
            for i in range(1,nx-1):
                B[1:-1] = - u_n[i+1,1:-1] - u_n[i-1,1:-1] + f[i,1:-1]*h**2
                u_n[i,1:-1] = help.TDMA(B[1:-1], l, d_n, nx, up)

            err = help.error(u_n, u_a)

            conv = np.concatenate((conv, [err]))

            if err < max_err:
                break

            if it >= max_it:
                break

    if method == 'adi':
        while True:
            it = it + 1
            for j in range(1,ny-1):
                B[1:-1] = - u_n[1:-1,j+1] - u_n[1:-1,j-1] + f[1:-1,j]*h**2
                u_n[1:-1,j] = help.TDMA(B[1:-1], l, d_n, ny, up)

            for i in range(1,nx-1):
                B[1:-1] = - u_n[i+1,1:-1] - u_n[i-1,1:-1] + f[i,1:-1]*h**2
                u_n[1:-1,j] = help.TDMA(B[1:-1], l, d_n, nx, up)

            err = help.error(u_n, u_a)

            conv = np.concatenate((conv, [err]))

            if err < max_err:
                break

            if it >= max_it:
                break

    t = time.time() - t

    print('Computation time: ' + ('%.5f' %t) + 's')
    print('Iterations: ', it)
    print('Maximum error: ' + ('%.4f' %err))
    help.plt2d(conv, it, 'Line Gauss-Seidel method')

    return u_n, it, conv, t

def line_sor(u, u_a, f, h, max_err, max_it, w, method):
    t = time.time()
    conv = []
    it = 0
    u_n = u.copy()
    nx, ny = np.shape(u)
    l, d_n, up = help.CMR(ny-2, w)
    B = np.zeros(nx)

    if method == 'column':
        while True:
            it = it + 1
            for j in range(1,ny-1):
                B[1:-1] = - 4*(1-w)*u_n[1:-1,j] - w*(u_n[1:-1,j+1] + u_n[1:-1,j-1] - f[1:-1,j]*h**2)
                u_n[1:-1,j] = help.TDMA(B[1:-1], l, d_n, ny, up)

            err = help.error(u_n, u_a)

            conv = np.concatenate((conv, [err]))

            if err < max_err:
                break

            if it >= max_it:
                break

    if method == 'row':
        while True:
            it = it + 1
            for i in range(1,nx-1):
                B[1:-1] = -4*(1-w)*u_n[i,1:-1] - w*(u_n[i+1,1:-1] + u_n[i-1,1:-1] - f[i,1:-1]*h**2)
                u_n[i,1:-1] = help.TDMA(B[1:-1], l, d_n, nx, up)

            err = help.error(u_n, u_a)

            conv = np.concatenate((conv, [err]))

            if err < max_err:
                break

            if it >= max_it:
                break

    if method == 'adi':
        while True:
            it = it + 1
            for j in range(1,ny-1):
                B[1:-1] = - 4*(1-w)*u_n[1:-1,j] - w*(u_n[1:-1,j+1] + u_n[1:-1,j-1] - f[1:-1,j]*h**2)
                u_n[1:-1,j] = help.TDMA(B[1:-1], l, d_n, ny, up)

            for i in range(1,nx-1):
                B[1:-1] = -4*(1-w)*u_n[i,1:-1] - w*(u_n[i+1,1:-1] + u_n[i-1,1:-1] - f[i,1:-1]*h**2)
                u_n[i,1:-1] = help.TDMA(B[1:-1], l, d_n, nx, up)

            err = help.error(u_n, u_a)

            conv = np.concatenate((conv, [err]))

            if err < max_err:
                break

            if it >= max_it:
                break

    t = time.time() - t

    print('Computation time: ' + ('%.5f' %t) + 's')
    print('Iterations: ', it)
    print('Maximum error: ' + ('%.4f' %err))
    help.plt2d(conv, it, 'Line SOR iteration method')

    return u_n, it, conv, t

def rb_gauss_seidel(u, u_a, f, h, max_err, max_it):
    t = time.time()
    conv = []
    it = 0
    u_n = u.copy()
    nx, ny = np.shape(u)

    x = np.zeros_like(u, dtype=np.int)
    x[1:-1,1:-1] = np.arange(1,(nx-2)**2 + 1).reshape(nx-2,nx-2)

    ir, jr = np.where((x>0) & (x%2 == 0))
    ib, jb = np.where((x>0) & (x%2 == 1))

    while True:
        it = it + 1

        u_n[ir,jr] = 0.25*(u_n[ir+1,jr] + u_n[ir-1,jr] + u_n[ir,jr+1] + u_n[ir,jr-1] - f[ir,jr]*h*h)
        u_n[ib,jb] = 0.25*(u_n[ib+1,jb] + u_n[ib-1,jb] + u_n[ib,jb+1] + u_n[ib,jb-1] - f[ib,jb]*h*h)


        err = help.error(u_n, u_a)

        conv = np.concatenate((conv, [err]))

        if err < max_err:
            break

        if it >= max_it:
            break

    t = time.time() - t

    print('Computation time: ' + ('%.5f' %t) + 's')
    print('Iterations: ', it)
    print('Maximum error: ' + ('%.4f' %err))
    help.plt2d(conv, it, 'Red-black Gauss-Seidel iteration method')

    return u_n, it, conv, t

def rb_sor(u, u_a, f, h, max_err, max_it, w):
    t = time.time()
    conv = []
    it = 0
    u_n = u.copy()
    nx, ny = np.shape(u)

    x = np.zeros_like(u, dtype=np.int)
    x[1:-1,1:-1] = np.arange(1,(nx-2)**2 + 1).reshape(nx-2,nx-2)

    ir, jr = np.where((x>0) & (x%2 == 0))
    ib, jb = np.where((x>0) & (x%2 == 1))

    while True:
        it = it + 1

        u_n[ir,jr] = (1-w)*u_n[ir,jr] + w*0.25*(u_n[ir+1,jr] + u_n[ir-1,jr] + u_n[ir,jr+1] + u_n[ir,jr-1] - f[ir,jr]*h*h)
        u_n[ib,jb] = (1-w)*u_n[ib,jb] + w*0.25*(u_n[ib+1,jb] + u_n[ib-1,jb] + u_n[ib,jb+1] + u_n[ib,jb-1] - f[ib,jb]*h*h)


        err = help.error(u_n, u_a)

        conv = np.concatenate((conv, [err]))

        if err < max_err:
            break

        if it >= max_it:
            break

    t = time.time() - t

    print('Computation time: ' + ('%.5f' %t) + 's')
    print('Iterations: ', it)
    print('Maximum error: ' + ('%.4f' %err))
    help.plt2d(conv, it, 'Red-black SOR iteration method')

    return u_n, it, conv, t