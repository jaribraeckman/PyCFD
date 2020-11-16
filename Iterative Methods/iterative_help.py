import matplotlib.pyplot as plt
import numpy as np

# Compute maximum absolute error between computed & analytical solution
def error(u,u_a):
    error = u - u_a
    err = (abs(error)).max()

    return err


# Plot 2D graph of maxmimum error vs. iteration
def plt2d(err, it, title):
    fig = plt.figure()
    fig.add_subplot(111)
    plt.plot(np.arange(1, it + 1), err)
    plt.title(title), plt.xlabel('iterations'), plt.ylabel('maximum error')

# Create tridiagonal elements for thomas algorithm
def CM(n):
    # build coefficients matrix A (AX = B)
    d = -4*np.ones(n) # main diagonal
    l = np.ones(n-1) # lower diagonal
    up = np.ones(n-1) # upper diagonal

    d_n = d.copy() # modified diagonal
    # d'[i] = d[i] - (l[i-1]/d[i-1])*up[i-1]
    for i in range(1,n):
        d_n[i] = d[i] - up[i-1]*l[i-1]/d_n[i-1]

    return l, d_n, up

# thomas algorithm
def TDMA(B, l, d_n, n, up):
    n = np.size(B)

    # forward elimination of lower-diagonal elements
    for i in range(1,N):
        B[i] = B[i] - B[i-1]*l[i-1]/d_n[i-1]

    X = np.zeros_like(B)

    # backward substitution
    X[-1] = B[-1]/d_n[-1]
    for i in range(n-2,-1,-1):
        X[i] = (B[i] - up[i]*X[i+1])/d_n[i]

    return X