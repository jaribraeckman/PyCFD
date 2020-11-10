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