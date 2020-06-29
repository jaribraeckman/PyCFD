import numpy
from matplotlib import pyplot, cm
from mpl_toolkits.mplot3d import Axes3D


def plot2D(x, y, p):
    fig = pyplot.figure(figsize=(11,7), dpi=100)
    ax = fig.gca(projection='3d')
    X, Y = numpy.meshgrid(x, y)
    surf = ax.plot_surface(X, Y, p[:], rstride=1, cstride=1, cmap=cm.viridis, linewidth=0, antialiased=False)
    ax.set_xlim(0,2)
    ax.set_ylim(0,1)
    ax.view_init(30,225)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    pyplot.show()

def plot_cavity(x,y,p,u,v):
    fig = pyplot.figure(figsize=(11,7), dpi=100)
    X,Y = numpy.meshgrid(x,y)
    # plotting the pressure field as a contour
    pyplot.contourf(X, Y, p, alpha=0.5, cmap=cm.viridis)
    pyplot.colorbar()
    # plotting the pressure field outlines
    pyplot.contour(X, Y, p, cmap=cm.viridis)
    # plotting velocity field
    pyplot.streamplot(X,Y,u,v)
    pyplot.xlabel('X')
    pyplot.ylabel('Y')
    pyplot.show()

def plot_channel(x,y,p,u,v):
    fig = pyplot.figure(figsize=(11,7), dpi=100)
    X,Y = numpy.meshgrid(x,y)
    pyplot.quiver(X[::3,::3], Y[::3,::3], u[::3,::3], v[::3,::3])
    pyplot.show()