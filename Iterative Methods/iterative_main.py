import numpy as np
import iterative_methods as solver
import matplotlib.pyplot as plt

# domain
L = 1
n = 101
h = L/(n-1)
x = y = np.linspace(0,L,n)
X, Y = np.meshgrid(x,y)

# source term via matrix element-wise multiplication
f = 2*np.multiply(X,(X-1)) + 2*np.multiply(Y,(Y-1))

# analytical solution for given boundary conditions
u_a = np.multiply(np.multiply(X,Y),np.multiply((X-1),(Y-1)))

# initial guess
u0 = np.ones((n,n))/20
u0[:,-1] = 0
u0[:,0] = 0
u0[-1,:] = 0
u0[0,:] = 0 # boundary conditions

# solve with Jacobi solver
u_j, it_j, conv_j, t_j = solver.jacobi(u0, u_a, f, h, 0.01, 1000)

# plot analytical solution and initial guess
fig = plt.figure(figsize = (16,6), dpi = 50)
ax = fig.add_subplot(221, projection = '3d')
ax.plot_surface(X, Y, u_a, rstride = 5, cstride = 5)
plt.title('Analytical solution')
ax.set_xlabel('x'), ax.set_ylabel('y'), ax.set_zlabel('u'), ax.set_zlim3d(bottom = 0, top = 0.07)
ax = fig.add_subplot(222, projection = '3d')
ax.plot_surface(X, Y, u0, rstride = 5, cstride = 5)
plt.title('Initial guess')
ax.set_xlabel('x'), ax.set_ylabel('y'), ax.set_zlabel('u'), ax.set_zlim3d(bottom = 0, top = 0.07)
ax = fig.add_subplot(223, projection = '3d')
ax.plot_surface(X, Y, u_j, rstride = 5, cstride = 5)
plt.title('Numerical solution')
ax.set_xlabel('x'), ax.set_ylabel('y'), ax.set_zlabel('u'), ax.set_zlim3d(bottom = 0, top = 0.07)
plt.show()