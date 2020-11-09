import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import time

# domain
L = 1
n = 11
h = L/(n-1)
x = y = np.linspace(0,L,n)
X, Y = np.meshgrid(x,y)

# source term
f = np.zeros((n,n))
for i in range(n):
    for j in range(n):
        f[i,j] = 2*x[i]*(x[i]-1)+2*y[j]*(y[j]-1)

# or via matrix multiplication
f = 2*X*(X-1) + 2*Y*(Y-1)

