import numpy as np
from numba import jit
import numba as nb
import glob

num_tmpfiles = len(glob.glob('inputfile.tmp-*'))
tmpfile = 'inputfile.tmp-' + str(num_tmpfiles)
with open(tmpfile) as f:
    for line in f:
        line1 = line.replace(" ", "")
        line1 = line1.rstrip('\n')
        name, value = line1.split("=")
        exec(str(line), globals())

hbar = 1
sigma = 1
#mass = 1000
mass = 1
A = A*1000

@nb.vectorize
def erf_vec(a):
    return math.erf(a)
@jit(nopython=True, fastmath=True)
def theta(x, y):
    return -np.pi*(np.exp(-(Bx*(x**2) + By*(y**2))/3))


@jit(nopython=True, fastmath=True)
def dtheta_x(x, y):  # dtheta/dx
    return (2*np.pi*Bx*x/3)*np.exp(-(Bx*(x**2) + By*(y**2))/3)

@jit(nopython=True, fastmath=True)
def dtheta_y(x, y):  # dtheta/dx
    return (2*np.pi*By*y/3)*np.exp(-(Bx*(x**2) + By*(y**2))/3)

@jit(nopython=True, fastmath=True)
def phi(x, y):
    return W * y

@jit(nopython=True, fastmath=True)
def dphi_x(x,y):
    return 0*x

@jit(nopython=True, fastmath=True)
def dphi_y(x,y):
    return W*np.ones_like(y)
@jit(nopython=True, fastmath=True)
def Vc(x,y):
    return A*np.ones_like(x)


@jit(nopython=True, fastmath=True)
def V_vec(r):
    x = r[:, 0]
    y = r[:, 1]
    out_mat = np.ascontiguousarray(np.zeros((2, 2, len(x)))) + 0.0j
    out_mat[0, 0, :] = -np.cos(theta(x, y)) * Vc(x,y)
    out_mat[0, 1, :] = np.sin(theta(x, y)) * np.exp(1.0j * phi(x, y)) * Vc(x,y)
    out_mat[1, 0, :] = np.sin(theta(x, y)) * np.exp(-1.0j * phi(x, y)) * Vc(x,y)
    out_mat[1, 1, :] = np.cos(theta(x, y)) * Vc(x,y)
    return out_mat