import os
from tqdm import tqdm
from numba import jit
import numba as nb
import math
import itertools
import sys
import numpy as np

args = sys.argv[1:]
if not args:
    print('Usage: python wp_heatmap.py inputfile')
inputfile = args[0]

with open(inputfile) as f:
    for line in f:
        line1 = line.replace(" ", "")
        line1 = line1.rstrip('\n')
        name, value = line1.split("=")
        exec(str(line), globals())
print(calcdir)

px_0 = pinit[0]
#loc = loc + 'p' + str(int(px_0)) + '/'
calcdir_full = calcdir #loc + calcdir

xedges = np.linspace(-10, 45, 350, endpoint=True)
yedges = np.linspace(-15, 15, 200, endpoint=True)
rxlist = xedges
rylist = yedges
rlist = np.array(tuple(itertools.product(rxlist, rylist)))
rxgrid = rlist[:, 0].reshape(len(rxlist), len(rylist))
rygrid = rlist[:, 1].reshape(len(rxlist), len(rylist))

def get_hist(r, state):
    pop_0 = np.real(state[0,0,:])
    pop_1 = np.real(state[1,1,:])
    H0, _,_ = np.histogram2d(r[:,0],r[:,1],bins=(xedges,yedges),weights=pop_0,density=False)
    H1, _,_ = np.histogram2d(r[:,0],r[:,1],bins=(xedges,yedges),weights=pop_1,density=False)
    if np.sum(H0) != 0:
        H0 = (H0 / np.sum(H0))*np.sum(pop_0)
    if np.sum(H1) != 0:
        H1 = (H1 / np.sum(H1))*np.sum(pop_1)
    return H0, H1

@nb.vectorize
def erf_vec(a):
    return math.erf(a)


@jit(nopython=True, fastmath=True)
def theta(x, y):
    return (np.pi / 2) * (erf_vec(Bx * x) + 1)  # (scipy.special.erf(B*x)+1)#

@jit(nopython=True, fastmath=True)
def Vc(x,y):
    return A*(1-alpha*np.exp(-1*((Bx**2)*(x**2) + (By**2)*(y**2))))

@jit(nopython=True, fastmath=True)
def phi(x, y):
    return W * y

@jit(nopython=True,fastmath=True)
def get_evecs_analytical(r):
    x = r[:,0]
    y = r[:,1]
    evec_0 = np.ascontiguousarray(np.zeros((2,len(r))))+0.0j
    evec_1 = np.ascontiguousarray(np.zeros((2, len(r)))) + 0.0j
    evec_0[0,:] = np.cos(theta(x,y)/2)*np.exp(1.0j*phi(x,y))#nan_num(-np.exp(1.0j*phi(x,y))*np.cos(theta(x,y)/2)/np.sin(theta(x,y)/2))#np.array([,])
    evec_0[1,:] = -np.sin(theta(x,y)/2)#1#
    evec_1[0,:] = np.sin(theta(x,y)/2)*np.exp(1.0j*phi(x,y))#nan_num(np.exp(1.0j*phi(x,y))*np.sin(theta(x,y)/2)/np.cos(theta(x,y)/2))#
    evec_1[1,:] = np.cos(theta(x,y)/2)#1#
    evecs_out = np.ascontiguousarray(np.zeros((2,2,len(r))))+0.0j
    evals_out = np.zeros((2,len(r)))
    evecs_out[:,1,:] = evec_1
    evecs_out[:,0,:] = evec_0
    evals_out[0,:] = -Vc(x,y)
    evals_out[1,:] = Vc(x,y)
    return evals_out, evecs_out

def rho_adb_to_db(rho_adb, evecs):
    rho_db = np.zeros_like(rho_adb)
    for n in range(np.shape(rho_adb)[-1]):
        rho_db[:,:,n] = np.matmul(evecs[:,:,n],np.matmul(rho_adb[:,:,n],np.conj(np.transpose(evecs[:,:,n]))))
    return rho_db

run_num = np.load(calcdir + '/run_num.npy')
for n in np.arange(0, run_num + 1):
    if n == 0:
        p_out = np.load(calcdir + '/p_' + str(n) + '.npy')
        r_out = np.load(calcdir + '/r_' + str(n) + '.npy')
        rho_adb_out = np.load(calcdir + '/rho_adb_' + str(n) + '.npy')
    else:
        p_n = np.load(calcdir + '/p_' + str(n) + '.npy')
        r_n = np.load(calcdir + '/r_' + str(n) + '.npy')
        rho_adb_n = np.load(calcdir + '/rho_adb_' + str(n) + '.npy')
        p_out = np.concatenate((p_out, p_n), axis=1)
        r_out = np.concatenate((r_out, r_n), axis=1)
        rho_adb_out = np.concatenate((rho_adb_out, rho_adb_n), axis=3)
num_points = np.shape(r_out)[1]
r_out = r_out
p_out = p_out
tdat = np.load(calcdir + '/tdat.npy')
if not(os.path.exists('./heatmap/')):
    os.mkdir('./heatmap/')

heatmap_dir = './heatmap'
np.savetxt(heatmap_dir + '/tdat.csv',tdat)
np.savetxt(heatmap_dir + '/rxgrid.csv',rxgrid)
np.savetxt(heatmap_dir + '/rygrid.csv',rygrid)
for t_ind in tqdm(range(len(tdat))):
    r = r_out[t_ind]
    rho_adb = rho_adb_out[t_ind]
    evals, evecs = get_evecs_analytical(r)
    rho_db = rho_adb_to_db(rho_adb, evecs)
    num = int(t_ind)
    rho_db_0_grid, rho_db_1_grid = get_hist(r, rho_db)
    rho_adb_0_grid, rho_adb_1_grid = get_hist(r, rho_adb)
    np.savetxt(heatmap_dir + '/rho_db_0_' + str(num) + '.csv', rho_db_0_grid)
    np.savetxt(heatmap_dir + '/rho_db_1_' + str(num) + '.csv', rho_db_1_grid)
    np.savetxt(heatmap_dir + '/rho_adb_0_' + str(num) + '.csv', rho_adb_0_grid)
    np.savetxt(heatmap_dir + '/rho_adb_1_' + str(num) + '.csv', rho_adb_1_grid)