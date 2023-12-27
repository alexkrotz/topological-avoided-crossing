import sys
import numpy as np
import itertools
import os
from tqdm import tqdm
from shutil import copyfile
from numba import jit
import numba as nb
import math
import glob

args = sys.argv[1:]
if not args:
    print('Usage: python wp_heatmap.py inputfile')
inputfile = args[0]
model = args[-1]
num_tmpfiles = len(glob.glob('inputfile.tmp-*')) + 1
tmpfile = 'inputfile.tmp-' + str(num_tmpfiles)
with open(inputfile) as f:
    for line in f:
        line1 = line.replace(" ", "")
        line1 = line1.rstrip('\n')
        name, value = line1.split("=")
        exec(str(line), globals())
copyfile(inputfile, tmpfile)
print(calcdir)

if model=='A':
    from functions_6 import *
elif model=='C':
    from functions_5 import *
elif model=='B':
    from functions_4 import *
else:
    print('model: ', model)
    exit()
px_0 = pinit[0]
#loc = loc + 'p' + str(int(px_0)) + '/'
calcdir_full = calcdir #loc + calcdir
Lx = xran[1] - xran[0]
Ly = yran[1] - yran[0]
jxlist = np.arange(0, Nx - 1 + 1, 1)  # j = 0,1,...,N-1
jylist = np.arange(0, Ny - 1 + 1, 1)
kxlist = np.concatenate((np.arange(0, Nx / 2 + 1, 1),
                         np.arange((-Nx / 2), 0))) * 2 * np.pi / Lx  # np.arange(0,Nx-1+1,1)# k = 0,1,...,N-1
kylist = np.concatenate((np.arange(0, Ny / 2 + 1, 1),
                         np.arange((-Ny / 2), 0))) * 2 * np.pi / Ly  # np.arange(0,Nx-1+1,1)# k = 0,1,...,N-1
klist = np.array(tuple(itertools.product(kxlist, kylist)))
kxgrid = klist[:, 0].reshape(len(kxlist), len(kylist))
kygrid = klist[:, 1].reshape(len(kxlist), len(kylist))
knorm = np.linalg.norm(klist, axis=1) ** 2
kgrid = knorm.reshape(len(kxlist), len(kylist))
rxlist = np.linspace(xran[0], xran[1], Nx + 1)
rylist = np.linspace(yran[0], yran[1], Ny + 1)
rlist = np.array(tuple(itertools.product(rxlist, rylist)))
rxgrid = rlist[:, 0].reshape(len(rxlist), len(rylist))
rygrid = rlist[:, 1].reshape(len(rxlist), len(rylist))
xdim = Nx + 1
ydim = Ny + 1


V_list = V_vec(rlist)
ev0 = np.zeros(len(V_list[0,0,:]),dtype=np.float64)
ev1 = np.zeros(len(V_list[0,0,:]),dtype=np.float64)
evecs = np.zeros((2,2,len(V_list[0,0,:])),dtype=complex)
H00 = np.zeros(len(V_list[0,0,:]),dtype=np.float64)
H11 = np.zeros(len(V_list[0,0,:]),dtype=np.float64)
for n in range(len(V_list[0,0,:])):
    eigvals,eigvecs = np.linalg.eigh(V_list[:,:,n])
    evecs[:,:,n] = eigvecs
    ev0[n] = eigvals[0]
    ev1[n] = eigvals[1]
    H00[n] = np.real(V_list[0,0,n])
    H11[n] = np.real(V_list[1,1,n])

def get_psi_adb(psi_db):
    psi_adb = np.zeros_like(psi_db,dtype=complex)
    for n in range(np.shape(psi_db)[-1]):
        psi_adb[:,n] = np.matmul(np.transpose(np.conj(evecs[:,:,n])),psi_db[:,n])
    return psi_adb

psi = np.load(calcdir_full + '/psi.npy')
tdat = np.load(calcdir_full + '/tdat.npy')

if not(os.path.exists('./heatmap/')):
    os.mkdir('./heatmap/')

heatmap_dir = './heatmap'
np.savetxt(heatmap_dir + '/tdat.csv',tdat)
np.savetxt(heatmap_dir + '/rxgrid.csv',rxgrid)
np.savetxt(heatmap_dir + '/rygrid.csv',rygrid)
for t_ind in tqdm(range(np.shape(psi)[0])):
    num = int(t_ind)
    state = psi[t_ind]
    state_adb = get_psi_adb(state)
    rho_db_0_grid = np.real(np.abs(state[0].reshape((xdim,ydim)))**2)
    rho_db_1_grid = np.real(np.abs(state[1].reshape((xdim,ydim)))**2)
    rho_adb_0_grid = np.real(np.abs(state_adb[0].reshape((xdim,ydim)))**2)
    rho_adb_1_grid = np.real(np.abs(state_adb[1].reshape((xdim,ydim)))**2)
    np.savetxt(heatmap_dir + '/rho_db_0_' + str(num) + '.csv', rho_db_0_grid)
    np.savetxt(heatmap_dir + '/rho_db_1_' + str(num) + '.csv', rho_db_1_grid)
    np.savetxt(heatmap_dir + '/rho_adb_0_' + str(num) + '.csv', rho_adb_0_grid)
    np.savetxt(heatmap_dir + '/rho_adb_1_' + str(num) + '.csv', rho_adb_1_grid)


