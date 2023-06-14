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


run_num = np.load(calcdir + '/run_num.npy')
for n in np.arange(0, run_num + 1):
    if n == 0:
        rho_db_0_out = np.load(calcdir + '/db_0_hist_' + str(n) + '.npy')
        rho_db_1_out = np.load(calcdir + '/db_1_hist_' + str(n) + '.npy')
        rho_adb_0_out = np.load(calcdir + '/adb_0_hist_' + str(n) + '.npy')
        rho_adb_1_out = np.load(calcdir + '/adb_1_hist_' + str(n) + '.npy')

    else:
        rho_db_0_out += np.load(calcdir + '/db_0_hist_' + str(n) + '.npy')
        rho_db_1_out += np.load(calcdir + '/db_1_hist_' + str(n) + '.npy')
        rho_adb_0_out += np.load(calcdir + '/adb_0_hist_' + str(n) + '.npy')
        rho_adb_1_out += np.load(calcdir + '/adb_1_hist_' + str(n) + '.npy')

#print(np.shape(rho_db_0_out))
xedges = np.linspace(-10, 45, np.shape(rho_db_0_out)[0], endpoint=True)
yedges = np.linspace(-15, 15, np.shape(rho_db_0_out)[1], endpoint=True)
rxlist = xedges
rylist = yedges
rlist = np.array(tuple(itertools.product(rxlist, rylist)))
rxgrid = rlist[:, 0].reshape(len(rxlist), len(rylist))
rygrid = rlist[:, 1].reshape(len(rxlist), len(rylist))

num_points = np.sum(rho_db_1_out[:,:,0] + rho_db_0_out[:,:,0])
rho_db_0_out = rho_db_0_out/num_points
rho_db_1_out = rho_db_1_out/num_points
rho_adb_0_out = rho_adb_0_out/num_points
rho_adb_1_out = rho_adb_1_out/num_points

tdat = np.load(calcdir + '/tdat.npy')
if not(os.path.exists('./heatmap/')):
    os.mkdir('./heatmap/')

heatmap_dir = './heatmap'
np.savetxt(heatmap_dir + '/tdat.csv',tdat)
np.savetxt(heatmap_dir + '/rxgrid.csv',rxgrid)
np.savetxt(heatmap_dir + '/rygrid.csv',rygrid)
for t_ind in tqdm(range(len(tdat))):
    rho_adb_1_grid = rho_adb_1_out[:,:,t_ind]
    rho_adb_0_grid = rho_adb_0_out[:,:,t_ind]
    rho_db_1_grid = rho_db_1_out[:,:,t_ind]
    rho_db_0_grid = rho_db_0_out[:,:,t_ind]
    num = int(t_ind)
    np.savetxt(heatmap_dir + '/rho_db_0_' + str(num) + '.csv', rho_db_0_grid)
    np.savetxt(heatmap_dir + '/rho_db_1_' + str(num) + '.csv', rho_db_1_grid)
    np.savetxt(heatmap_dir + '/rho_adb_0_' + str(num) + '.csv', rho_adb_0_grid)
    np.savetxt(heatmap_dir + '/rho_adb_1_' + str(num) + '.csv', rho_adb_1_grid)