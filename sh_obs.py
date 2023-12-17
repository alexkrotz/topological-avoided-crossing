import numpy as np
import itertools
import glob
import os
import sys
import scipy
import scipy.special

args = sys.argv[1:]
if not args:
    print('Usage: python sh_obs.py data_dir')
data_dir = args[0]
def get_vals_sh(inputfile, loc):
    with open(inputfile) as f:
        for line in f:
            line1 = line.replace(" ", "")
            line1 = line1.rstrip('\n')
            name, value = line1.split("=")
            exec(str(line), globals())
    px_0 = pinit[0]
    loc = loc + 'p'+str(int(px_0))+'/'
    calcdir_full = loc + calcdir
    run_num = np.load(calcdir_full + '/run_num.npy')
    for n in np.arange(0, run_num + 1):
        if n == 0:
            p_adb_out = np.load(calcdir_full + '/p_adb_' + str(n) + '.npy')
            p_db_out = np.load(calcdir_full + '/p_db_' + str(n) + '.npy')
            pop_adb_out = np.load(calcdir_full + '/pop_adb_' + str(n) + '.npy')
            pop_db_out = np.load(calcdir_full + '/pop_db_' + str(n) + '.npy')
            rho_db_0_out = np.load(calcdir_full + '/db_0_hist_' + str(n) + '.npy')
            rho_db_1_out = np.load(calcdir_full + '/db_1_hist_' + str(n) + '.npy')
            Ec_out = np.load(calcdir_full + '/Ec_' + str(run_num) + '.npy')
            Eq_out = np.load(calcdir_full + '/Eq_' + str(run_num) + '.npy')
        else:
            p_adb_out += np.load(calcdir_full + '/p_adb_' + str(n) + '.npy')
            p_db_out += np.load(calcdir_full + '/p_db_' + str(n) + '.npy')
            pop_adb_out += np.load(calcdir_full + '/pop_adb_' + str(n) + '.npy')
            pop_db_out += np.load(calcdir_full + '/pop_db_' + str(n) + '.npy')
            rho_db_0_out += np.load(calcdir_full + '/db_0_hist_' + str(n) + '.npy')
            rho_db_1_out += np.load(calcdir_full + '/db_1_hist_' + str(n) + '.npy')
            Ec_out += np.load(calcdir_full + '/Ec_' + str(run_num) + '.npy')
            Eq_out += np.load(calcdir_full + '/Eq_' + str(run_num) + '.npy')

    num_points = np.sum(rho_db_1_out[:, :, 0] + rho_db_0_out[:, :, 0])
    p_adb_out = p_adb_out / num_points
    p_db_out = p_db_out / num_points
    pop_adb_out = pop_adb_out / num_points
    pop_db_out = pop_db_out / num_points
    Ec_out = Ec_out / num_points
    Eq_out = Eq_out / num_points
    #print(Ec_out + Eq_out)
    pop_db_0 = pop_db_out[0]
    pop_db_1 = pop_db_out[1]
    pop_adb_0 = pop_adb_out[0]
    pop_adb_1 = pop_adb_out[1]
    px0 = p_db_out[0, -1]
    py0 = p_db_out[1, -1]
    px1 = p_db_out[2, -1]
    py1 = p_db_out[3, -1]
    px0_adb = p_adb_out[0, -1]
    py0_adb = p_adb_out[1, -1]
    px1_adb = p_adb_out[2, -1]
    py1_adb = p_adb_out[3, -1]
    if num_points != 10000:
        print(calcdir, num_points)
    tdat = np.load(calcdir_full + '/tdat.npy')
    lastpop_m1_0=pop_db_0[-2]
    lastpop_0=pop_db_0[-1]
    lastpop_m1_1=pop_db_1[-2]
    lastpop_1=pop_db_1[-1]
    if np.abs(lastpop_m1_0 - lastpop_0) > 1e-4 or np.abs(lastpop_m1_1 - lastpop_1) > 1e-4:
        print('ERROR not stable', np.abs(lastpop_m1_0 - lastpop_0), np.abs(lastpop_m1_1 - lastpop_1), inputfile)

    return pinit[0], lastpop_0, lastpop_1, px1, py1, px0, py0 #, rx1, ry1, rx0, ry0  # px1 + px0, py1 + py0

inputfiles_sh = glob.glob('./'+data_dir+'/p*/FSSH_*.in')
if len(inputfiles_sh) > 0:
    print('Found ' + data_dir +' files...')
else:
    print('Data not found')
    exit()

px_list_sh = np.zeros((len(inputfiles_sh)))
pop0_list_sh = np.zeros((len(inputfiles_sh)))
pop1_list_sh = np.zeros((len(inputfiles_sh)))
tpx1_list_sh = np.zeros((len(inputfiles_sh)))
tpy1_list_sh = np.zeros((len(inputfiles_sh)))
tpx0_list_sh = np.zeros((len(inputfiles_sh)))
tpy0_list_sh = np.zeros((len(inputfiles_sh)))
n=0
for file in inputfiles_sh:
    print('loading '+file)
    px_list_sh[n], pop0_list_sh[n], pop1_list_sh[n], tpx1_list_sh[n], tpy1_list_sh[n], tpx0_list_sh[n], tpy0_list_sh[n]= get_vals_sh(file,'./'+data_dir + '/')
    n += 1
px_sort = px_list_sh.argsort()
px_list_sh = px_list_sh[px_sort]
pop0_list_sh = pop0_list_sh[px_sort]
pop1_list_sh = pop1_list_sh[px_sort]
tpx1_list_sh = tpx1_list_sh[px_sort]
tpy1_list_sh = tpy1_list_sh[px_sort]
tpx0_list_sh = tpx0_list_sh[px_sort]
tpy0_list_sh = tpy0_list_sh[px_sort]

if not(os.path.exists('./data/')):
    os.mkdir('./data/')
filename = data_dir.replace(".","")
np.savetxt('./data/px_list_'+filename+'_sh.csv',px_list_sh)
np.savetxt('./data/pop0_list_'+filename+'_sh.csv',pop0_list_sh)
np.savetxt('./data/pop1_list_'+filename+'_sh.csv',pop1_list_sh)
np.savetxt('./data/tpx1_list_'+filename+'_sh.csv',tpx1_list_sh)
np.savetxt('./data/tpy1_list_'+filename+'_sh.csv',tpy1_list_sh)
np.savetxt('./data/tpx0_list_'+filename+'_sh.csv',tpx0_list_sh)
np.savetxt('./data/tpy0_list_'+filename+'_sh.csv',tpy0_list_sh)

exit()
## old code below ###

save_01 = False
save_005=False
save_002=False
save_001=False

inputfiles_sh_A01 = glob.glob('./A0.1/p*/FSSH_*.in')
if len(inputfiles_sh_A01) > 0:
    print('Found A = 0.1 files...')
    save_01 = True
px_list_A01_sh = np.zeros((len(inputfiles_sh_A01)))
pop0_list_A01_sh = np.zeros((len(inputfiles_sh_A01)))
pop1_list_A01_sh = np.zeros((len(inputfiles_sh_A01)))
tpx1_list_A01_sh = np.zeros((len(inputfiles_sh_A01)))
tpy1_list_A01_sh = np.zeros((len(inputfiles_sh_A01)))
tpx0_list_A01_sh = np.zeros((len(inputfiles_sh_A01)))
tpy0_list_A01_sh = np.zeros((len(inputfiles_sh_A01)))
n=0
for file in inputfiles_sh_A01:
    print('loading '+file)
    px_list_A01_sh[n], pop0_list_A01_sh[n], pop1_list_A01_sh[n], tpx1_list_A01_sh[n], tpy1_list_A01_sh[n], tpx0_list_A01_sh[n], tpy0_list_A01_sh[n]= get_vals_sh(file,'./A0.1/')
    n += 1
px_sort = px_list_A01_sh.argsort()
px_list_A01_sh = px_list_A01_sh[px_sort]
pop0_list_A01_sh = pop0_list_A01_sh[px_sort]
pop1_list_A01_sh = pop1_list_A01_sh[px_sort]
tpx1_list_A01_sh = tpx1_list_A01_sh[px_sort]
tpy1_list_A01_sh = tpy1_list_A01_sh[px_sort]
tpx0_list_A01_sh = tpx0_list_A01_sh[px_sort]
tpy0_list_A01_sh = tpy0_list_A01_sh[px_sort]
inputfiles_sh_A005 = glob.glob('./A0.05/p*/FSSH_*.in')
if len(inputfiles_sh_A005) > 0:
    print('Found A = 0.05 files...')
    save_005 = True
px_list_A005_sh = np.zeros((len(inputfiles_sh_A005)))
pop0_list_A005_sh = np.zeros((len(inputfiles_sh_A005)))
pop1_list_A005_sh = np.zeros((len(inputfiles_sh_A005)))
tpx1_list_A005_sh = np.zeros((len(inputfiles_sh_A005)))
tpy1_list_A005_sh = np.zeros((len(inputfiles_sh_A005)))
tpx0_list_A005_sh = np.zeros((len(inputfiles_sh_A005)))
tpy0_list_A005_sh = np.zeros((len(inputfiles_sh_A005)))
n=0
for file in inputfiles_sh_A005:
    print('loading ' + file)
    px_list_A005_sh[n], pop0_list_A005_sh[n], pop1_list_A005_sh[n], tpx1_list_A005_sh[n], tpy1_list_A005_sh[n], tpx0_list_A005_sh[n], tpy0_list_A005_sh[n]= get_vals_sh(file,'./A0.05/')
    n += 1
px_sort = px_list_A005_sh.argsort()
px_list_A005_sh = px_list_A005_sh[px_sort]
pop0_list_A005_sh = pop0_list_A005_sh[px_sort]
pop1_list_A005_sh = pop1_list_A005_sh[px_sort]
tpx1_list_A005_sh = tpx1_list_A005_sh[px_sort]
tpy1_list_A005_sh = tpy1_list_A005_sh[px_sort]
tpx0_list_A005_sh = tpx0_list_A005_sh[px_sort]
tpy0_list_A005_sh = tpy0_list_A005_sh[px_sort]
inputfiles_sh_A002 = glob.glob('./A0.02/p*/FSSH_*.in')
if len(inputfiles_sh_A002) > 0:
    print('Found A = 0.02 files...')
    save_002 = True
px_list_A002_sh = np.zeros((len(inputfiles_sh_A002)))
pop0_list_A002_sh = np.zeros((len(inputfiles_sh_A002)))
pop1_list_A002_sh = np.zeros((len(inputfiles_sh_A002)))
tpx1_list_A002_sh = np.zeros((len(inputfiles_sh_A002)))
tpy1_list_A002_sh = np.zeros((len(inputfiles_sh_A002)))
tpx0_list_A002_sh = np.zeros((len(inputfiles_sh_A002)))
tpy0_list_A002_sh = np.zeros((len(inputfiles_sh_A002)))
n=0
for file in inputfiles_sh_A002:
    print('loading ' + file)
    px_list_A002_sh[n], pop0_list_A002_sh[n], pop1_list_A002_sh[n], tpx1_list_A002_sh[n], tpy1_list_A002_sh[n], tpx0_list_A002_sh[n], tpy0_list_A002_sh[n]= get_vals_sh(file,'./A0.02/')
    n += 1
px_sort = px_list_A002_sh.argsort()
px_list_A002_sh = px_list_A002_sh[px_sort]
pop0_list_A002_sh = pop0_list_A002_sh[px_sort]
pop1_list_A002_sh = pop1_list_A002_sh[px_sort]
tpx1_list_A002_sh = tpx1_list_A002_sh[px_sort]
tpy1_list_A002_sh = tpy1_list_A002_sh[px_sort]
tpx0_list_A002_sh = tpx0_list_A002_sh[px_sort]
tpy0_list_A002_sh = tpy0_list_A002_sh[px_sort]
inputfiles_sh_A001 = glob.glob('./A0.01/p*/FSSH_*.in')
if len(inputfiles_sh_A001) > 0:
    print('Found A = 0.01 files...')
    save_001 = True
px_list_A001_sh = np.zeros((len(inputfiles_sh_A001)))
pop0_list_A001_sh = np.zeros((len(inputfiles_sh_A001)))
pop1_list_A001_sh = np.zeros((len(inputfiles_sh_A001)))
tpx1_list_A001_sh = np.zeros((len(inputfiles_sh_A001)))
tpy1_list_A001_sh = np.zeros((len(inputfiles_sh_A001)))
tpx0_list_A001_sh = np.zeros((len(inputfiles_sh_A001)))
tpy0_list_A001_sh = np.zeros((len(inputfiles_sh_A001)))
n=0
for file in inputfiles_sh_A001:
    print('loading ' + file)
    px_list_A001_sh[n], pop0_list_A001_sh[n], pop1_list_A001_sh[n], tpx1_list_A001_sh[n], tpy1_list_A001_sh[n], tpx0_list_A001_sh[n], tpy0_list_A001_sh[n]= get_vals_sh(file,'./A0.01/')
    n += 1
px_sort = px_list_A001_sh.argsort()
px_list_A001_sh = px_list_A001_sh[px_sort]
pop0_list_A001_sh = pop0_list_A001_sh[px_sort]
pop1_list_A001_sh = pop1_list_A001_sh[px_sort]
tpx1_list_A001_sh = tpx1_list_A001_sh[px_sort]
tpy1_list_A001_sh = tpy1_list_A001_sh[px_sort]
tpx0_list_A001_sh = tpx0_list_A001_sh[px_sort]
tpy0_list_A001_sh = tpy0_list_A001_sh[px_sort]

os.mkdir('./data/')
if save_001:
    np.savetxt('./data/px_list_A001_sh.csv',px_list_A001_sh)
    np.savetxt('./data/pop0_list_A001_sh.csv',pop0_list_A001_sh)
    np.savetxt('./data/pop1_list_A001_sh.csv',pop1_list_A001_sh)
    np.savetxt('./data/tpx1_list_A001_sh.csv',tpx1_list_A001_sh)
    np.savetxt('./data/tpy1_list_A001_sh.csv',tpy1_list_A001_sh)
    np.savetxt('./data/tpx0_list_A001_sh.csv',tpx0_list_A001_sh)
    np.savetxt('./data/tpy0_list_A001_sh.csv',tpy0_list_A001_sh)

if save_002:
    np.savetxt('./data/px_list_A002_sh.csv',px_list_A002_sh)
    np.savetxt('./data/pop0_list_A002_sh.csv',pop0_list_A002_sh)
    np.savetxt('./data/pop1_list_A002_sh.csv',pop1_list_A002_sh)
    np.savetxt('./data/tpx1_list_A002_sh.csv',tpx1_list_A002_sh)
    np.savetxt('./data/tpy1_list_A002_sh.csv',tpy1_list_A002_sh)
    np.savetxt('./data/tpx0_list_A002_sh.csv',tpx0_list_A002_sh)
    np.savetxt('./data/tpy0_list_A002_sh.csv',tpy0_list_A002_sh)

if save_005:
    np.savetxt('./data/px_list_A005_sh.csv',px_list_A005_sh)
    np.savetxt('./data/pop0_list_A005_sh.csv',pop0_list_A005_sh)
    np.savetxt('./data/pop1_list_A005_sh.csv',pop1_list_A005_sh)
    np.savetxt('./data/tpx1_list_A005_sh.csv',tpx1_list_A005_sh)
    np.savetxt('./data/tpy1_list_A005_sh.csv',tpy1_list_A005_sh)
    np.savetxt('./data/tpx0_list_A005_sh.csv',tpx0_list_A005_sh)
    np.savetxt('./data/tpy0_list_A005_sh.csv',tpy0_list_A005_sh)

if save_01:
    np.savetxt('./data/px_list_A01_sh.csv',px_list_A01_sh)
    np.savetxt('./data/pop0_list_A01_sh.csv',pop0_list_A01_sh)
    np.savetxt('./data/pop1_list_A01_sh.csv',pop1_list_A01_sh)
    np.savetxt('./data/tpx1_list_A01_sh.csv',tpx1_list_A01_sh)
    np.savetxt('./data/tpy1_list_A01_sh.csv',tpy1_list_A01_sh)
    np.savetxt('./data/tpx0_list_A01_sh.csv',tpx0_list_A01_sh)
    np.savetxt('./data/tpy0_list_A01_sh.csv',tpy0_list_A01_sh)