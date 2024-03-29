import numpy as np
import itertools
import glob
import os
import sys

args = sys.argv[1:]
if not args:
    print('Usage: python wp_obs.py data_dir')
data_dir = args[0]
def get_vals_wp(inputfile, loc):
    with open(inputfile) as f:
        for line in f:
            line1 = line.replace(" ", "")
            line1 = line1.rstrip('\n')
            name, value = line1.split("=")
            exec(str(line), globals())
    print(calcdir)
    px_0 = pinit[0]
    loc = loc + 'p' + str(int(px_0)) + '/'
    calcdir_full = loc + calcdir
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

    def adjust_odd(num):
        if num % 2 == 0:
            return num
        else:
            return num + 1


    r_min = 1.0
    Nx2 = adjust_odd(len(rxlist[rxlist > r_min]))
    rxmin = rxlist[len(rxlist) - Nx2 - 2]
    Lx2 = xran[1] - rxmin
    kxlist2 = np.concatenate((np.arange(0, Nx2 / 2 + 1, 1),
                             np.arange((-Nx2 / 2), 0))) * 2 * np.pi / Lx2  # np.arange(0,Nx-1+1,1)# k = 0,1,...,N-1
    klist2 = np.array(tuple(itertools.product(kxlist2, kylist)))
    kxgrid2 = klist2[:, 0].reshape(len(kxlist2), len(kylist))
    kygrid2 = klist2[:, 1].reshape(len(kxlist2), len(kylist))
    def get_tx_grid(grid):
        tx_grid = grid[rxgrid > rxmin]
        return tx_grid.reshape((int(len(tx_grid)/len(rylist)),len(rylist)))

    def get_px(wplist):
        wpgrid = get_tx_grid(wplist.reshape(Nx + 1, Ny + 1))
        wpgrid_k = np.fft.fft2(wpgrid)
        return np.real(np.sum(np.conj(wpgrid) * np.fft.ifft2(kxgrid2 * wpgrid_k)) / (np.sum(np.abs(wpgrid) ** 2)))

    def get_py(wplist):
        wpgrid = get_tx_grid(wplist.reshape(Nx + 1, Ny + 1))
        wpgrid_k = np.fft.fft2(wpgrid)
        return np.real(np.sum(np.conj(wpgrid) * np.fft.ifft2(kygrid2 * wpgrid_k)) / (np.sum(np.abs(wpgrid) ** 2)))

    def get_rx(wplist):
        wpgrid = wplist.reshape(Nx + 1, Ny + 1)
        return np.real(np.sum(np.conj(wpgrid) * rxgrid * wpgrid) / (np.sum(np.abs(wpgrid) ** 2)))

    def get_ry(wplist):
        wpgrid = wplist.reshape(Nx + 1, Ny + 1)
        return np.real(np.sum(np.conj(wpgrid) * rygrid * wpgrid) / (np.sum(np.abs(wpgrid) ** 2)))

    psi = np.load(calcdir_full + '/psi.npy') # load wavefunction
    tdat = np.load(calcdir_full + '/tdat.npy') # load time
    px1 = get_px(psi[-1][1])
    py1 = get_py(psi[-1][1])
    px0 = get_px(psi[-1][0])
    py0 = get_py(psi[-1][0])
    rx1 = get_rx(psi[-1][1])
    ry1 = get_ry(psi[-1][1])
    rx0 = get_rx(psi[-1][0])
    ry0 = get_ry(psi[-1][0])
    lastpop_0 = np.sum(np.abs(psi[-1][0]) ** 2)
    lastpop_1 = np.sum(np.abs(psi[-1][1]) ** 2)
    lastpop_m1_0 = np.sum(np.abs(psi[-2][0]) ** 2)
    lastpop_m1_1 = np.sum(np.abs(psi[-2][1]) ** 2)
    # determine if the populations are stable (ie fully transmitted and not leaving the grid)
    if np.abs(lastpop_m1_0 - lastpop_0) > 1e-4 or np.abs(lastpop_m1_1 - lastpop_1) > 1e-4:
        print('ERROR not stable', np.abs(lastpop_m1_0 - lastpop_0), np.abs(lastpop_m1_1 - lastpop_1))

    return pinit[0], lastpop_0, lastpop_1, px1, py1, px0, py0, rx1, ry1, rx0, ry0  # px1 + px0, py1 + py0


inputfiles_wp = glob.glob('./'+data_dir+'/p*/WP_*.in')
if len(inputfiles_wp) > 0:
    print('Found ' + data_dir +' files...')
else:
    print('Data not found')
    exit()
px_list_wp = np.zeros((len(inputfiles_wp)))
pop0_list_wp = np.zeros((len(inputfiles_wp)))
pop1_list_wp = np.zeros((len(inputfiles_wp)))
tpx1_list_wp = np.zeros((len(inputfiles_wp)))
tpy1_list_wp = np.zeros((len(inputfiles_wp)))
tpx0_list_wp = np.zeros((len(inputfiles_wp)))
tpy0_list_wp = np.zeros((len(inputfiles_wp)))
trx1_list_wp = np.zeros((len(inputfiles_wp)))
try1_list_wp = np.zeros((len(inputfiles_wp)))
trx0_list_wp = np.zeros((len(inputfiles_wp)))
try0_list_wp = np.zeros((len(inputfiles_wp)))

n=0
for file in inputfiles_wp:
    px_list_wp[n], pop0_list_wp[n], pop1_list_wp[n], tpx1_list_wp[n], tpy1_list_wp[n], tpx0_list_wp[n], tpy0_list_wp[n],\
    trx1_list_wp[n], try1_list_wp[n], trx0_list_wp[n], try0_list_wp[n]= get_vals_wp(file,'./'+data_dir+'/')
    n += 1
px_sort = px_list_wp.argsort()
px_list_wp = px_list_wp[px_sort]
pop0_list_wp = pop0_list_wp[px_sort]
pop1_list_wp = pop1_list_wp[px_sort]
tpx1_list_wp = tpx1_list_wp[px_sort]
tpy1_list_wp = tpy1_list_wp[px_sort]
tpx0_list_wp = tpx0_list_wp[px_sort]
tpy0_list_wp = tpy0_list_wp[px_sort]
trx1_list_wp = trx1_list_wp[px_sort]
try1_list_wp = try1_list_wp[px_sort]
trx0_list_wp = trx0_list_wp[px_sort]
try0_list_wp = try0_list_wp[px_sort]
print(px_list_wp)
print(tpx1_list_wp)
print(tpx0_list_wp)
if not(os.path.exists('./data/')):
    os.mkdir('./data/')
filename = data_dir.replace(".","")
np.savetxt('./data/px_list_'+filename+'_wp.csv',px_list_wp) # initial x direction momentum
np.savetxt('./data/pop0_list_'+filename+'_wp.csv',pop0_list_wp) # transmitted population on lower diabatic surface
np.savetxt('./data/pop1_list_'+filename+'_wp.csv',pop1_list_wp) # transmitted population on upper diabatic surface
np.savetxt('./data/tpx1_list_'+filename+'_wp.csv',tpx1_list_wp) # transmitted p_x on upper diabatic surface
np.savetxt('./data/tpy1_list_'+filename+'_wp.csv',tpy1_list_wp) # transmitted p_y on upper diabatic surface
np.savetxt('./data/tpx0_list_'+filename+'_wp.csv',tpx0_list_wp) # transmitted p_x on lower diabatic surface
np.savetxt('./data/tpy0_list_'+filename+'_wp.csv',tpy0_list_wp) # transmitted p_y on lower diabatic surface
np.savetxt('./data/trx1_list_'+filename+'_wp.csv',trx1_list_wp) # transmitted r_x on upper diabatic surface
np.savetxt('./data/try1_list_'+filename+'_wp.csv',try1_list_wp) # transmitted r_y on upper diabatic surface
np.savetxt('./data/trx0_list_'+filename+'_wp.csv',trx0_list_wp) # transmitted r_x on lower diabatic surface
np.savetxt('./data/try0_list_'+filename+'_wp.csv',try0_list_wp) # transmitted r_y on lower diabatic surface


exit()
# below is deprecated
inputfiles_wp_A01 = glob.glob('./A0.1/p*/WP_*.in')
save_01 = False
if len(inputfiles_wp_A01) > 0:
    print('Found A = 0.1 files...')
    save_01 = True
px_list_A01_wp = np.zeros((len(inputfiles_wp_A01)))
pop0_list_A01_wp = np.zeros((len(inputfiles_wp_A01)))
pop1_list_A01_wp = np.zeros((len(inputfiles_wp_A01)))
tpx1_list_A01_wp = np.zeros((len(inputfiles_wp_A01)))
tpy1_list_A01_wp = np.zeros((len(inputfiles_wp_A01)))
tpx0_list_A01_wp = np.zeros((len(inputfiles_wp_A01)))
tpy0_list_A01_wp = np.zeros((len(inputfiles_wp_A01)))
trx1_list_A01_wp = np.zeros((len(inputfiles_wp_A01)))
try1_list_A01_wp = np.zeros((len(inputfiles_wp_A01)))
trx0_list_A01_wp = np.zeros((len(inputfiles_wp_A01)))
try0_list_A01_wp = np.zeros((len(inputfiles_wp_A01)))

n=0
for file in inputfiles_wp_A01:
    px_list_A01_wp[n], pop0_list_A01_wp[n], pop1_list_A01_wp[n], tpx1_list_A01_wp[n], tpy1_list_A01_wp[n], tpx0_list_A01_wp[n], tpy0_list_A01_wp[n],\
    trx1_list_A01_wp[n], try1_list_A01_wp[n], trx0_list_A01_wp[n], try0_list_A01_wp[n]= get_vals_wp(file,'./A0.1/')
    n += 1
px_sort = px_list_A01_wp.argsort()
px_list_A01_wp = px_list_A01_wp[px_sort]
pop0_list_A01_wp = pop0_list_A01_wp[px_sort]
pop1_list_A01_wp = pop1_list_A01_wp[px_sort]
tpx1_list_A01_wp = tpx1_list_A01_wp[px_sort]
tpy1_list_A01_wp = tpy1_list_A01_wp[px_sort]
tpx0_list_A01_wp = tpx0_list_A01_wp[px_sort]
tpy0_list_A01_wp = tpy0_list_A01_wp[px_sort]
trx1_list_A01_wp = trx1_list_A01_wp[px_sort]
try1_list_A01_wp = try1_list_A01_wp[px_sort]
trx0_list_A01_wp = trx0_list_A01_wp[px_sort]
try0_list_A01_wp = try0_list_A01_wp[px_sort]
print(px_list_A01_wp)
print(tpx1_list_A01_wp)
print(tpx0_list_A01_wp)
inputfiles_wp_A005 = glob.glob('./A0.05/p*/WP_*.in')
save_005 = False
if len(inputfiles_wp_A005) > 0:
    print('Found A = 0.05 files...')
    save_005 = True
px_list_A005_wp = np.zeros((len(inputfiles_wp_A005)))
pop0_list_A005_wp = np.zeros((len(inputfiles_wp_A005)))
pop1_list_A005_wp = np.zeros((len(inputfiles_wp_A005)))
tpx1_list_A005_wp = np.zeros((len(inputfiles_wp_A005)))
tpy1_list_A005_wp = np.zeros((len(inputfiles_wp_A005)))
tpx0_list_A005_wp = np.zeros((len(inputfiles_wp_A005)))
tpy0_list_A005_wp = np.zeros((len(inputfiles_wp_A005)))
trx1_list_A005_wp = np.zeros((len(inputfiles_wp_A005)))
try1_list_A005_wp = np.zeros((len(inputfiles_wp_A005)))
trx0_list_A005_wp = np.zeros((len(inputfiles_wp_A005)))
try0_list_A005_wp = np.zeros((len(inputfiles_wp_A005)))
n=0
for file in inputfiles_wp_A005:
    px_list_A005_wp[n], pop0_list_A005_wp[n], pop1_list_A005_wp[n], tpx1_list_A005_wp[n], tpy1_list_A005_wp[n], tpx0_list_A005_wp[n], tpy0_list_A005_wp[n],\
    trx1_list_A005_wp[n], try1_list_A005_wp[n], trx0_list_A005_wp[n], try0_list_A005_wp[n]= get_vals_wp(file,'./A0.05/')
    n += 1
px_sort = px_list_A005_wp.argsort()
px_list_A005_wp = px_list_A005_wp[px_sort]
pop0_list_A005_wp = pop0_list_A005_wp[px_sort]
pop1_list_A005_wp = pop1_list_A005_wp[px_sort]
tpx1_list_A005_wp = tpx1_list_A005_wp[px_sort]
tpy1_list_A005_wp = tpy1_list_A005_wp[px_sort]
tpx0_list_A005_wp = tpx0_list_A005_wp[px_sort]
tpy0_list_A005_wp = tpy0_list_A005_wp[px_sort]
trx1_list_A005_wp = trx1_list_A005_wp[px_sort]
try1_list_A005_wp = try1_list_A005_wp[px_sort]
trx0_list_A005_wp = trx0_list_A005_wp[px_sort]
try0_list_A005_wp = try0_list_A005_wp[px_sort]

inputfiles_wp_A002 = glob.glob('./A0.02/p*/WP_*.in')
save_002 = False
if len(inputfiles_wp_A002) > 0:
    print('Found A = 0.02 files...')
    save_002 = True
px_list_A002_wp = np.zeros((len(inputfiles_wp_A002)))
pop0_list_A002_wp = np.zeros((len(inputfiles_wp_A002)))
pop1_list_A002_wp = np.zeros((len(inputfiles_wp_A002)))
tpx1_list_A002_wp = np.zeros((len(inputfiles_wp_A002)))
tpy1_list_A002_wp = np.zeros((len(inputfiles_wp_A002)))
tpx0_list_A002_wp = np.zeros((len(inputfiles_wp_A002)))
tpy0_list_A002_wp = np.zeros((len(inputfiles_wp_A002)))
trx1_list_A002_wp = np.zeros((len(inputfiles_wp_A002)))
try1_list_A002_wp = np.zeros((len(inputfiles_wp_A002)))
trx0_list_A002_wp = np.zeros((len(inputfiles_wp_A002)))
try0_list_A002_wp = np.zeros((len(inputfiles_wp_A002)))
n=0
for file in inputfiles_wp_A002:
    px_list_A002_wp[n], pop0_list_A002_wp[n], pop1_list_A002_wp[n], tpx1_list_A002_wp[n], tpy1_list_A002_wp[n], tpx0_list_A002_wp[n], tpy0_list_A002_wp[n],\
    trx1_list_A002_wp[n], try1_list_A002_wp[n], trx0_list_A002_wp[n], try0_list_A002_wp[n]= get_vals_wp(file,'./A0.02/')
    n += 1
px_sort = px_list_A002_wp.argsort()
px_list_A002_wp = px_list_A002_wp[px_sort]
pop0_list_A002_wp = pop0_list_A002_wp[px_sort]
pop1_list_A002_wp = pop1_list_A002_wp[px_sort]
tpx1_list_A002_wp = tpx1_list_A002_wp[px_sort]
tpy1_list_A002_wp = tpy1_list_A002_wp[px_sort]
tpx0_list_A002_wp = tpx0_list_A002_wp[px_sort]
tpy0_list_A002_wp = tpy0_list_A002_wp[px_sort]
trx1_list_A002_wp = trx1_list_A002_wp[px_sort]
try1_list_A002_wp = try1_list_A002_wp[px_sort]
trx0_list_A002_wp = trx0_list_A002_wp[px_sort]
try0_list_A002_wp = try0_list_A002_wp[px_sort]

inputfiles_wp_A001 = glob.glob('./A0.01/p*/WP_*.in')
save_001 = False
if len(inputfiles_wp_A001) > 0:
    print('Found A = 0.01 files...')
    save_001 = True
px_list_A001_wp = np.zeros((len(inputfiles_wp_A001)))
pop0_list_A001_wp = np.zeros((len(inputfiles_wp_A001)))
pop1_list_A001_wp = np.zeros((len(inputfiles_wp_A001)))
tpx1_list_A001_wp = np.zeros((len(inputfiles_wp_A001)))
tpy1_list_A001_wp = np.zeros((len(inputfiles_wp_A001)))
tpx0_list_A001_wp = np.zeros((len(inputfiles_wp_A001)))
tpy0_list_A001_wp = np.zeros((len(inputfiles_wp_A001)))
trx1_list_A001_wp = np.zeros((len(inputfiles_wp_A001)))
try1_list_A001_wp = np.zeros((len(inputfiles_wp_A001)))
trx0_list_A001_wp = np.zeros((len(inputfiles_wp_A001)))
try0_list_A001_wp = np.zeros((len(inputfiles_wp_A001)))
n=0
for file in inputfiles_wp_A001:
    px_list_A001_wp[n], pop0_list_A001_wp[n], pop1_list_A001_wp[n], tpx1_list_A001_wp[n], tpy1_list_A001_wp[n], tpx0_list_A001_wp[n], tpy0_list_A001_wp[n],\
    trx1_list_A001_wp[n], try1_list_A001_wp[n], trx0_list_A001_wp[n], try0_list_A001_wp[n]= get_vals_wp(file,'./A0.01/')
    n += 1
px_sort = px_list_A001_wp.argsort()
px_list_A001_wp = px_list_A001_wp[px_sort]
pop0_list_A001_wp = pop0_list_A001_wp[px_sort]
pop1_list_A001_wp = pop1_list_A001_wp[px_sort]
tpx1_list_A001_wp = tpx1_list_A001_wp[px_sort]
tpy1_list_A001_wp = tpy1_list_A001_wp[px_sort]
tpx0_list_A001_wp = tpx0_list_A001_wp[px_sort]
tpy0_list_A001_wp = tpy0_list_A001_wp[px_sort]
trx1_list_A001_wp = trx1_list_A001_wp[px_sort]
try1_list_A001_wp = try1_list_A001_wp[px_sort]
trx0_list_A001_wp = trx0_list_A001_wp[px_sort]
try0_list_A001_wp = try0_list_A001_wp[px_sort]
os.mkdir('./data/')
if save_001:
    np.savetxt('./data/px_list_A001_wp.csv',px_list_A001_wp)
    np.savetxt('./data/pop0_list_A001_wp.csv',pop0_list_A001_wp)
    np.savetxt('./data/pop1_list_A001_wp.csv',pop1_list_A001_wp)
    np.savetxt('./data/tpx1_list_A001_wp.csv',tpx1_list_A001_wp)
    np.savetxt('./data/tpy1_list_A001_wp.csv',tpy1_list_A001_wp)
    np.savetxt('./data/tpx0_list_A001_wp.csv',tpx0_list_A001_wp)
    np.savetxt('./data/tpy0_list_A001_wp.csv',tpy0_list_A001_wp)
    np.savetxt('./data/trx1_list_A001_wp.csv',trx1_list_A001_wp)
    np.savetxt('./data/try1_list_A001_wp.csv',try1_list_A001_wp)
    np.savetxt('./data/trx0_list_A001_wp.csv',trx0_list_A001_wp)
    np.savetxt('./data/try0_list_A001_wp.csv',try0_list_A001_wp)

if save_002:
    np.savetxt('./data/px_list_A002_wp.csv',px_list_A002_wp)
    np.savetxt('./data/pop0_list_A002_wp.csv',pop0_list_A002_wp)
    np.savetxt('./data/pop1_list_A002_wp.csv',pop1_list_A002_wp)
    np.savetxt('./data/tpx1_list_A002_wp.csv',tpx1_list_A002_wp)
    np.savetxt('./data/tpy1_list_A002_wp.csv',tpy1_list_A002_wp)
    np.savetxt('./data/tpx0_list_A002_wp.csv',tpx0_list_A002_wp)
    np.savetxt('./data/tpy0_list_A002_wp.csv',tpy0_list_A002_wp)
    np.savetxt('./data/trx1_list_A002_wp.csv',trx1_list_A002_wp)
    np.savetxt('./data/try1_list_A002_wp.csv',try1_list_A002_wp)
    np.savetxt('./data/trx0_list_A002_wp.csv',trx0_list_A002_wp)
    np.savetxt('./data/try0_list_A002_wp.csv',try0_list_A002_wp)

if save_005:
    np.savetxt('./data/px_list_A005_wp.csv',px_list_A005_wp)
    np.savetxt('./data/pop0_list_A005_wp.csv',pop0_list_A005_wp)
    np.savetxt('./data/pop1_list_A005_wp.csv',pop1_list_A005_wp)
    np.savetxt('./data/tpx1_list_A005_wp.csv',tpx1_list_A005_wp)
    np.savetxt('./data/tpy1_list_A005_wp.csv',tpy1_list_A005_wp)
    np.savetxt('./data/tpx0_list_A005_wp.csv',tpx0_list_A005_wp)
    np.savetxt('./data/tpy0_list_A005_wp.csv',tpy0_list_A005_wp)
    np.savetxt('./data/trx1_list_A005_wp.csv',trx1_list_A005_wp)
    np.savetxt('./data/try1_list_A005_wp.csv',try1_list_A005_wp)
    np.savetxt('./data/trx0_list_A005_wp.csv',trx0_list_A005_wp)
    np.savetxt('./data/try0_list_A005_wp.csv',try0_list_A005_wp)

if save_01:
    np.savetxt('./data/px_list_A01_wp.csv',px_list_A01_wp)
    np.savetxt('./data/pop0_list_A01_wp.csv',pop0_list_A01_wp)
    np.savetxt('./data/pop1_list_A01_wp.csv',pop1_list_A01_wp)
    np.savetxt('./data/tpx1_list_A01_wp.csv',tpx1_list_A01_wp)
    np.savetxt('./data/tpy1_list_A01_wp.csv',tpy1_list_A01_wp)
    np.savetxt('./data/tpx0_list_A01_wp.csv',tpx0_list_A01_wp)
    np.savetxt('./data/tpy0_list_A01_wp.csv',tpy0_list_A01_wp)
    np.savetxt('./data/trx1_list_A01_wp.csv',trx1_list_A01_wp)
    np.savetxt('./data/try1_list_A01_wp.csv',try1_list_A01_wp)
    np.savetxt('./data/trx0_list_A01_wp.csv',trx0_list_A01_wp)
    np.savetxt('./data/try0_list_A01_wp.csv',try0_list_A01_wp)