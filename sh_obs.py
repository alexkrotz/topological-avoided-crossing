import numpy as np
import itertools
import glob
import os
import scipy
import scipy.special



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
            p_out = np.load(calcdir_full + '/p_' + str(n) + '.npy')
            r_out = np.load(calcdir_full + '/r_' + str(n) + '.npy')
            rho_adb_out = np.load(calcdir_full + '/rho_adb_' + str(n) + '.npy')
        else:
            p_n = np.load(calcdir_full + '/p_' + str(n) + '.npy')
            r_n = np.load(calcdir_full + '/r_' + str(n) + '.npy')
            rho_adb_n = np.load(calcdir_full + '/rho_adb_' + str(n) + '.npy')
            p_out = np.concatenate((p_out, p_n), axis=1)
            r_out = np.concatenate((r_out, r_n), axis=1)
            rho_adb_out = np.concatenate((rho_adb_out, rho_adb_n), axis=3)

    def phi(x, y):
        return W * y

    def theta(x, y):
        return (np.pi / 2) * (scipy.special.erf(Bx * x) + 1)  # (scipy.special.erf(B*x)+1)#

    def Vc(x, y):
        return A * (1 - alpha * np.exp(-1 * ((Bx ** 2) * (x ** 2) + (By ** 2) * (y ** 2))))

    def get_evecs_analytical(r):
        x = r[:, 0]
        y = r[:, 1]
        evec_0 = np.ascontiguousarray(np.zeros((2, len(r)))) + 0.0j
        evec_1 = np.ascontiguousarray(np.zeros((2, len(r)))) + 0.0j
        evec_0[0, :] = np.cos(theta(x, y) / 2) * np.exp(
            1.0j * phi(x, y))  # nan_num(-np.exp(1.0j*phi(x,y))*np.cos(theta(x,y)/2)/np.sin(theta(x,y)/2))#np.array([,])
        evec_0[1, :] = -np.sin(theta(x, y) / 2)  # 1#
        evec_1[0, :] = np.sin(theta(x, y) / 2) * np.exp(
            1.0j * phi(x, y))  # nan_num(np.exp(1.0j*phi(x,y))*np.sin(theta(x,y)/2)/np.cos(theta(x,y)/2))#
        evec_1[1, :] = np.cos(theta(x, y) / 2)  # 1#
        evecs_out = np.ascontiguousarray(np.zeros((2, 2, len(r)))) + 0.0j
        evals_out = np.zeros((2, len(r)))
        evecs_out[:, 1, :] = evec_1
        evecs_out[:, 0, :] = evec_0
        evals_out[0, :] = -Vc(x, y)
        evals_out[1, :] = Vc(x, y)
        return evals_out, evecs_out

    def rho_adb_to_db(rho_adb, evecs):
        rho_db = np.zeros_like(rho_adb)
        for n in range(np.shape(rho_adb)[-1]):
            rho_db[:, :, n] = np.matmul(evecs[:, :, n],
                                        np.matmul(rho_adb[:, :, n], np.conj(np.transpose(evecs[:, :, n]))))
        return rho_db

    #def get_p_rho(rho, p, N):
    #    px0 = np.real(np.sum(rho[0, 0, :] * p[:, 0] / N) / np.sum(rho[0, 0, :] / N))
    #    py0 = np.real(np.sum(rho[0, 0, :] * p[:, 1] / N) / np.sum(rho[0, 0, :] / N))
    #    px1 = np.real(np.sum(rho[1, 1, :] * p[:, 0] / N) / np.sum(rho[1, 1, :] / N))
    #    py1 = np.real(np.sum(rho[1, 1, :] * p[:, 1] / N) / np.sum(rho[1, 1, :] / N))
    #    return px0, py0, px1, py1
    r_min = 1.0

    def get_p_rho(rho, p, r, N):
        rtrans_ind = np.arange(len(r),dtype=int)[r[:,0] > r_min]
        px0 = np.real(np.sum(rho[0, 0, rtrans_ind] * p[rtrans_ind, 0] / N) / np.sum(rho[0, 0, rtrans_ind] / N))
        py0 = np.real(np.sum(rho[0, 0, rtrans_ind] * p[rtrans_ind, 1] / N) / np.sum(rho[0, 0, rtrans_ind] / N))
        px1 = np.real(np.sum(rho[1, 1, rtrans_ind] * p[rtrans_ind, 0] / N) / np.sum(rho[1, 1, rtrans_ind] / N))
        py1 = np.real(np.sum(rho[1, 1, rtrans_ind] * p[rtrans_ind, 1] / N) / np.sum(rho[1, 1, rtrans_ind] / N))
        return px0, py0, px1, py1


    def get_r_rho(rho, r, N):
        rx0 = np.real(np.sum(rho[0, 0, :] * r[:, 0] / N) / np.sum(rho[0, 0, :] / N))
        ry0 = np.real(np.sum(rho[0, 0, :] * r[:, 1] / N) / np.sum(rho[0, 0, :] / N))
        rx1 = np.real(np.sum(rho[1, 1, :] * r[:, 0] / N) / np.sum(rho[1, 1, :] / N))
        ry1 = np.real(np.sum(rho[1, 1, :] * r[:, 1] / N) / np.sum(rho[1, 1, :] / N))
        return rx0, ry0, rx1, ry1

    num_points = np.shape(r_out)[1]
    if num_points != 10000:
        print(calcdir, num_points)
    tdat = np.load(calcdir_full + '/tdat.npy')
    last_r = r_out[-1]
    rtrans_ind = np.arange(len(last_r), dtype=int)[last_r[:, 0] > r_min]
    last_m1_r = r_out[-2]
    last_p = p_out[-1]
    last_m1_p = r_out[-2]
    evals, evecs = get_evecs_analytical(last_r)
    evals, evecs_m1 = get_evecs_analytical(last_m1_r)
    last_rho_adb = rho_adb_out[-1]
    last_rho_adb_m1 = rho_adb_out[-2]
    last_rho_db = rho_adb_to_db(last_rho_adb, evecs)
    last_rho_db_m1 = rho_adb_to_db(last_rho_adb_m1, evecs_m1)
    lastpop_0 = np.sum(np.real(last_rho_db[0, 0, rtrans_ind])) / num_points
    lastpop_1 = np.sum(np.real(last_rho_db[1, 1, rtrans_ind])) / num_points
    lastpop_m1_0 = np.sum(np.real(last_rho_db_m1[0, 0, rtrans_ind])) / num_points
    lastpop_m1_1 = np.sum(np.real(last_rho_db_m1[1, 1, rtrans_ind])) / num_points
    px0, py0, px1, py1 = get_p_rho(last_rho_db, last_p, last_r, num_points)
    rx0, ry0, rx1, ry1 = get_r_rho(last_rho_db, last_r, num_points)
    if np.abs(lastpop_m1_0 - lastpop_0) > 1e-4 or np.abs(lastpop_m1_1 - lastpop_1) > 1e-4:
        print('ERROR not stable', np.abs(lastpop_m1_0 - lastpop_0), np.abs(lastpop_m1_1 - lastpop_1), inputfile)

    return pinit[0], lastpop_0, lastpop_1, px1, py1, px0, py0, rx1, ry1, rx0, ry0  # px1 + px0, py1 + py0

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
trx1_list_A01_sh = np.zeros((len(inputfiles_sh_A01)))
try1_list_A01_sh = np.zeros((len(inputfiles_sh_A01)))
trx0_list_A01_sh = np.zeros((len(inputfiles_sh_A01)))
try0_list_A01_sh = np.zeros((len(inputfiles_sh_A01)))
n=0
for file in inputfiles_sh_A01:
    print('loading '+file)
    px_list_A01_sh[n], pop0_list_A01_sh[n], pop1_list_A01_sh[n], tpx1_list_A01_sh[n], tpy1_list_A01_sh[n], tpx0_list_A01_sh[n], tpy0_list_A01_sh[n],\
    trx1_list_A01_sh[n], try1_list_A01_sh[n], trx0_list_A01_sh[n], try0_list_A01_sh[n]= get_vals_sh(file,'./A0.1/')
    n += 1
px_sort = px_list_A01_sh.argsort()
px_list_A01_sh = px_list_A01_sh[px_sort]
pop0_list_A01_sh = pop0_list_A01_sh[px_sort]
pop1_list_A01_sh = pop1_list_A01_sh[px_sort]
tpx1_list_A01_sh = tpx1_list_A01_sh[px_sort]
tpy1_list_A01_sh = tpy1_list_A01_sh[px_sort]
tpx0_list_A01_sh = tpx0_list_A01_sh[px_sort]
tpy0_list_A01_sh = tpy0_list_A01_sh[px_sort]
trx1_list_A01_sh = trx1_list_A01_sh[px_sort]
try1_list_A01_sh = try1_list_A01_sh[px_sort]
trx0_list_A01_sh = trx0_list_A01_sh[px_sort]
try0_list_A01_sh = try0_list_A01_sh[px_sort]
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
trx1_list_A005_sh = np.zeros((len(inputfiles_sh_A005)))
try1_list_A005_sh = np.zeros((len(inputfiles_sh_A005)))
trx0_list_A005_sh = np.zeros((len(inputfiles_sh_A005)))
try0_list_A005_sh = np.zeros((len(inputfiles_sh_A005)))
n=0
for file in inputfiles_sh_A005:
    print('loading ' + file)
    px_list_A005_sh[n], pop0_list_A005_sh[n], pop1_list_A005_sh[n], tpx1_list_A005_sh[n], tpy1_list_A005_sh[n], tpx0_list_A005_sh[n], tpy0_list_A005_sh[n],\
    trx1_list_A005_sh[n], try1_list_A005_sh[n], trx0_list_A005_sh[n], try0_list_A005_sh[n]= get_vals_sh(file,'./A0.05/')
    n += 1
px_sort = px_list_A005_sh.argsort()
px_list_A005_sh = px_list_A005_sh[px_sort]
pop0_list_A005_sh = pop0_list_A005_sh[px_sort]
pop1_list_A005_sh = pop1_list_A005_sh[px_sort]
tpx1_list_A005_sh = tpx1_list_A005_sh[px_sort]
tpy1_list_A005_sh = tpy1_list_A005_sh[px_sort]
tpx0_list_A005_sh = tpx0_list_A005_sh[px_sort]
tpy0_list_A005_sh = tpy0_list_A005_sh[px_sort]
trx1_list_A005_sh = trx1_list_A005_sh[px_sort]
try1_list_A005_sh = try1_list_A005_sh[px_sort]
trx0_list_A005_sh = trx0_list_A005_sh[px_sort]
try0_list_A005_sh = try0_list_A005_sh[px_sort]
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
trx1_list_A002_sh = np.zeros((len(inputfiles_sh_A002)))
try1_list_A002_sh = np.zeros((len(inputfiles_sh_A002)))
trx0_list_A002_sh = np.zeros((len(inputfiles_sh_A002)))
try0_list_A002_sh = np.zeros((len(inputfiles_sh_A002)))
n=0
for file in inputfiles_sh_A002:
    print('loading ' + file)
    px_list_A002_sh[n], pop0_list_A002_sh[n], pop1_list_A002_sh[n], tpx1_list_A002_sh[n], tpy1_list_A002_sh[n], tpx0_list_A002_sh[n], tpy0_list_A002_sh[n],\
    trx1_list_A002_sh[n], try1_list_A002_sh[n], trx0_list_A002_sh[n], try0_list_A002_sh[n]= get_vals_sh(file,'./A0.02/')
    n += 1
px_sort = px_list_A002_sh.argsort()
px_list_A002_sh = px_list_A002_sh[px_sort]
pop0_list_A002_sh = pop0_list_A002_sh[px_sort]
pop1_list_A002_sh = pop1_list_A002_sh[px_sort]
tpx1_list_A002_sh = tpx1_list_A002_sh[px_sort]
tpy1_list_A002_sh = tpy1_list_A002_sh[px_sort]
tpx0_list_A002_sh = tpx0_list_A002_sh[px_sort]
tpy0_list_A002_sh = tpy0_list_A002_sh[px_sort]
trx1_list_A002_sh = trx1_list_A002_sh[px_sort]
try1_list_A002_sh = try1_list_A002_sh[px_sort]
trx0_list_A002_sh = trx0_list_A002_sh[px_sort]
try0_list_A002_sh = try0_list_A002_sh[px_sort]
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
trx1_list_A001_sh = np.zeros((len(inputfiles_sh_A001)))
try1_list_A001_sh = np.zeros((len(inputfiles_sh_A001)))
trx0_list_A001_sh = np.zeros((len(inputfiles_sh_A001)))
try0_list_A001_sh = np.zeros((len(inputfiles_sh_A001)))
n=0
for file in inputfiles_sh_A001:
    print('loading ' + file)
    px_list_A001_sh[n], pop0_list_A001_sh[n], pop1_list_A001_sh[n], tpx1_list_A001_sh[n], tpy1_list_A001_sh[n], tpx0_list_A001_sh[n], tpy0_list_A001_sh[n],\
    trx1_list_A001_sh[n], try1_list_A001_sh[n], trx0_list_A001_sh[n], try0_list_A001_sh[n]= get_vals_sh(file,'./A0.01/')
    n += 1
px_sort = px_list_A001_sh.argsort()
px_list_A001_sh = px_list_A001_sh[px_sort]
pop0_list_A001_sh = pop0_list_A001_sh[px_sort]
pop1_list_A001_sh = pop1_list_A001_sh[px_sort]
tpx1_list_A001_sh = tpx1_list_A001_sh[px_sort]
tpy1_list_A001_sh = tpy1_list_A001_sh[px_sort]
tpx0_list_A001_sh = tpx0_list_A001_sh[px_sort]
tpy0_list_A001_sh = tpy0_list_A001_sh[px_sort]
trx1_list_A001_sh = trx1_list_A001_sh[px_sort]
try1_list_A001_sh = try1_list_A001_sh[px_sort]
trx0_list_A001_sh = trx0_list_A001_sh[px_sort]
try0_list_A001_sh = try0_list_A001_sh[px_sort]

os.mkdir('./data/')
if save_001:
    np.savetxt('./data/px_list_A001_sh.csv',px_list_A001_sh)
    np.savetxt('./data/pop0_list_A001_sh.csv',pop0_list_A001_sh)
    np.savetxt('./data/pop1_list_A001_sh.csv',pop1_list_A001_sh)
    np.savetxt('./data/tpx1_list_A001_sh.csv',tpx1_list_A001_sh)
    np.savetxt('./data/tpy1_list_A001_sh.csv',tpy1_list_A001_sh)
    np.savetxt('./data/tpx0_list_A001_sh.csv',tpx0_list_A001_sh)
    np.savetxt('./data/tpy0_list_A001_sh.csv',tpy0_list_A001_sh)
    np.savetxt('./data/trx1_list_A001_sh.csv',trx1_list_A001_sh)
    np.savetxt('./data/try1_list_A001_sh.csv',try1_list_A001_sh)
    np.savetxt('./data/trx0_list_A001_sh.csv',trx0_list_A001_sh)
    np.savetxt('./data/try0_list_A001_sh.csv',try0_list_A001_sh)

if save_002:
    np.savetxt('./data/px_list_A002_sh.csv',px_list_A002_sh)
    np.savetxt('./data/pop0_list_A002_sh.csv',pop0_list_A002_sh)
    np.savetxt('./data/pop1_list_A002_sh.csv',pop1_list_A002_sh)
    np.savetxt('./data/tpx1_list_A002_sh.csv',tpx1_list_A002_sh)
    np.savetxt('./data/tpy1_list_A002_sh.csv',tpy1_list_A002_sh)
    np.savetxt('./data/tpx0_list_A002_sh.csv',tpx0_list_A002_sh)
    np.savetxt('./data/tpy0_list_A002_sh.csv',tpy0_list_A002_sh)
    np.savetxt('./data/trx1_list_A002_sh.csv',trx1_list_A002_sh)
    np.savetxt('./data/try1_list_A002_sh.csv',try1_list_A002_sh)
    np.savetxt('./data/trx0_list_A002_sh.csv',trx0_list_A002_sh)
    np.savetxt('./data/try0_list_A002_sh.csv',try0_list_A002_sh)

if save_005:
    np.savetxt('./data/px_list_A005_sh.csv',px_list_A005_sh)
    np.savetxt('./data/pop0_list_A005_sh.csv',pop0_list_A005_sh)
    np.savetxt('./data/pop1_list_A005_sh.csv',pop1_list_A005_sh)
    np.savetxt('./data/tpx1_list_A005_sh.csv',tpx1_list_A005_sh)
    np.savetxt('./data/tpy1_list_A005_sh.csv',tpy1_list_A005_sh)
    np.savetxt('./data/tpx0_list_A005_sh.csv',tpx0_list_A005_sh)
    np.savetxt('./data/tpy0_list_A005_sh.csv',tpy0_list_A005_sh)
    np.savetxt('./data/trx1_list_A005_sh.csv',trx1_list_A005_sh)
    np.savetxt('./data/try1_list_A005_sh.csv',try1_list_A005_sh)
    np.savetxt('./data/trx0_list_A005_sh.csv',trx0_list_A005_sh)
    np.savetxt('./data/try0_list_A005_sh.csv',try0_list_A005_sh)

if save_01:
    np.savetxt('./data/px_list_A01_sh.csv',px_list_A01_sh)
    np.savetxt('./data/pop0_list_A01_sh.csv',pop0_list_A01_sh)
    np.savetxt('./data/pop1_list_A01_sh.csv',pop1_list_A01_sh)
    np.savetxt('./data/tpx1_list_A01_sh.csv',tpx1_list_A01_sh)
    np.savetxt('./data/tpy1_list_A01_sh.csv',tpy1_list_A01_sh)
    np.savetxt('./data/tpx0_list_A01_sh.csv',tpx0_list_A01_sh)
    np.savetxt('./data/tpy0_list_A01_sh.csv',tpy0_list_A01_sh)
    np.savetxt('./data/trx1_list_A01_sh.csv',trx1_list_A01_sh)
    np.savetxt('./data/try1_list_A01_sh.csv',try1_list_A01_sh)
    np.savetxt('./data/trx0_list_A01_sh.csv',trx0_list_A01_sh)
    np.savetxt('./data/try0_list_A01_sh.csv',try0_list_A01_sh)