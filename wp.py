import numpy as np
import numba as nb
from numba import jit, complex128
import scipy.integrate as it
import itertools
import math
import os
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from tqdm import tqdm
import glob

num_tmpfiles = len(glob.glob('inputfile.tmp-*'))
tmpfile = 'inputfile.tmp-' + str(num_tmpfiles)
with open(tmpfile) as f:
    for line in f:
        line1 = line.replace(" ", "")
        line1 = line1.rstrip('\n')
        name, value = line1.split("=")
        exec(str(line), globals())

if model == "model_A":
    from functions_A import *
    print('model A')
if model == "model_B":
    from functions_B import *
    print('model B')
if model == "model_C":
    from functions_C import *
    print('model C')

A = A*1000

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

hbar = 1
sigma = 1
mass = 1

# initial wavepackets
def wp1(r):
    if init_diab == 1:
        return np.exp((1.0j/hbar)*np.dot(r,pinit))*np.exp(-(1/sigma**2)*np.linalg.norm(r - rinit,axis=1)**2)
    if init_diab == 0:
        return 0 * r[:,0]

def wp0(r):
    if init_diab == 0:
        return np.exp((1.0j/hbar)*np.dot(r,pinit))*np.exp(-(1/sigma**2)*np.linalg.norm(r - rinit,axis=1)**2)
    if init_diab == 1:
        return 0 * r[:,0]

# initialize grid parameters
#xran=[-5,5]
#Nx = Nx; # N
#Ny = Ny;
Lx = xran[1]-xran[0]
Ly = yran[1]-yran[0]
jxlist = np.arange(0,Nx-1+1,1,dtype=np.float64)# j = 0,1,...,N-1
jylist = np.arange(0,Ny-1+1,1,dtype=np.float64)
kxlist = np.concatenate((np.arange(0,Nx/2+1,1,dtype=np.float64),np.arange((-Nx/2),0,dtype=np.float64)))*2*np.pi/Lx#np.arange(0,Nx-1+1,1)# k = 0,1,...,N-1
kylist = np.concatenate((np.arange(0,Ny/2+1,1,dtype=np.float64),np.arange((-Ny/2),0,dtype=np.float64)))*2*np.pi/Ly#np.arange(0,Nx-1+1,1)# k = 0,1,...,N-1
klist = np.array(tuple(itertools.product(kxlist,kylist)),dtype=np.float64)
kxgrid = klist[:,0].reshape(len(kxlist),len(kylist))
kygrid = klist[:,1].reshape(len(kxlist),len(kylist))
knorm = np.linalg.norm(klist,axis=1)**2
kgrid = knorm.reshape(len(kxlist),len(kylist))
xjlist = 2*np.pi*jxlist/Nx
yjlist = 2*np.pi*jylist/Ny
dkx = 2*np.pi/Lx
dky = 2*np.pi/Ly
print('FFT x grid ranges from, ',-np.pi*Nx/Lx, 'to ',np.pi*Nx/Lx)
print('FFT y grid ranges from, ',-np.pi*Ny/Ly, 'to ',np.pi*Ny/Ly)
#xran = [-L/2,L/2]
xlist = np.linspace(xran[0],xran[1],Nx+1,dtype=np.float64)
ylist = np.linspace(yran[0],yran[1],Ny+1,dtype=np.float64)
rxlist = xlist
rylist = ylist
dx = (xran[1]-xran[0])/(Nx)
dy = (yran[1]-yran[0])/(Ny)
rlist = np.array(tuple(itertools.product(xlist, ylist)),dtype=np.float64)
rxgrid = rlist[:, 0].reshape(len(rxlist), len(rylist))
rygrid = rlist[:, 1].reshape(len(rxlist), len(rylist))

# compute electronic eigenvectors on grid
V_list = V_vec(rlist)
ev0 = np.zeros(len(V_list[0,0,:]),dtype=np.float64)
ev1 = np.zeros(len(V_list[0,0,:]),dtype=np.float64)
evecs = np.zeros((2,2,len(V_list[0,0,:])),dtype=complex)
H00 = np.zeros(len(V_list[0,0,:]),dtype=np.float64)
H11 = np.zeros(len(V_list[0,0,:]),dtype=np.float64)
for n in range(len(V_list[0,0,:])):
    eigvals,eigvecs = np.linalg.eigh(V_list[:,:,n])
    # eigenvectors
    evecs[:,:,n] = eigvecs
    # adiabatic surfaces
    ev0[n] = eigvals[0]
    ev1[n] = eigvals[1]
    # diabatic surfaces
    H00[n] = np.real(V_list[0,0,n])
    H11[n] = np.real(V_list[1,1,n])

# convert diabatic wavefunction to adiabatic basis
def get_psi_adb(psi_db):
    psi_adb = np.zeros_like(psi_db,dtype=complex)
    for n in range(np.shape(psi_db)[-1]):
        psi_adb[:,n] = np.matmul(np.transpose(np.conj(evecs[:,:,n])),psi_db[:,n])
    return psi_adb


def normalize(state_list):
    tot = np.sum(np.abs(state_list)**2)
    return state_list/np.sqrt(tot)

# initial wavefunction
state_list = normalize(np.array([wp0(rlist), wp1(rlist)],dtype=complex))


def get_grad(wplist):
    # compute the kinetic energy term of the Hamiltonian
    ## kgrid is k^2
    xdim = Nx+1
    ydim = Ny+1
    wpgrid = wplist.reshape(xdim, ydim)
    wpgrid_k = np.fft.fft2(wpgrid)
    return (1/(2*mass))*np.fft.ifft2(-kgrid * wpgrid_k).reshape(xdim*ydim)
def adjust_odd(num):
    if num % 2 == 0:
        return num
    else:
        return num + 1
r_min = 1.0
Nx2 = adjust_odd(len(rxlist[rxlist > r_min]))
rxmin = rxlist[len(rxlist) - Nx2 - 2]
Lx2 = xran[1] - rxmin
jxlist2 = np.arange(0,Nx2 - 1 + 1, 1)
kxlist2 = np.concatenate((np.arange(0, Nx2 / 2 + 1, 1),
                             np.arange((-Nx2 / 2), 0))) * 2 * np.pi / Lx2  # np.arange(0,Nx-1+1,1)# k = 0,1,...,N-1
klist2 = np.array(tuple(itertools.product(kxlist2, kylist)))
kxgrid2 = klist2[:, 0].reshape(len(kxlist2), len(kylist))
kygrid2 = klist2[:, 1].reshape(len(kxlist2), len(kylist))
def get_tx_grid(grid): # get grid for calculating transmitted values
    tx_grid = grid[rxgrid > rxmin]
    return tx_grid.reshape((int(len(tx_grid)/len(rylist)),len(rylist)))
def get_px(wplist): # compute transmitted x direction momentum
    wpgrid = get_tx_grid(wplist.reshape(Nx + 1, Ny + 1))
    wpgrid_k = np.fft.fft2(wpgrid)
    return np.real(np.sum(np.conj(wpgrid) * np.fft.ifft2(kxgrid2 * wpgrid_k)) / (np.sum(np.abs(wpgrid) ** 2)))
def get_py(wplist): # compute transmitted y direction momentum
    wpgrid = get_tx_grid(wplist.reshape(Nx + 1, Ny + 1))
    wpgrid_k = np.fft.fft2(wpgrid)
    return np.real(np.sum(np.conj(wpgrid) * np.fft.ifft2(kygrid2 * wpgrid_k)) / (np.sum(np.conj(wpgrid) * wpgrid)))

#print('<0|Px|0>: ',np.round(get_px(state_list[0]),5),' <0|Py|0>: ', np.round(get_py(state_list[0]),5))
#print('<1|Px|1>: ',np.round(get_px(state_list[1]),5),' <1|Py|1>: ', np.round(get_py(state_list[1]),5))
state_vec_adb = get_psi_adb(state_list)
#print('Adiabatic: ')
#print('<0|Px|0>: ',np.round(get_px(state_vec_adb[0]),5),' <0|Py|0>: ', np.round(get_py(state_vec_adb[0]),5))
#print('<1|Px|1>: ',np.round(get_px(state_vec_adb[1]),5),' <1|Py|1>: ', np.round(get_py(state_vec_adb[1]),5))
def get_grad_state(state_list): # apply kinetic term to wavefunction
    out_list = np.zeros_like(state_list,dtype=complex)
    for n in range(len(state_list)):
        out_list[n] = get_grad(state_list[n])
    return out_list


def get_V_state(state_list): # Apply potential to wavefunction
    out_list = np.zeros_like(state_list,dtype=complex)
    for n in range(np.shape(V_list)[-1]):
        v = V_list[:,:,n]
        vec = state_list[:,n]
        out_list[:,n] = np.matmul(v,vec)
    return out_list

pmax = np.max(np.abs(state_list)**2)


def timestep(state, dt): # evolve wavefunction forward one timestep (first timestep only)
    grad = get_grad_state(state)
    pot = get_V_state(state)
    out_vec = np.zeros_like(grad,dtype=complex)
    for n in range(np.shape(state)[-1]):
        def f(t,psi0):
            return 1.0j*grad[:,n] -1.0j*pot[:,n]
        soln4 = it.solve_ivp(f,(0,dt), state[:,n], method='RK45', t_eval=[dt])
        out_vec[:,n] = soln4.y.flatten()
    return out_vec


def get_E(state): # compute energy of state
    grad = get_grad_state(state)
    pot = get_V_state(state)
    return np.real(np.sum(np.conj(state)*(-grad + pot)))


#@jit(nopython=True,fastmath=True)
def timestep2(state_n,state_nm1,dt): # inexpensive wavefunction evolution
    grad = get_grad_state(state_n)
    pot = get_V_state(state_n)
    state_np1 = state_nm1 + (1.0j*grad - 1.0j*pot)*2*dt
    return state_np1


def plot_state(state_vec,filename): # visualize wavefunction
    xdim = Nx+1
    ydim = Ny+1
    psi1 = state_vec[1]
    psi0 = state_vec[0]
    pmax = np.max(np.abs(state_vec)**2)
    pmax0 = np.max(np.abs(psi0)**2)
    pmax1 = np.max(np.abs(psi1)**2)
    fig = plt.figure(tight_layout=False,dpi=100)
    spec = gridspec.GridSpec(ncols=2,nrows=1,figure=fig)
    ax0 = fig.add_subplot(spec[0])
    ax1 = fig.add_subplot(spec[1])
    ax0.imshow(np.abs(psi0.reshape(xdim,ydim))**2,vmin=0,vmax=pmax0)
    ax0.set_yticks(np.linspace(0,Nx,5))
    ax0.set_yticklabels(np.linspace(xran[0],xran[1],5))
    ax0.set_xticks(np.linspace(0, Ny, 5))
    ax0.set_xticklabels(np.linspace(yran[0], yran[1], 5))
    ax1.imshow(np.abs(psi1.reshape(xdim,ydim))**2,vmin=0,vmax=pmax1)
    ax1.set_yticks(np.linspace(0, Nx, 5))
    ax1.set_yticklabels(np.linspace(xran[0], xran[1], 5))
    ax1.set_xticks(np.linspace(0, Ny, 5))
    ax1.set_xticklabels(np.linspace(yran[0], yran[1], 5))
    plt.savefig(filename)
    #plt.show()
    plt.close()
    return


def runSim():
    tdat = np.arange(0,tmax+dt,dt)
    tdat_bath = np.arange(0,tmax+dt_bath,dt_bath)
    psi_out = np.zeros((len(tdat),len(state_list),len(rlist)),dtype=complex)
    state = state_list
    t_ind = 0
    for t_bath_ind in tqdm(range(len(tdat_bath))):
        if len(tdat_bath) == t_bath_ind:
            break
        if len(tdat) == t_ind:
            break
        if tdat[t_ind] <= tdat_bath[t_bath_ind] + 0.5 * dt_bath or t_bath_ind == len(tdat_bath) - 1:
            psi_out[t_ind,:,:] += state
            if np.abs(1-np.sum(np.abs(state)**2)) > 1e-3:
                print('Normalization error: ',np.abs(1-np.sum(np.abs(state)**2)))
                exit()
            t_ind += 1
        if t_bath_ind == 0:
            psi_nm1 = state_list
            psi_n = timestep(psi_nm1, dt_bath)
        else:
            state = timestep2(psi_n, psi_nm1, dt_bath)
            psi_nm1 = np.copy(psi_n)
            psi_n = np.copy(state)
    np.save(calcdir + '/psi.npy', psi_out)
    np.save(calcdir + '/tdat.npy',tdat)
    return


def genviz():
    print('Generating Visualization...')
    psi = np.load(calcdir + '/psi.npy')
    tdat = np.load(calcdir + '/tdat.npy')
    if not(os.path.exists(calcdir+'/images/')):
        os.mkdir(calcdir+'/images/')
    pop_db_0 = np.zeros(len(tdat),dtype=np.float64)
    pop_db_1 = np.zeros(len(tdat),dtype=np.float64)
    pop_adb_0 = np.zeros(len(tdat),dtype=np.float64)
    pop_adb_1 = np.zeros(len(tdat),dtype=np.float64)
    px0 = np.zeros(len(tdat),dtype=np.float64)
    px1 = np.zeros(len(tdat),dtype=np.float64)
    py0 = np.zeros(len(tdat),dtype=np.float64)
    py1 = np.zeros(len(tdat),dtype=np.float64)
    px0_adb = np.zeros(len(tdat),dtype=np.float64)
    px1_adb = np.zeros(len(tdat),dtype=np.float64)
    py0_adb = np.zeros(len(tdat),dtype=np.float64)
    py1_adb = np.zeros(len(tdat),dtype=np.float64)
    e_db = np.zeros(len(tdat),dtype=np.float64)
    for t_ind in tqdm(range(np.shape(psi)[0])):
        num = '{0:0>3}'.format(t_ind)
        state = psi[t_ind]
        print('Writing image: ',calcdir+'/images/state_'+str(num)+'.png')
        plot_state(state, calcdir+'/images/state_' + str(num) + '.png')
        pop_db_0[t_ind] = np.sum(np.abs(state[0]) ** 2)
        pop_db_1[t_ind] = np.sum(np.abs(state[1]) ** 2)
        state_vec_adb = get_psi_adb(state)
        pop_adb_0[t_ind] = np.sum(np.abs(state_vec_adb[0]) ** 2)
        pop_adb_1[t_ind] = np.sum(np.abs(state_vec_adb[1]) ** 2)
        px0[t_ind] = get_px(state[0])
        px1[t_ind] = get_px(state[1])
        py0[t_ind] = get_py(state[0])
        py1[t_ind] = get_py(state[1])
        px0_adb[t_ind] = get_px(state_vec_adb[0])
        px1_adb[t_ind] = get_px(state_vec_adb[1])
        py0_adb[t_ind] = get_py(state_vec_adb[0])
        py1_adb[t_ind] = get_py(state_vec_adb[1])
        e_db[t_ind] = get_E(state)
    fig = plt.figure(tight_layout=False, dpi=300)
    spec = gridspec.GridSpec(ncols=2, nrows=3, figure=fig)
    ax0 = fig.add_subplot(spec[0])
    ax1 = fig.add_subplot(spec[1])
    ax2 = fig.add_subplot(spec[2])
    ax3 = fig.add_subplot(spec[3])
    ax4 = fig.add_subplot(spec[4])
    ax5 = fig.add_subplot(spec[5])
    ax0.plot(tdat, pop_db_0, label=r'$\rho_{00}$')
    ax0.plot(tdat, pop_db_1, label=r'$\rho_{11}$')
    ax0.set_title('Diabatic populations')
    ax0.legend()
    ax1.plot(tdat, pop_adb_0, label=r'$\rho_{00}$')
    ax1.plot(tdat, pop_adb_1, label=r'$\rho_{11}$')
    ax1.set_title('Adiabatic populations')
    ax1.legend()
    ax2.plot(tdat, px0, label=r'$\langle 0 | P_{x}| 0 \rangle$')
    ax2.plot(tdat, px1, label=r'$\langle 1 | P_{x}| 1 \rangle$')
    ax2.set_title('Diabatic Px')
    ax2.legend()
    ax3.plot(tdat, py0, label=r'$\langle 0 | P_{y}| 0 \rangle$')
    ax3.plot(tdat, py1, label=r'$\langle 1 | P_{y}| 1 \rangle$')
    #ax3.plot(tdat, np.sqrt(py0**2 + px0**2) + np.sqrt(py1**2 + px1**2),label='total p')
    #ax3.plot(tdat, e_db, label='total e')
    print(e_db)
    ax3.set_title('Diabatic Py')
    ax3.legend()
    ax4.plot(tdat, px0_adb, label=r'$\langle 0 | P_{x}| 0 \rangle$')
    ax4.plot(tdat, px1_adb, label=r'$\langle 1 | P_{x}| 1 \rangle$')
    ax4.set_title('Adiabatic Px')
    ax4.legend()
    ax5.plot(tdat, py0_adb, label=r'$\langle 0 | P_{y}| 0 \rangle$')
    ax5.plot(tdat, py1_adb, label=r'$\langle 1 | P_{y}| 1 \rangle$')
    ax5.set_title('Adiabatic Py')
    ax5.legend()
    fig.set_figwidth(8)
    fig.set_figheight(8)
    plt.subplots_adjust(right=.99, left=0.05, top=0.95, bottom=0.05, hspace=.2, wspace=0.275)
    plt.savefig(calcdir+'/plots.pdf')
    plt.close()
    print('Finished.')
    return
