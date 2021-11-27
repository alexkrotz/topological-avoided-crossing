import numpy as np
import numba as nb
from numba import jit
import scipy.integrate as it
import itertools
import math
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from tqdm import tqdm

with open('inputfile.tmp') as f:
    for line in f:
        line1 = line.replace(" ", "")
        line1 = line1.rstrip('\n')
        name, value = line1.split("=")
        exec(str(line), globals())

@nb.vectorize
def erf_vec(a):
    return math.erf(a)
@jit(nopython=True,fastmath=True)
def theta(x,y):
    return (np.pi/2)*(erf_vec(B*x) + 1)#(scipy.special.erf(B*x)+1)#
@jit(nopython=True,fastmath=True)
def phi(x,y):
    return W*y
@jit(nopython=True,fastmath=True)
def dtheta(x,y): #dtheta/dx
    return (B*np.exp(-1.0 * B**2 * x**2)*np.sqrt(np.pi))
@jit(nopython=True,fastmath=True)
def dphi(x,y): #dphi/dy
    return W
@jit(nopython=True,fastmath=True)
def V_vec(r):
    x = r[:,0]
    y = r[:,1]
    out_mat = np.ascontiguousarray(np.zeros((2,2,len(x))))+0.0j
    out_mat[0,0,:] = -np.cos(theta(x,y)) * A
    out_mat[0,1,:] = np.sin(theta(x,y))*np.exp(1.0j*phi(x,y)) * A
    out_mat[1,0,:] = np.sin(theta(x,y))*np.exp(-1.0j*phi(x,y)) * A
    out_mat[1,1,:] = np.cos(theta(x,y)) * A
    return out_mat

hbar = 1
sigma = 1
mass = 1000

def wp1(r):
    return np.exp((1.0j/hbar)*np.dot(r,pinit))*np.exp(-(1/sigma**2)*np.linalg.norm(r - rinit,axis=1)**2)

def wp0(r):
    return  0 * r[:,0]


Nx = N; # N
L = xran[1]-xran[0]
jlist = np.arange(0,Nx-1+1,1)# j = 0,1,...,N-1
kxlist = np.concatenate((np.arange(0,Nx/2+1,1),np.arange((-Nx/2),0)))*2*np.pi/L#np.arange(0,Nx-1+1,1)# k = 0,1,...,N-1
kylist = np.copy(kxlist)
klist = np.array(tuple(itertools.product(kxlist,kylist)))
kxgrid = klist[:,0].reshape(len(kxlist),len(kylist))
kygrid = klist[:,1].reshape(len(kxlist),len(kylist))
knorm = np.linalg.norm(klist,axis=1)**2
kgrid = knorm.reshape(len(kxlist),len(kylist))
xjlist = 2*np.pi*jlist/Nx
dk = 2*np.pi/L
print('FFT grid ranges from, ',-np.pi*Nx/L, 'to ',np.pi*Nx/L)
#xran = [-L/2,L/2]
xlist = np.linspace(xran[0],xran[1],Nx+1)
dx = (xran[1]-xran[0])/(Nx)

ylist = np.copy(xlist)
rlist = np.array(tuple(itertools.product(xlist, ylist)))
V_list = V_vec(rlist)
ev0 = np.zeros(len(V_list[0,0,:]))
ev1 = np.zeros(len(V_list[0,0,:]))
evecs = np.zeros((2,2,len(V_list[0,0,:])),dtype=complex)
H00 = np.zeros(len(V_list[0,0,:]))
H11 = np.zeros(len(V_list[0,0,:]))
for n in range(len(V_list[0,0,:])):
    eigvals,eigvecs = np.linalg.eigh(V_list[:,:,n])
    evecs[:,:,n] = eigvecs
    ev0[n] = eigvals[0]
    ev1[n] = eigvals[1]
    H00[n] = np.real(V_list[0,0,n])
    H11[n] = np.real(V_list[1,1,n])

def get_psi_adb(psi_db):
    psi_adb = np.zeros_like(psi_db)
    for n in range(np.shape(psi_db)[-1]):
        psi_adb[:,n] = np.matmul(psi_db[:,n],evecs[:,:,n])
    return psi_adb

def normalize(state_list):
    tot = np.sum(np.abs(state_list)**2)
    return state_list/np.sqrt(tot)

state_list = normalize(np.array([wp0(rlist), wp1(rlist)]))

def get_grad(wplist):
    xdim = Nx+1
    ydim = Nx+1
    wpgrid = wplist.reshape(xdim, ydim)
    wpgrid_k = np.fft.fft2(wpgrid)
    return (1/(2*mass))*np.fft.ifft2(-kgrid * wpgrid_k).reshape(xdim*ydim)

def get_px(wplist):
    wpgrid = wplist.reshape(Nx+1,Nx+1)
    wpgrid_k = np.fft.fft2(wpgrid)
    return np.real(np.sum(np.conj(wpgrid)*np.fft.ifft2(kxgrid*wpgrid_k)))
def get_py(wplist):
    wpgrid = wplist.reshape(Nx+1,Nx+1)
    wpgrid_k = np.fft.fft2(wpgrid)
    return np.real(np.sum(np.conj(wpgrid)*np.fft.ifft2(kygrid*wpgrid_k)))

print('<0|Px|0>: ',get_px(state_list[0]),' <0|Py|0>: ', get_py(state_list[0]))
print('<1|Px|1>: ',get_px(state_list[1]),' <1|Py|1>: ', get_py(state_list[1]))

def get_grad_state(state_list):
    out_list = np.zeros_like(state_list)
    for n in range(len(state_list)):
        out_list[n] = get_grad(state_list[n])
    return out_list
def get_V_state(state_list):
    out_list = np.zeros_like(state_list)
    for n in range(np.shape(V_list)[-1]):
        v = V_list[:,:,n]
        vec = state_list[:,n]
        out_list[:,n] = np.matmul(v,vec)
    return out_list

pmax = np.max(np.abs(state_list)**2)

def timestep(state, dt):
    grad = get_grad_state(state)
    pot = get_V_state(state)
    out_vec = np.zeros_like(grad)
    for n in range(np.shape(state)[-1]):
        def f(t,psi0):
            return 1.0j*grad[:,n] -1.0j*pot[:,n]
        soln4 = it.solve_ivp(f,(0,dt), state[:,n], method='RK45', t_eval=[dt])
        out_vec[:,n] = soln4.y.flatten()
    return out_vec

#@jit(nopython=True,fastmath=True)
def timestep2(state_n,state_nm1,dt):
    grad = get_grad_state(state_n)
    pot = get_V_state(state_n)
    state_np1 = state_nm1 + (1.0j*grad - 1.0j*pot)*2*dt
    return state_np1


def plot_state(state_vec,filename):
    xdim = Nx+1
    ydim = Nx+1
    psi1 = state_vec[1]
    psi0 = state_vec[0]
    pmax = np.max(np.abs(state_vec)**2)
    fig = plt.figure(tight_layout=False,dpi=300)
    spec = gridspec.GridSpec(ncols=2,nrows=1,figure=fig)
    ax0 = fig.add_subplot(spec[0])
    ax1 = fig.add_subplot(spec[1])
    ax0.imshow(np.abs(psi0.reshape(xdim,ydim))**2,vmin=0,vmax=pmax)
    ax1.imshow(np.abs(psi1.reshape(xdim,ydim))**2,vmin=0,vmax=pmax)
    plt.savefig(filename)
    #plt.show()
    plt.close()
    return

def runSim():
    tdat = np.arange(0,tmax+dt,dt)
    tdat_bath = np.arange(0,tmax+dt_bath,dt_bath)
    psi_out = np.zeros((len(tdat),len(state_list),len(rlist)),dtype=complex)
    pop_db_0 = np.zeros(len(tdat))
    pop_db_1 = np.zeros(len(tdat))
    pop_adb_0 = np.zeros(len(tdat))
    pop_adb_1 = np.zeros(len(tdat))
    px0 = np.zeros(len(tdat))
    px1 = np.zeros(len(tdat))
    py0 = np.zeros(len(tdat))
    py1 = np.zeros(len(tdat))
    state = state_list
    t_ind = 0
    for t_bath_ind in tqdm(range(len(tdat_bath))):  # tqdm(range(len(tdat_bath))):
        # print('loop num: ',t_bath_ind)
        if tdat[t_ind] <= tdat_bath[t_bath_ind] + 0.5 * dt_bath or t_bath_ind == len(tdat_bath) - 1:
            num = '{0:0>3}'.format(t_ind)
            plot_state(state, 'imagedir/run2d/state_' + str(num) + '.png')
            pop_db_0[t_ind] = np.sum(np.abs(state[0]) ** 2)
            pop_db_1[t_ind] = np.sum(np.abs(state[1]) ** 2)
            state_vec_adb = get_psi_adb(state)
            pop_adb_0[t_ind] = np.sum(np.abs(state_vec_adb[0]) ** 2)
            pop_adb_1[t_ind] = np.sum(np.abs(state_vec_adb[1]) ** 2)
            px0[t_ind] = get_px(state[0])
            px1[t_ind] = get_px(state[1])
            py0[t_ind] = get_py(state[0])
            py1[t_ind] = get_py(state[1])
            psi_out[t_ind,:,:] += state
            t_ind += 1
        if t_bath_ind == 0:
            psi_nm1 = state_list
            psi_n = timestep(psi_nm1, dt_bath)
        else:
            state = timestep2(psi_n, psi_nm1, dt_bath)
            psi_nm1 = np.copy(psi_n)
            psi_n = np.copy(state)

    return psi_out