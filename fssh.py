import numpy as np
import numba as nb
from numba import jit
import scipy.integrate as it
from matplotlib.image import NonUniformImage
import math
import os
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib as mpl
from os.path import expanduser
import matplotlib.font_manager as font_manager
from tqdm import tqdm
fontpath = expanduser('/home/akrotz/Documents/Research/pyMQC/fonts/truetype/msttcorefonts/Times_New_Roman.ttf')
prop = font_manager.FontProperties(fname=fontpath)
mpl.rcParams['font.family'] = prop.get_name()
mpl.rcParams['text.usetex'] = False
mpl.rcParams['mathtext.rm']=prop.get_name()
nice_fonts = {
    "text.usetex": True,
    "font.family": "serif",
    "font.serif" : "Times New Roman",
    "font.weight"  : "bold",
}
mpl.rcParams.update(nice_fonts)


with open('inputfile.tmp') as f:
    for line in f:
        line1 = line.replace(" ", "")
        line1 = line1.rstrip('\n')
        name, value = line1.split("=")
        exec(str(line), globals())

hbar = 1
sigma = 1
mass = 1000

#np.random.seed(1234)


@nb.vectorize
def erf_vec(a):
    return math.erf(a)


@jit(nopython=True, fastmath=True)
def theta(x, y):
    return (np.pi / 2) * (erf_vec(B * x) + 1)  # (scipy.special.erf(B*x)+1)#


@jit(nopython=True, fastmath=True)
def phi(x, y):
    return W * y


@jit(nopython=True, fastmath=True)
def dtheta(x, y):  # dtheta/dx
    return (B * np.exp(-1.0 * (B ** 2) * (x ** 2)) * np.sqrt(np.pi))


@jit(nopython=True, fastmath=True)
def dphi(x, y):  # dphi/dy
    return W


@jit(nopython=True, fastmath=True)
def V(r):
    x = r[0]
    y = r[1]
    out_mat = A * np.array([[-np.cos(theta(x, y)), np.sin(theta(x, y)) * np.exp(1.0j * phi(x, y))] \
                               , [np.sin(theta(x, y)) * np.exp(-1.0j * phi(x, y)), np.cos(theta(x, y))]])
    return out_mat


@jit(nopython=True, fastmath=True)
def V_vec(r):
    x = r[:, 0]
    y = r[:, 1]
    out_mat = np.ascontiguousarray(np.zeros((2, 2, len(x)))) + 0.0j
    out_mat[0, 0, :] = -np.cos(theta(x, y)) * A
    out_mat[0, 1, :] = np.sin(theta(x, y)) * np.exp(1.0j * phi(x, y)) * A
    out_mat[1, 0, :] = np.sin(theta(x, y)) * np.exp(-1.0j * phi(x, y)) * A
    out_mat[1, 1, :] = np.cos(theta(x, y)) * A
    return out_mat


def wp1(r):
    return np.exp((1.0j / hbar) * np.dot(r, pinit)) * np.exp(-(1 / sigma ** 2) * np.linalg.norm(r - rinit, axis=1) ** 2)


def wp0(r):
    return (0.0 + 0.0j) * np.linalg.norm(r, axis=1)


def get_rp(r0, p0, n):
    x = np.random.normal(r0[0], sigma / 2, n)
    y = np.random.normal(r0[1], sigma / 2, n)
    px = np.random.normal(p0[0], hbar / sigma, n)
    py = np.random.normal(p0[1], hbar / sigma, n)
    return np.transpose(np.array([x, y])), np.transpose(np.array([px, py]))


@jit(nopython=True, fastmath=True)
def dVx(r):
    x = r[0]
    y = r[1]
    out_mat = np.ascontiguousarray(np.zeros((2, 2))) + 0.0j
    out_mat[0, 0] = np.sin(theta(x, y)) * dtheta(x, y) * A
    out_mat[0, 1] = np.exp(1.0j * phi(x, y)) * np.cos(theta(x, y)) * dtheta(x, y) * A
    out_mat[1, 0] = np.exp(-1.0j * phi(x, y)) * np.cos(theta(x, y)) * dtheta(x, y) * A
    out_mat[1, 1] = -np.sin(theta(x, y)) * dtheta(x, y) * A
    return out_mat

@jit(nopython=True, fastmath=True)
def dVy(r):
    x = r[0]
    y = r[1]
    out_mat = np.ascontiguousarray(np.zeros((2, 2))) + 0.0j
    out_mat[0, 1] = 1.0j * np.exp(1.0j * phi(x, y)) * np.sin(theta(x, y)) * dphi(x, y) * A
    out_mat[1, 0] = -1.0j * np.exp(-1.0j * phi(x, y)) * np.sin(theta(x, y)) * dphi(x, y) * A
    return out_mat

@jit(nopython=True, fastmath=True)
def dVx_vec(r):
    x = r[:, 0]
    y = r[:, 1]
    out_mat = np.ascontiguousarray(np.zeros((2, 2, len(x)))) + 0.0j
    out_mat[0, 0, :] = np.sin(theta(x, y)) * dtheta(x, y) * A
    out_mat[0, 1, :] = np.exp(1.0j * phi(x, y)) * np.cos(theta(x, y)) * dtheta(x, y) * A
    out_mat[1, 0, :] = np.exp(-1.0j * phi(x, y)) * np.cos(theta(x, y)) * dtheta(x, y) * A
    out_mat[1, 1, :] = -np.sin(theta(x, y)) * dtheta(x, y) * A
    return out_mat


@jit(nopython=True, fastmath=True)
def dVy_vec(r):
    x = r[:, 0]
    y = r[:, 1]
    out_mat = np.ascontiguousarray(np.zeros((2, 2, len(x)))) + 0.0j
    out_mat[0, 1, :] = 1.0j * np.exp(1.0j * phi(x, y)) * np.sin(theta(x, y)) * dphi(x, y) * A
    out_mat[1, 0, :] = -1.0j * np.exp(-1.0j * phi(x, y)) * np.sin(theta(x, y)) * dphi(x, y) * A
    return out_mat


@jit(nopython=True, fastmath=True)
def matprod(state_vec, dV):
    dV00 = dV[0, 0, :]
    dV10 = dV[1, 0, :]
    dV01 = dV[0, 1, :]
    dV11 = dV[1, 1, :]
    c0 = state_vec[0]
    c1 = state_vec[1]
    return np.conj(c0) * c0 * dV00 + np.conj(c0) * c1 * dV01 + np.conj(c1) * c0 * dV10 + np.conj(c1) * c1 * dV11


@jit(nopython=True)
def get_F(state_vec, r):
    Fx = matprod(state_vec, dVx_vec(r))
    Fy = matprod(state_vec, dVy_vec(r))
    out = np.zeros((len(Fx),2))
    out[:,0] = np.real(Fx)
    out[:,1] = np.real(Fy)
    return out#np.real(np.transpose(np.array([Fx, Fy])))

def get_Fmag(r,p,act_surf_ind):
    px = p[:,0]
    py = p[:,1]
    rx = r[:,0]
    ry = r[:,1]
    f1x = (hbar / (2 * mass)) * dtheta(rx, ry) * dphi(rx, ry) * np.sin(theta(rx, ry)) * (-py)
    f1y = (hbar / (2 * mass)) * dtheta(rx, ry) * dphi(rx, ry) * np.sin(theta(rx, ry)) * (px)
    f0x = (hbar / (2 * mass)) * dtheta(rx, ry) * dphi(rx, ry) * np.sin(theta(rx, ry)) * (py)
    f0y = (hbar / (2 * mass)) * dtheta(rx, ry) * dphi(rx, ry) * np.sin(theta(rx, ry)) * (-px)
    out = np.zeros_like(p)
    pos0 = np.where(act_surf_ind == 0)
    pos1 = np.where(act_surf_ind == 1)
    out[pos0,0] = f0x[pos0]
    out[pos0,1] = f0y[pos0]
    out[pos1,0] = f1x[pos1]
    out[pos1,1] = f1y[pos1]
    return out
def get_dkk(evec_j, evec_k, ev_diff, dvmat):
    return np.matmul(np.transpose(np.conj(evec_j)),np.matmul(dvmat,evec_k))/ev_diff


def timestepRK_C(x, px, Fx, dt):
    dim = len(x)

    def f_h(t, Y):
        p1 = Y[0:dim]
        q1 = Y[dim:]
        return np.concatenate((Fx, p1 / mass))

    soln = it.solve_ivp(f_h, (0, dt), np.concatenate((px, x)), method='RK45', t_eval=[dt])
    p2 = soln.y[0:dim].flatten()
    q2 = soln.y[dim:].flatten()
    return q2, p2


@jit(nopython=True, fastmath=True)
def RK4(q_bath, p_bath, Fq, dt):
    K1 = dt * (p_bath / (mass))
    L1 = -dt * (Fq)  # [wn2] is w_alpha ^ 2
    K2 = dt * (((p_bath / (mass)) + 0.5 * L1))
    L2 = -dt * (Fq)
    K3 = dt * (((p_bath / (mass)) + 0.5 * L2))
    L3 = -dt * (Fq)
    K4 = dt * (((p_bath / (mass)) + L3))
    L4 = -dt * (Fq)
    q_bath = q_bath + 0.166667 * (K1 + 2 * K2 + 2 * K3 + K4)
    p_bath = p_bath + 0.166667 * (L1 + 2 * L2 + 2 * L3 + L4)
    return q_bath, p_bath


# @jit(nopython=True,fastmath=True)
def prop_C(r, p, F, dt):
    r_out = np.zeros_like(r)
    p_out = np.zeros_like(p)
    for n in range(len(r[0])):
        r_out[:, n], p_out[:, n] = timestepRK_C(r[:, n], p[:, n], F[:, n], dt)  # RK4(r[:,n],p[:,n],F[:,n],dt)#
    # r_out,p_out = RK4(r,p,F,dt)
    return r_out, p_out


def get_V_state(state_list, r):
    V_list = V_vec(r)
    return np.einsum('ijn,jn->in', V_list, state_list)


def get_E(state_list, r):
    V_list = V_vec(r)
    return np.real(np.sum(np.einsum('in,in->n', np.einsum('ijn,jn->in', V_list, state_list), np.conj(state_list))))

def get_E_adb(rho_adb,evals):
    return np.real(np.sum(rho_adb[0,0,:]*evals[0,:] +  rho_adb[1,1,:]*evals[1,:]))
# print(state_list[:,2])


def prop_Q0(state, r, dt):
    pot = get_V_state(state, r)
    out_vec = np.zeros_like(state)
    for n in range(np.shape(state)[-1]):
        def f(t, psi0):
            return (-1.0j * pot[:, n])
        soln4 = it.solve_ivp(f, (0, dt), state[:, n], t_eval=[dt])
        out_vec[:, n] = soln4.y.flatten()
    return out_vec


def prop_Q(state_n, state_nm1, r, dt):
    pot = get_V_state(state_n, r)
    state_np1 = state_nm1 + ((-1.0j * pot) * 2 * dt)
    return state_np1


def wigner(r, p):
    return (1 / (np.pi * hbar)) * np.exp(-(1 / 2) * (np.linalg.norm(r - rinit) / (sigma / 2)) ** 2) * np.exp(
        -(1 / 2) * (np.linalg.norm(p - pinit) / (hbar / sigma)) ** 2)


def wigner_vec(r, p):
    out_vec = np.zeros((len(r)))
    for n in range(len(r)):
        out_vec[n] = wigner(r[n], p[n])
    return out_vec

num_points = N

def get_p(state, p, N):
    px0 = np.real(np.sum(np.abs(state[0]) ** 2 * p[:, 0] / N) / np.sum((np.abs(state[0]) ** 2) / N))
    py0 = np.real(np.sum(np.abs(state[0]) ** 2 * p[:, 1] / N) / np.sum((np.abs(state[0]) ** 2) / N))
    px1 = np.real(np.sum(np.abs(state[1]) ** 2 * p[:, 0] / N) / np.sum((np.abs(state[1]) ** 2) / N))
    py1 = np.real(np.sum(np.abs(state[1]) ** 2 * p[:, 1] / N) / np.sum((np.abs(state[1]) ** 2) / N))
    return px0, py0, px1, py1


def get_p_rho(rho, p, N):
    px0 = np.real(np.sum(rho[0, 0, :] * p[:, 0] / N) / np.sum(rho[0, 0, :] / N))
    py0 = np.real(np.sum(rho[0, 0, :] * p[:, 1] / N) / np.sum(rho[0, 0, :] / N))
    px1 = np.real(np.sum(rho[1, 1, :] * p[:, 0] / N) / np.sum(rho[1, 1, :] / N))
    py1 = np.real(np.sum(rho[1, 1, :] * p[:, 1] / N) / np.sum(rho[1, 1, :] / N))
    return px0, py0, px1, py1

def plot_state(r, state, filename):
    xedges = np.linspace(-5,5,100,endpoint=True)
    yedges = np.linspace(-5,5,100,endpoint=True)
    pop_0 = np.real(state[0,0,:])#np.abs(state[0])**2
    pop_1 = np.real(state[1,1,:])#np.abs(state[1])**2
    fig = plt.figure(tight_layout=False,dpi=300)
    spec = gridspec.GridSpec(ncols=2,nrows=1,figure=fig)
    ax0 = fig.add_subplot(spec[0],aspect='equal')
    ax1 = fig.add_subplot(spec[1],aspect='equal')
    H0, xedges, yedges = np.histogram2d(r[:,0],r[:,1],bins=(xedges,yedges),weights=pop_0,density=False)

    H1, xedges, yedges = np.histogram2d(r[:,0],r[:,1],bins=(xedges,yedges),weights=pop_1,density=False)
    max1 = np.max(H1)
    max0 = np.max(H0)
    global_max = np.max(np.array([max0,max1]))
    #H0 = np.flip(H0,axis=0)
    #xcenters=(xedges[:-1] + xedges[1:])/2
    #ycenters=(yedges[:-1] + yedges[1:])/2
    ax0.imshow(H0,interpolation='bilinear', vmin=0, vmax = global_max)
    #im0 = NonUniformImage(ax0, interpolation='bilinear', extent=(-5, 5, -5, 5), cmap='viridis')
    #ax0.set_xlim([-5,5])
    #ax0.set_ylim([-5,5])
    #print(np.shape(xcenters),np.shape(ycenters),np.shape(H0))
    #im0.set_data(xcenters,ycenters,H0)
    #ax0.images.append(im0)
    #H1 = np.flip(H1,axis=0)
    #xcenters=(xedges[:-1] + xedges[1:])/2
    #ycenters=(yedges[:-1] + yedges[1:])/2
    #im1 = NonUniformImage(ax1, interpolation='bilinear', extent=(-5, 5, -5, 5), cmap='viridis')
    #ax1.set_xlim([-5,5])
    #ax1.set_ylim([-5,5])
    ax1.imshow(H1,interpolation='bilinear',vmin=0,vmax=global_max)
    #im1.set_data(xcenters,ycenters,H1)
    #ax1.images.append(im1)
    plt.savefig(filename)
    #plt.show()
    plt.close()
    return

def get_evecs(r):
    V_list = V_vec(r)
    evecs = np.zeros((2, 2, len(V_list[0, 0, :])), dtype=complex)
    evals = np.zeros((2,len(V_list[0,0,:])))
    for n in range(len(V_list[0, 0, :])):
        eigvals, eigvecs = np.linalg.eigh(V_list[:, :, n])
        evecs[:, :, n] = eigvecs
        evals[:, n] = eigvals
    return evals, evecs
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
    evals_out[0,:] = -A
    evals_out[1,:] = A
    return evals_out, evecs_out


@jit(nopython=True,fastmath=True)
def get_dkk_analytical(r):
    x = r[:,0]
    y = r[:,1]
    dkkx = (1/2)*dtheta(x,y)
    dkky = (1.0j/2)*np.sin(theta(x,y))*dphi(x,y)
    return dkkx, dkky
@jit(nopython=True,fastmath=True)
def get_dkk_analytical2(r):
    x = r[0]
    y = r[1]
    dkkx = (1/2)*dtheta(x,y)
    dkky = (1.0j/2)*np.sin(theta(x,y))*dphi(x,y)
    return dkkx, dkky



@jit(nopython=True)
def vec_db_to_adb(psi_db, evecs):
    psi_adb = np.ascontiguousarray(np.zeros(np.shape(psi_db)))+0.0j
    evecs = np.conj(evecs)
    psi_adb[0, :] = (evecs[0, 0, :] * psi_db[0, :]) + (evecs[1, 0, :] * psi_db[1, :])
    psi_adb[1, :] = (evecs[0, 1, :] * psi_db[0, :]) + (evecs[1, 1, :] * psi_db[1, :])
    #for n in range(np.shape(psi_db)[-1]):
    #    psi_adb[:, n] = np.dot(np.transpose(np.conj(np.ascontiguousarray(evecs[:, :, n]))),np.ascontiguousarray(psi_db[:, n]))
    return psi_adb
@jit(nopython=True)
def vec_adb_to_db(psi_adb, evecs):
    psi_db = np.ascontiguousarray(np.zeros(np.shape(psi_adb)))+0.0j
    psi_db[0, :] = (evecs[0, 0, :] * psi_adb[0, :]) + (evecs[0, 1, :] * psi_adb[1, :])
    psi_db[1, :] = (evecs[1, 0, :] * psi_adb[0, :]) + (evecs[1, 1, :] * psi_adb[1, :])
    #psi_db2 = np.ascontiguousarray(np.zeros(np.shape(psi_adb))) + 0.0j
    #for n in range(np.shape(psi_adb)[-1]):
    #    psi_db2[:, n] = np.dot(np.ascontiguousarray(evecs[:,:,n]), np.ascontiguousarray(psi_adb[:, n]))
    #print(np.sum(np.abs(psi_db2 - psi_db)))
    return psi_db
def rho_adb_to_db(rho_adb, evecs):
    rho_db = np.zeros_like(rho_adb)
    for n in range(np.shape(rho_adb)[-1]):
        rho_db[:,:,n] = np.matmul(evecs[:,:,n],np.matmul(rho_adb[:,:,n],np.conj(np.transpose(evecs[:,:,n]))))
    return rho_db
def rho_db_to_adb(rho_db, evecs):
    rho_adb = np.zeros_like(rho_db)
    for n in range(np.shape(rho_db)[-1]):
        rho_adb[:,:,n] = np.matmul(np.conj(np.transpose(evecs[:,:,n])),np.matmul(rho_db[:,:,n], evecs[:,:,n]))
    return rho_adb
def init_act_surf(adb_pops):
    rand = np.random.rand(np.shape(adb_pops)[-1])
    act_surf_ind = np.zeros((np.shape(adb_pops)[-1]),dtype=int)
    act_surf = np.zeros(np.shape(adb_pops),dtype=complex)
    for n in range(np.shape(adb_pops)[-1]):
        intervals = np.zeros(len(adb_pops[:,n]))
        for m in range(len(adb_pops[:,n])):
            intervals[m] = np.sum(np.real(adb_pops[:,n][0:m+1]))
        act_surf_ind[n] = (np.arange(len(adb_pops[:,n]))[intervals > rand[n]])[0]
        act_surf[:,n][act_surf_ind[n]] = 1.0
    return act_surf_ind, act_surf

def sign_adjust(eigvec_sort,eigvec_prev):
    wf_overlap = np.sum(np.conj(eigvec_prev)*eigvec_sort,axis=0)
    phase = wf_overlap/np.abs(wf_overlap)
    eigvec_out = np.zeros_like(eigvec_sort)
    for n in range(len(eigvec_sort)):
        eigvec_out[:,n] = eigvec_sort[:,n]*np.conj(phase[n])
    return eigvec_out

@nb.vectorize
def nan_num(num):
    if np.isnan(num):
        return 0.0
    if num == np.inf:
        return 100e100
    if num == -np.inf:
        return -100e100
    else:
        return num

@jit(nopython=True)
def method2(dkkqx, dkkqy):
    phase_x = np.conj(dkkqx/np.abs(dkkqx))
    dkkqx = dkkqx * phase_x
    dkkqy = dkkqy * phase_x
    fac1 = np.real(2*dkkqx)**2
    fac2 = np.real(-1.0j*2*dkkqy)**2
    if fac1 > fac2:
        return np.array([1,0])
    else:
        return np.array([0,1])


@jit(nopython=True)
def method2_analytical(dkkqx,dkkqy):
    fac1 = np.real(2*dkkqx)**2
    fac2 = np.real(-1.0j*2*dkkqy)**2
    out_vec = np.ascontiguousarray(np.zeros((len(dkkqx),2)))
    pos10 = np.where(fac1 > fac2)
    pos01 = np.where(fac2 > fac1)
    out_vec[pos10][:,0] = 1
    out_vec[pos10][:,1] = 0
    out_vec[pos01][:,0] = 0
    out_vec[pos01][:,1] = 1
    return out_vec


def hop2old(r, p, evals, evecs, evecs_prev, cg_adb, act_surf, act_surf_ind):
    rand = np.random.rand(len(r))
    ind = range(len(act_surf_ind))
    prod = np.einsum('jn,jin->in',np.conj(evecs[:,act_surf_ind,ind]),evecs_prev) # transposed inside einsum, column * column
    hop_prob = 2 * np.real(cg_adb/cg_adb[act_surf_ind,ind] * prod)
    hop_prob[act_surf_ind,ind] = 0
    bin_edge = np.zeros((len(r)))
    for k in range(len(hop_prob)):
        hop_prob[k,:] = nan_num_vec(hop_prob[k,:])
        bin_edge = bin_edge + hop_prob[k,:]
        hop_pos = np.where(rand < bin_edge)[0]
        if len(hop_pos)>0:
            for pos in hop_pos:
                #evec_k = evecs[:,:,pos][:,act_surf_ind[pos]]
                #evec_j = evecs[:,:,pos][:,k]
                eval_k = evals[:,pos][act_surf_ind[pos]]
                eval_j = evals[:, pos][k]
                ev_diff = eval_j - eval_k
                #dkkqx = get_dkk(evec_k, evec_j, ev_diff, dVx(r[pos]))
                #dkkqy = get_dkk(evec_k, evec_j, ev_diff, dVy(r[pos]))
                dkkqx, dkkqy = get_dkk_analytical(r[pos])
                if act_surf_ind[pos] == 1:
                    dkkqx = -1.0*np.conj(dkkqx)
                    dkkqy = -1.0*np.conj(dkkqy)
                dkkq = method2(dkkqx, dkkqy)
                print(dkkq)
                akkq = (1 / (2 * mass)) * np.dot(dkkq, dkkq)
                bkkq = np.dot(p[pos]/mass, dkkq)
                disc = (bkkq ** 2) - 4 * (akkq) * ev_diff
                print(disc)
                if disc >= 0:
                    if bkkq < 0:
                        gamma = bkkq + np.sqrt(disc)
                    else:
                        gamma = bkkq - np.sqrt(disc)
                    if akkq == 0:
                        gamma = 0
                    else:
                        gamma = gamma / (2 * (akkq))
                    p[pos] = p[pos] - (np.real(gamma) * dkkq)
                    #print(act_surf_ind, act_surf_ind[pos])
                    #print(act_surf)
                    act_surf_ind[pos] = k
                    #print(act_surf_ind, act_surf_ind[pos], pos, k)
                    act_surf[:, pos] = 0
                    act_surf[act_surf_ind[pos], pos] = 1
                    #print(act_surf)
            hop_prob[:,pos] = 0 # make it so once you hop you can't hop back next time.

    return p, act_surf, act_surf_ind, prod
        #if len(hop_pos) > 0:


@jit(nopython=True)
def hopold(r, p, evals, evecs, evecs_prev, cg_adb, act_surf, act_surf_ind):
    rand = np.random.rand(len(r))
    dkkqx_list, dkkqy_list = get_dkk_analytical(r)
    for i in range(len(r)):
        #product = np.matmul(np.conj(evecs[:,:,i][:,act_surf_ind[i]]),evecs_prev[:,:,i])
        #print(np.matmul(np.conj(evecs[:,:,i][:,act_surf_ind[i]]),evecs[:,:,i]),np.matmul(np.conj(evecs[:,:,i][:,act_surf_ind[i]]),evecs[:,:,i]))
        #hop_prob = 2 * np.real(cg_adb[:,i]/cg_adb[act_surf_ind[i],i] * product )
        if act_surf_ind[i] == 0:
            hop_prob = -2 * dt_bath * np.real((cg_adb[:, i] / cg_adb[act_surf_ind[i], i])
                    * np.sum((p[i, :] / mass) * np.array([dkkqx_list[i], dkkqy_list[i]])))
        if act_surf_ind[i] == 1:
            hop_prob = -2 * dt_bath * np.real((cg_adb[:, i] / cg_adb[act_surf_ind[i], i])
                    * np.sum((p[i, :] / mass) * np.array([-1.0*np.conj(dkkqx_list[i]), -1.0*np.conj(dkkqy_list[i])])))

        hop_prob[act_surf_ind[i]] = 0
        bin_edge = 0
        for k in range(len(hop_prob)):
            hop_prob[k] = nan_num(hop_prob[k])
            bin_edge = bin_edge + hop_prob[k]
            if rand[i] < bin_edge:
                #evec_k = evecs[:,:,i][:,act_surf_ind[i]]
                #evec_j = evecs[:,:,i][:,k]
                eval_k = evals[:,i][act_surf_ind[i]]
                eval_j = evals[:,i][k]
                ev_diff = eval_j - eval_k
                if act_surf_ind[i] == 0:
                    dkkq = method2(dkkqx_list[i], dkkqy_list[i])
                else:
                    dkkq = method2(-1.0*np.conj(dkkqx_list[i]), -1.0*np.conj(dkkqy_list[i]))
                akkq = (1/(2*mass))*np.sum(dkkq*dkkq)
                bkkq = np.sum((p[i]/mass) * dkkq)
                disc = (bkkq**2) - 4*(akkq)*ev_diff
                #print('disc',disc)
                if disc >= 0:
                    #print(disc)
                    if bkkq < 0:
                        gamma = bkkq + np.sqrt(disc)
                    else:
                        gamma = bkkq - np.sqrt(disc)
                    if akkq == 0:
                        gamma = 0
                    else:
                        gamma = gamma/(2*(akkq))
                    #print(np.real(gamma)*dkkq)
                    p[i] = p[i] - (np.real(gamma) * dkkq)
                    act_surf_ind[i] = k
                    act_surf[:,i] = 0
                    act_surf[act_surf_ind[i],i] = 1

                break
    return p, act_surf, act_surf_ind

@jit(nopython=True)
def get_evec_act(evecs,act_surf_ind):
    out = np.ascontiguousarray(np.zeros((2,len(act_surf_ind))))+0.0j
    for n in range(len(act_surf_ind)):
        out[:,n] = evecs[:,act_surf_ind[n],n]
    return out

@jit(nopython=True)
def get_cgadb_act(cg_adb,act_surf_ind):
    out = np.ascontiguousarray(np.zeros((len(act_surf_ind))))+0.0j
    for n in range(len(act_surf_ind)):
        out[n] = cg_adb[act_surf_ind[n],n]
    return out

@jit(nopython=True)
def hop2(r, p, evals, evecs, evecs_prev, cg_adb, act_surf, act_surf_ind):
    rand = np.random.rand(len(r))
    k_ind = (act_surf_ind - 1) * -1
    dkkqx_list, dkkqy_list = get_dkk_analytical(r)
    #print(np.shape(p),np.shape(dkkqx_list))
    d = np.ascontiguousarray(np.zeros(len(p)))+0.0j
    d[act_surf_ind==0] = ((p[:,0][act_surf_ind==0] / mass) * dkkqx_list[act_surf_ind==0]) + \
                                ((p[:,1][act_surf_ind==0]/mass)*dkkqy_list[act_surf_ind==0])
    d[act_surf_ind==1] = ((p[:,0][act_surf_ind==1] / mass) * -np.conj(dkkqx_list[act_surf_ind==1])) + \
                                ((p[:,1][act_surf_ind==1]/mass)* -np.conj(dkkqy_list[act_surf_ind==1]))
    hop_prob = 2 * dt_bath * np.real((get_cgadb_act(cg_adb,k_ind) / get_cgadb_act(cg_adb, act_surf_ind)) * d)
    #prod = np.sum(np.conj(get_evec_act(evecs,act_surf_ind))*get_evec_act(evecs_prev,k_ind),axis=0)
    #hop_prob = 2 * np.real((get_cgadb_act(cg_adb, k_ind) / get_cgadb_act(cg_adb, act_surf_ind)) * prod)
    #print(d)
    hop_prob = nan_num(hop_prob)
    print('hop2',hop_prob)
    hop_pos = np.where(rand < hop_prob)[0]
    if len(hop_pos) > 0:
        for pos in hop_pos:
            if act_surf_ind[pos] == 0:
                k = 1
            else:
                k = 0
            eval_k = evals[:, pos][act_surf_ind[pos]]
            eval_j = evals[:, pos][k]
            ev_diff = eval_j - eval_k
            dkkqx, dkkqy = dkkqx_list[pos],dkkqy_list[pos]#get_dkk_analytical2(r[pos])#
            if act_surf_ind[pos] == 1:
                dkkqx = -1.0 * np.conj(dkkqx)
                dkkqy = -1.0 * np.conj(dkkqy)
            dkkq = method2(dkkqx, dkkqy)
            akkq = (1 / (2 * mass)) * np.sum(dkkq*dkkq)
            bkkq = np.sum((p[pos] / mass)* dkkq)
            disc = (bkkq ** 2) - 4 * (akkq) * ev_diff
            if disc >= 0:
                if bkkq < 0:
                    gamma = bkkq + np.sqrt(disc)
                else:
                    gamma = bkkq - np.sqrt(disc)
                if akkq == 0:
                    gamma = 0
                else:
                    gamma = gamma / (2 * (akkq))
                p[pos] = p[pos] - (np.real(gamma) * dkkq)
                act_surf_ind[pos] = k
                act_surf[:, pos] = 0
                act_surf[act_surf_ind[pos], pos] = 1
    return p, act_surf, act_surf_ind, d#prod

@jit(nopython=True)
def hop(r, p, evals, evecs, evecs_prev, cg_adb, act_surf, act_surf_ind):
    rand = np.random.rand(len(r))
    dkkqx_list, dkkqy_list = get_dkk_analytical(r)
    for i in range(len(r)):
        #product = np.matmul(np.conj(evecs[:,:,i][:,act_surf_ind[i]]),evecs_prev[:,:,i])
        #print(np.matmul(np.conj(evecs[:,:,i][:,act_surf_ind[i]]),evecs[:,:,i]),np.matmul(np.conj(evecs[:,:,i][:,act_surf_ind[i]]),evecs[:,:,i]))
        #hop_prob = 2 * np.real(cg_adb[:,i]/cg_adb[act_surf_ind[i],i] * product )
        if act_surf_ind[i] == 1:
            hop_prob = -2 * dt_bath * np.real((cg_adb[:, i] / cg_adb[act_surf_ind[i], i])
                    * np.sum((p[i, :] / mass) * np.array([dkkqx_list[i], dkkqy_list[i]])))
        if act_surf_ind[i] == 0:
            hop_prob = -2 * dt_bath * np.real((cg_adb[:, i] / cg_adb[act_surf_ind[i], i])
                    * np.sum((p[i, :] / mass) * np.array([-1.0*np.conj(dkkqx_list[i]), -1.0*np.conj(dkkqy_list[i])])))

        hop_prob[act_surf_ind[i]] = 0
        bin_edge = 0
        if act_surf_ind[i] == 0:
            k = 1
        else:
            k = 0
        if rand[i] < hop_prob[k]:
            #evec_k = evecs[:,:,i][:,act_surf_ind[i]]
            #evec_j = evecs[:,:,i][:,k]
            eval_k = evals[:,i][act_surf_ind[i]]
            eval_j = evals[:,i][k]
            ev_diff = eval_j - eval_k
            if act_surf_ind[i] == 1:
                dkkq = method2(dkkqx_list[i], dkkqy_list[i])
            else:
                dkkq = method2(-1.0*np.conj(dkkqx_list[i]), -1.0*np.conj(dkkqy_list[i]))
            akkq = (1/(2*mass))*np.sum(dkkq*dkkq)
            bkkq = np.sum((p[i]/mass) * dkkq)
            disc = (bkkq**2) - 4*(akkq)*ev_diff
            #print('disc',disc)
            if disc >= 0:
                #print(disc)
                if bkkq < 0:
                    gamma = bkkq + np.sqrt(disc)
                else:
                    gamma = bkkq - np.sqrt(disc)
                if akkq == 0:
                    gamma = 0
                else:
                    gamma = gamma/(2*(akkq))
                #print(np.real(gamma)*dkkq)
                p[i] = p[i] - (np.real(gamma) * dkkq)
                act_surf_ind[i] = k
                act_surf[:,i] = 0
                act_surf[act_surf_ind[i],i] = 1
    return p, act_surf, act_surf_ind


#@jit(nopython=True)
def new_hop(r,p,cg_db,act_surf,act_surf_ind):
    evals, evecs = get_evecs_analytical(r)
    evals_exp = np.exp(-1.0j * evals * dt_bath)
    state_adb_t0 = vec_db_to_adb(cg_db, evecs)
    cg_adb = np.ascontiguousarray(np.zeros(np.shape(cg_db))) + 0.0j
    cg_adb[0, :] = evals_exp[0, :] * state_adb_t0[0, :]
    cg_adb[1, :] = evals_exp[1, :] * state_adb_t0[1, :]
    cg_db = vec_adb_to_db(cg_adb, evecs)
    rand = np.random.rand(len(r))
    dkkx_list, dkky_list = get_dkk_analytical(r)
    b_surf_ind = (act_surf_ind - 1 ) * -1
    cg_adb_a = get_cgadb_act(cg_adb,act_surf_ind)#cg_adb[act_surf_ind,ran]
    cg_adb_b = get_cgadb_act(cg_adb,b_surf_ind)#cg_adb[b_surf_ind,ran]
    pdab = np.ascontiguousarray(np.zeros(np.shape(act_surf_ind)))+0.0j
    ind_0 = np.where(act_surf_ind == 0)
    pdab[ind_0] = (p[:,0][ind_0]/mass) * dkkx_list[ind_0] + (p[:,1][ind_0]/mass) * dkky_list[ind_0]
    ind_1 = np.where(act_surf_ind == 1)
    pdab[ind_1] = (p[:,0][ind_1]/mass) * -np.conj(dkkx_list[ind_1]) + (p[:,1][ind_1]/mass) * -np.conj(dkky_list[ind_1])
    hop_prob =  2 * dt_bath * np.real((cg_adb_b/cg_adb_a) * (pdab))
    hop_prob = nan_num(hop_prob)
    hop_pos = np.where(rand < hop_prob)[0]
    if len(hop_pos) > 0:
        for pos in hop_pos:
            dkkqx, dkkqy = dkkx_list[pos], dkky_list[pos]  # get_dkk_analytical2(r[pos])#
            if act_surf_ind[pos] == 0:
                k = 1
                ev_diff = 2*A
            else:
                k = 0
                ev_diff = -2*A
                dkkqx = -1.0 * np.conj(dkkqx)
                dkkqy = -1.0 * np.conj(dkkqy)
            dkkq = method2(dkkqx, dkkqy)
            akkq = (1 / (2 * mass)) * np.sum(dkkq * dkkq)
            bkkq = np.sum((p[pos]/mass) * dkkq)
            disc = (bkkq ** 2) - 4 * (akkq) * ev_diff
            if disc >= 0:
                if bkkq < 0:
                    gamma = bkkq + np.sqrt(disc)
                else:
                    gamma = bkkq - np.sqrt(disc)
                if akkq == 0:
                    gamma = 0
                else:
                    gamma = gamma / (2 * (akkq))
                p[pos] = p[pos] - (np.real(gamma) * dkkq)
                act_surf_ind[pos] = k
                act_surf[:, pos] = 0
                act_surf[act_surf_ind[pos], pos] = 1
    return p, act_surf, act_surf_ind, cg_adb, cg_db

#@jit(nopython=True)
def new_hop2(r,p,cg_db,act_surf,act_surf_ind):
    p_out = np.copy(p)
    act_surf_out = np.copy(act_surf)
    act_surf_ind_out = np.copy(act_surf_ind)
    evals, evecs = get_evecs_analytical(r)
    evals_exp = np.exp(-1.0j * evals * dt_bath)
    state_adb_t0 = vec_db_to_adb(cg_db, evecs)
    cg_adb = np.ascontiguousarray(np.zeros(np.shape(cg_db))) + 0.0j
    cg_adb[0, :] = evals_exp[0, :] * state_adb_t0[0, :]
    cg_adb[1, :] = evals_exp[1, :] * state_adb_t0[1, :]
    cg_db = vec_adb_to_db(cg_adb, evecs)
    rand = np.random.rand(len(r))
    dkkx_list, dkky_list = get_dkk_analytical(r)
    b_surf_ind = (act_surf_ind - 1 ) * -1
    cg_adb_a = get_cgadb_act(cg_adb,act_surf_ind)#cg_adb[act_surf_ind,ran]
    cg_adb_b = get_cgadb_act(cg_adb,b_surf_ind)#cg_adb[b_surf_ind,ran]
    pdab = np.ascontiguousarray(np.zeros(np.shape(act_surf_ind)))+0.0j
    ind_0 = np.where(act_surf_ind == 0)
    pdab[ind_0] = p[:,0][ind_0] * dkkx_list[ind_0] + p[:,1][ind_0] * dkky_list[ind_0]
    ind_1 = np.where(act_surf_ind == 1)
    pdab[ind_1] = p[:,0][ind_1] * -np.conj(dkkx_list[ind_1]) + p[:,1][ind_1] * -np.conj(dkky_list[ind_1])
    hop_prob = -2 * dt_bath * np.real((cg_adb_b/cg_adb_a) * pdab)
    hop_prob = nan_num(hop_prob)
    hop_pos = np.where(rand < hop_prob)[0]
    if len(hop_pos) > 0:
        dkkqx, dkkqy = dkkx_list[hop_pos], dkky_list[hop_pos]
        act_surf_ind_hop = act_surf_ind[hop_pos]
        p_hop = p[hop_pos]
        ev_diff = ((act_surf_ind_hop * 2) - 1) * -2 * A
        dkkqx[act_surf_ind_hop == 1] = -1.0 * np.conj(dkkqx[act_surf_ind_hop == 1])
        dkkqy[act_surf_ind_hop == 1] = -1.0 * np.conj(dkkqy[act_surf_ind_hop == 1])
        dkkq = method2_analytical(dkkqx, dkkqy)
        akkq = (1/(2*mass)) + np.zeros(len(dkkq))
        bkkq = (p_hop[:,0]/mass)*dkkq[:,0] + (p_hop[:,1]/mass)*dkkq[:,1]
        disc = (bkkq**2) - 4 * (akkq) * ev_diff
        allow = np.where(disc >= 0)
        if len(allow[0]) > 0:
            bkkq_a = bkkq[allow]
            akkq_a = akkq[allow]
            disc_a = disc[allow]
            sign = ((bkkq_a < 0) * 2) - 1
            gamma = (bkkq_a + sign*np.sqrt(disc_a))/(2*akkq_a)
            p_out[hop_pos][allow][:, 0] = np.array(p_hop[allow][:, 0] - np.real(gamma) * dkkq[allow][:, 0])
            p_out[hop_pos][allow][:, 1] = np.array(p_hop[allow][:, 1] - np.real(gamma) * dkkq[allow][:, 1])
            print(act_surf_ind[hop_pos][allow],(act_surf_ind[hop_pos][allow] - 1) * -1)
            act_surf_ind_out[hop_pos][allow] = np.copy(np.array((act_surf_ind[hop_pos][allow] - 1) * -1))
            print(act_surf_ind_out[hop_pos][allow])
            act_surf_out[:,hop_pos][:,allow[0]] = (act_surf[:,hop_pos][:,allow[0]] - 1) * -1
    return p_out, act_surf_out, act_surf_ind_out, cg_adb, cg_db


def runSim():
    if os.path.exists(calcdir+'/run_num.npy'):
        run_num = np.load(calcdir+'/run_num.npy')
        run_num += 1
    else:
        run_num = 0
    tdat = np.arange(0,tmax+dt,dt)
    tdat_bath = np.arange(0,tmax+dt_bath,dt_bath)
    r, p = get_rp(rinit, pinit, num_points)
    state_vec = np.zeros((2, num_points), dtype=complex)  # + np.sqrt(0.5) + 0.0j
    state_vec[1] = 1 + 0.0j  # wp1(r)#/np.sqrt(num_points)
    rho_adb_out = np.zeros((len(tdat),len(state_vec),len(state_vec),len(r)),dtype=complex)
    state_db = state_vec
    evals, evecs = get_evecs_analytical(r)#get_evecs(r)
    state_adb = vec_db_to_adb(state_db, evecs)
    pops_adb = np.real(np.abs(state_adb)**2)
    act_surf_ind, act_surf = init_act_surf(pops_adb)
    px0, py0, px1, py1 = get_p(state_db, p, num_points)
    print('<0|Px|0>: ', np.round(px0, 5), ' <0|Py|0>: ', np.round(py0, 5))
    print('<1|Px|1>: ', np.round(px1, 5), ' <1|Py|1>: ', np.round(py1, 5))
    p_out = np.zeros((len(tdat),len(p), 2))
    r_out = np.zeros((len(tdat),len(r), 2))
    t_ind = 0
    prod = np.zeros(len(r))
    dablist = np.zeros(len(tdat))
    for t_bath_ind in tqdm(range(len(tdat_bath))):
        if tdat[t_ind] <= tdat_bath[t_bath_ind] + 0.5 * dt_bath or t_bath_ind == len(tdat_bath) - 1:
            rho_adb = np.zeros((2,2,np.shape(state_adb)[-1]),dtype=complex)
            rho_adb[0,0,:] = act_surf[0,:]
            rho_adb[1,1,:] = act_surf[1,:]
            rho_adb[0,1,:] = np.conj(state_adb[0,:])*state_adb[1,:]
            rho_adb[1,0,:] = state_adb[0,:]*np.conj(state_adb[1,:])
            rho_adb_out[t_ind, :, :, :] = rho_adb
            p_out[t_ind,:] = p
            r_out[t_ind,:] = r
            dablist[t_ind] = np.real(np.sum(prod))
            t_ind += 1
        F = np.zeros_like(r)
        Fmag = get_Fmag(r,p,act_surf_ind)
        if not(include_fmag):
            Fmag *= 0
        r, p = prop_C(r, p, -F+Fmag, dt_bath)
        p, act_surf, act_surf_ind, state_adb, state_db = new_hop(r, p, state_db, act_surf, act_surf_ind)
    np.save(calcdir + '/rho_adb_'+str(run_num)+'.npy',rho_adb_out)
    np.save(calcdir + '/tdat.npy',tdat)
    np.save(calcdir + '/p_' + str(run_num) + '.npy', p_out)
    np.save(calcdir + '/r_' + str(run_num) + '.npy', r_out)
    np.save(calcdir + '/run_num.npy', run_num)
    return

def genviz():
    print('Generating Visualization...')
    run_num = np.load(calcdir + '/run_num.npy')
    for n in np.arange(0,run_num+1):
        if n == 0:
            p_out = np.load(calcdir + '/p_' + str(n) + '.npy')
            r_out = np.load(calcdir + '/r_' + str(n) + '.npy')
            rho_adb_out = np.load(calcdir + '/rho_adb_'+str(n)+'.npy')
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
    if not (os.path.exists(calcdir + '/images/')):
        os.mkdir(calcdir + '/images/')
    pop_db_0 = np.zeros(len(tdat))
    pop_db_1 = np.zeros(len(tdat))
    pop_adb_0 = np.zeros(len(tdat))
    pop_adb_1 = np.zeros(len(tdat))
    px0 = np.zeros(len(tdat))
    px1 = np.zeros(len(tdat))
    py0 = np.zeros(len(tdat))
    py1 = np.zeros(len(tdat))
    px0_adb = np.zeros(len(tdat))
    px1_adb = np.zeros(len(tdat))
    py0_adb = np.zeros(len(tdat))
    py1_adb = np.zeros(len(tdat))
    eq_db = np.zeros(len(tdat))
    ec_db = np.zeros(len(tdat))
    for t_ind in tqdm(range(len(tdat))):
        r = r_out[t_ind]
        p = p_out[t_ind]
        rho_adb = rho_adb_out[t_ind]
        evals, evecs = get_evecs_analytical(r)
        rho_db = rho_adb_to_db(rho_adb,evecs)
        #num = '{0:0>3}'.format(t_ind)
        #plot_state(r, rho_db, calcdir + '/images/state_' + str(num) + '.png')
        pop_db_0[t_ind] = np.sum(np.real(rho_db[0,0,:]))/num_points
        pop_db_1[t_ind] = np.sum(np.real(rho_db[1,1,:]))/num_points
        pop_adb_0[t_ind] = np.sum(np.real(rho_adb[0,0,:]))/num_points
        pop_adb_1[t_ind] = np.sum(np.real(rho_adb[1,1,:]))/num_points
        px0_n, py0_n, px1_n, py1_n = get_p_rho(rho_db, p, num_points)
        px0[t_ind] = px0_n
        px1[t_ind] = px1_n
        py0[t_ind] = py0_n
        py1[t_ind] = py1_n
        px0_n_adb, py0_n_adb, px1_n_adb, py1_n_adb = get_p_rho(rho_adb, p, num_points)
        px0_adb[t_ind] = px0_n_adb
        px1_adb[t_ind] = px1_n_adb
        py0_adb[t_ind] = py0_n_adb
        py1_adb[t_ind] = py1_n_adb
        ec_db[t_ind] = (np.sum((1/(2*mass))*p**2))/num_points#get_E(state, r))/num_points
        eq_db[t_ind] =  get_E_adb(rho_adb,evals)/num_points
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
    #ax3.plot(tdat, np.sqrt(py0 ** 2 + px0 ** 2) + np.sqrt(py1 ** 2 + px1 ** 2), label='total p')
    #ax3.plot(tdat, eq_db, label='eq')
    #ax3.plot(tdat, ec_db, label='ec')
    print(eq_db+ec_db)
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
    plt.savefig(calcdir + '/plots.pdf')
    plt.close()
    print('Finished.')

    return