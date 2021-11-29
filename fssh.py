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
    return (B * np.exp(-1.0 * B ** 2 * x ** 2) * np.sqrt(np.pi))


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



def get_F(state_vec, r):
    Fx = matprod(state_vec, dVx_vec(r))
    Fy = matprod(state_vec, dVy_vec(r))
    return np.real(np.transpose(np.array([Fx, Fy])))

def get_dkk(evec_j, evec_k, ev_diff, dvmat):
    return np.matmul(np.transpose(np.conj(evec_j)),np.matmul(dvmat,evec_k))/ev_diff


def timestepRK_C(x, px, Fx, dt):
    dim = len(x)

    def f_h(t, Y):
        p1 = Y[0:dim]
        q1 = Y[dim:]
        return np.concatenate((-Fx, p1 / mass))

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
    px0 = np.real(np.sum(np.abs(state[0]) ** 2 * p[:, 0] / N))
    py0 = np.real(np.sum(np.abs(state[0]) ** 2 * p[:, 1] / N))
    px1 = np.real(np.sum(np.abs(state[1]) ** 2 * p[:, 0] / N))
    py1 = np.real(np.sum(np.abs(state[1]) ** 2 * p[:, 1] / N))
    return px0, py0, px1, py1
def get_p_rho(rho, p, N):
    px0 = np.real(np.sum(rho[0,0,:] * p[:, 0] / N))
    py0 = np.real(np.sum(rho[0,0,:] * p[:, 1] / N))
    px1 = np.real(np.sum(rho[1,1,:] * p[:, 0] / N))
    py1 = np.real(np.sum(rho[1,1,:] * p[:, 1] / N))
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
    H0 = np.flip(H0,axis=0)
    xcenters=(xedges[:-1] + xedges[1:])/2
    ycenters=(yedges[:-1] + yedges[1:])/2
    #ax0.imshow(H0,interpolation='bilinear')
    im0 = NonUniformImage(ax0, interpolation='bilinear', extent=(-5, 5, -5, 5), cmap='viridis')
    ax0.set_xlim([-5,5])
    ax0.set_ylim([-5,5])
    #print(np.shape(xcenters),np.shape(ycenters),np.shape(H0))
    im0.set_data(xcenters,ycenters,H0)
    ax0.images.append(im0)
    H1, xedges, yedges = np.histogram2d(r[:,0],r[:,1],bins=(xedges,yedges),weights=pop_1,density=False)
    H1 = np.flip(H1,axis=0)
    xcenters=(xedges[:-1] + xedges[1:])/2
    ycenters=(yedges[:-1] + yedges[1:])/2
    im1 = NonUniformImage(ax1, interpolation='bilinear', extent=(-5, 5, -5, 5), cmap='viridis')
    ax1.set_xlim([-5,5])
    ax1.set_ylim([-5,5])
    #ax1.imshow(H1,interpolation='bilinear')
    im1.set_data(xcenters,ycenters,H1)
    ax1.images.append(im1)
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
def vec_db_to_adb(psi_db, evecs):
    psi_adb = np.zeros_like(psi_db)
    for n in range(np.shape(psi_db)[-1]):
        psi_adb[:, n] = np.matmul(np.transpose(np.conj(evecs[:, :, n])),psi_db[:, n])
    return psi_adb
def vec_adb_to_db(psi_adb, evecs):
    psi_db = np.zeros_like(psi_adb)
    for n in range(np.shape(psi_adb)[-1]):
        psi_db[:, n] = np.matmul(evecs[:,:,n], psi_adb[:, n])
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

@jit(nopython=True)
def nan_num(num):
    if np.isnan(num):
        return 0.0
    if num == np.inf:
        return 100e100
    if num == -np.inf:
        return -100e100
    else:
        return num

nan_num_vec = np.vectorize(nan_num)

def hop(r, p, evals, evecs, evecs_prev, cg_adb, act_surf, act_surf_ind):
    rand = np.random.rand(len(r))
    dVx_mat = dVx_vec(r)
    dVy_mat = dVy_vec(r)
    for i in range(len(r)):
        product = np.real(np.dot(np.transpose(np.conj(evecs[:,:,i])),evecs_prev[:,:,i][:,act_surf_ind[i]]))
        hop_prob = 2 * np.real(cg_adb[:,i]/cg_adb[act_surf_ind[i],i] * product)
        hop_prob[act_surf_ind[i]] = 0
        bin_edge = 0
        for k in range(len(hop_prob)):
            hop_prob[k] = nan_num(hop_prob[k])
            bin_edge = bin_edge + hop_prob[k]
            if rand[i] < bin_edge:
                evec_k = evecs[:,:,i][:,act_surf_ind[i]]
                evec_j = evecs[:,:,i][:,k]
                eval_k = evals[:,i][act_surf_ind[i]]
                eval_j = evals[:,i][k]
                ev_diff = eval_j - eval_k
                dkkqx = get_dkk(evec_k, evec_j, ev_diff, dVx_mat[:,:,i])
                dkkqy = get_dkk(evec_k, evec_j, ev_diff, dVy_mat[:,:,i])
                dkkq = np.array([np.abs(dkkqx)*np.sign(np.real(dkkqx)), np.abs(dkkqy)*np.sign(np.real(dkkqy))]).flatten()
                print(dkkq)
                akkq = (1/(2*mass))*np.dot(dkkq,dkkq)
                bkkq = np.dot(p[i],dkkq)
                disc = (bkkq**2) - 4*(akkq)*mass*ev_diff
                if disc >= 0:
                    if bkkq < 0:
                        gamma = bkkq + np.sqrt(disc)
                    else:
                        gamma = bkkq - np.sqrt(disc)
                    if akkq == 0:
                        gamma = 0
                    else:
                        gamma = gamma/(2*(akkq))
                    p[i] = p[i] - (np.real(gamma) * dkkq/mass)
                    act_surf_ind[i] = k
                    act_surf[:,i] = 0
                    act_surf[act_surf_ind[i],i] = 1
                break
    return p, act_surf, act_surf_ind



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
    rho_db_out = np.zeros((len(tdat), len(state_vec), len(state_vec), len(r)), dtype=complex)
    psi_out = np.zeros((len(tdat),len(state_vec),len(r)),dtype=complex)
    state_db = state_vec
    evals, evecs = get_evecs(r)
    state_adb = vec_db_to_adb(state_db, evecs)
    pops_adb = np.real(np.abs(state_adb)**2)
    act_surf_ind, act_surf = init_act_surf(pops_adb)
    px0, py0, px1, py1 = get_p(state_db, p, num_points)
    print('<0|Px|0>: ', np.round(px0, 5), ' <0|Py|0>: ', np.round(py0, 5))
    print('<1|Px|1>: ', np.round(px1, 5), ' <1|Py|1>: ', np.round(py1, 5))
    p_out = np.zeros((len(tdat),len(p), 2))
    r_out = np.zeros((len(tdat),len(r), 2))
    t_ind = 0
    for t_bath_ind in tqdm(range(len(tdat_bath))):  # tqdm(range(len(tdat_bath))):
        # print('loop num: ',t_bath_ind)
        if tdat[t_ind] <= tdat_bath[t_bath_ind] + 0.5 * dt_bath or t_bath_ind == len(tdat_bath) - 1:
            rho_adb = np.zeros((2,2,np.shape(state_adb)[-1]),dtype=complex)
            rho_adb[0,0,:] = act_surf[0,:]
            rho_adb[1,1,:] = act_surf[1,:]
            rho_adb[0,1,:] = np.conj(state_adb[0,:])*state_adb[1,:]
            rho_adb[1,0,:] = state_adb[0,:]*np.conj(state_adb[1,:])
            rho_adb_out[t_ind, :, :] = rho_adb
            p_out[t_ind,:] = p
            r_out[t_ind,:] = r
            t_ind += 1
        F = np.zeros_like(r)#get_F(state_adb, r) # this is always zero
        #print(F)
        r, p = prop_C(r, p, F, dt_bath)
        evecs_previous = np.copy(evecs)
        evals_previous = np.copy(evals)
        evals, evecs = get_evecs(r)
        evecs = sign_adjust(evecs, evecs_previous)
        evals_exp = np.exp(-1.0j*evals*dt_bath)
        diag_matrix = np.zeros((2,2,len(r)),dtype=complex)
        diag_matrix[0,0,:] = evals_exp[0,:]
        diag_matrix[1,1,:] = evals_exp[1,:]
        state_adb = np.einsum('jin,jn->in',diag_matrix,vec_db_to_adb(state_db,evecs))
        state_db = vec_adb_to_db(state_adb,evecs)
        p, act_surf, act_surf_ind = hop(r, p, evals, evecs, evecs_previous, state_adb, act_surf, act_surf_ind)

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
    e_db = np.zeros(len(tdat))
    for t_ind in tqdm(range(len(tdat))):
        r = r_out[t_ind]
        p = p_out[t_ind]
        V_list = V_vec(r)
        rho_adb = rho_adb_out[t_ind]
        evecs = np.zeros((2, 2, len(V_list[0, 0, :])), dtype=complex)
        evals = np.zeros((2,len(V_list[0,0,:])))
        for n in range(len(V_list[0, 0, :])):
            eigvals, eigvecs = np.linalg.eigh(V_list[:, :, n])
            evecs[:, :, n] = eigvecs
            evals[:,n]  = eigvals
        rho_db = rho_adb_to_db(rho_adb,evecs)
        num = '{0:0>3}'.format(t_ind)
        plot_state(r, rho_db, calcdir + '/images/state_' + str(num) + '.png')
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
        e_db[t_ind] = (np.sum((1/(2*mass))*p**2) + get_E_adb(rho_adb,evals))/num_points#get_E(state, r))/num_points
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
    ax3.plot(tdat, np.sqrt(py0 ** 2 + px0 ** 2) + np.sqrt(py1 ** 2 + px1 ** 2), label='total p')
    ax3.plot(tdat, e_db, label='total e')
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
    plt.savefig(calcdir + '/plots.pdf')
    plt.close()
    print('Finished.')

    return