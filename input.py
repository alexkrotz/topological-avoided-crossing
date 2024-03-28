import numpy as np
def write_input(filename,input_data):
    with open(filename, 'w') as f:
        f.write(input_data)
    return



def adjust_odd(num):
    if num%2==0:
        return num
    else:
        return num + 1
mass = 1
def gen_wp_input(alpha,A,Bx,By,W,px,py,xran,yran,rinit,pxmax,pymax,dtfac,t_mult, init_diab=1, model=4):
    out = "dim = 2\r\n\
sim = 'WP'\r\n\
Nx = " + str(adjust_odd(int(pxmax*np.abs(xran[1]-xran[0])/np.pi)+1)) + "\r\n\
Ny = " + str(adjust_odd(int(pymax*np.abs(yran[1]-yran[0])/np.pi)+1)) + "\r\n\
alpha = " + str(alpha) + "\r\n\
A = " + str(A) + "\r\n\
Bx = " + str(Bx) + "\r\n\
By = " + str(By) + "\r\n\
W = " + str(W) + "\r\n\
init_diab= " + str(init_diab) + "\r\n\
model= \'" + str(model) + "\'\r\n\
dt_fac = "+str(dtfac)+"\r\n\
tmax = np.round("+str(t_mult)+"*3*(3/("+str(px)+"/"+str(mass)+")),3)\r\n\
dt = tmax/100\r\n\
dt_bath= dt_fac * dt/70\r\n\
xran=" + str(xran) + "\r\n\
yran=" + str(yran) + "\r\n\
rinit=" + str(rinit) + "\r\n\
pinit=["+str(px)+","+str(py)+"]\r\n\
calcdir = str(sim)+'_'+str(dim)+'_'+str(alpha)+'_'+str(A)+'_'+str(Bx)+'_'+str(By)+'_'+str(W)+'_'+str(rinit)+'_'+str(pinit)+'_'+str(Nx)+'_'+str(Ny)+'_'+str(xran)+'_'+str(yran)+'_'+str(dt)+'_'+str(dt_bath)+'_'+str(tmax)"
    filename = 'WP_' +str(alpha)+'_'+str(A)+ '_' + str(Bx) +'_' + str(By) + '_' + str(W) + '_' + str(px) + '_' + str(py)+'.in'
    return out, filename

def gen_sh_input(alpha, A,Bx, By,W,N,px,py,rx,ry,rescale,t_mult,decohere, init_diab=1, model = 4):
    out = "dim = 2\r\n\
sim = 'FSSH'\r\n\
N = " + str(N) + "\r\n\
A = " + str(A) + "\r\n\
Bx = " + str(Bx) + "\r\n\
By = " + str(By) + "\r\n\
W = " + str(W) + "\r\n\
init_diab = " + str(init_diab) + "\r\n\
alpha=" + str(alpha) + "\r\n\
rescale_method="+str(rescale)+"\r\n\
tmax = np.round("+str(t_mult)+"*3*(3/("+str(px)+"/"+str(mass)+")),3)\r\n\
dt = tmax/100\r\n\
dt_bath=(dt)/210\r\n\
xran=[-10,15]\r\n\
rinit=["+str(rx)+","+str(ry)+"]\r\n\
pinit=["+str(px)+","+str(py)+"]\r\n\
include_fmag=True\r\n\
decohere="+str(decohere)+"\r\n\
model=\'"+str(model)+"\'\r\n\
calcdir = str(sim)+'_'+str(include_fmag)+'_'+str(dim)+'_'+str(alpha)+'_'+str(A)+'_'+str(Bx)+'_'+str(By)+'_'+str(W)+'_'+str(rinit)+'_'+str(pinit)+'_'+str(dt)+'_'+str(dt_bath)+'_'+str(tmax)"
    filename = 'FSSH_'+str(alpha)+'_'+str(A)+'_'+str(Bx)+'_'+str(By)+'_'+str(W)+'_'+str(px)+'_'+str(py)+'.in'
    return out, filename
#set decohere = 0 to turn off
# decohere = 1 turns off hopping after the initial hop
# decohere = 2 turns off the hop if px(t) * px(0) < 0