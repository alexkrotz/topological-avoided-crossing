
def write_input(filename,input_data):
    with open(filename, 'w') as f:
        f.write(input_data)
    return

def gen_wp_input(A,B,W,px,py):
    out = "dim = 2\r\n\
sim = 'WP'\r\n\
Nx = 200\r\n\
Ny = 200\r\n\
A = " + str(A) + "\r\n\
B = " + str(B) + "\r\n\
W = " + str(W) + "\r\n\
tmax = int(2.5*(3/("+str(px)+"/1000)))+1\r\n\
dt = int(tmax/100)\r\n\
dt_bath=np.round(dt/70,2)\r\n\
xran=[-5,10]\r\n\
yran=[-5,10]\r\n\
rinit=[-3,0]\r\n\
pinit=["+str(px)+","+str(py)+"]\r\n\
calcdir = str(sim)+'_'+str(dim)+'_'+str(A)+'_'+str(B)+'_'+str(W)+'_'+str(rinit)+'_'+str(pinit)+'_'+str(Nx)+'_'+str(Ny)+'_'+str(xran)+'_'+str(yran)+'_'+str(dt)+'_'+str(dt_bath)+'_'+str(tmax)"
    filename = 'WP_' + str(A) + '_' + str(B) + '_' + str(W) + '_' + str(px) + '_' + str(py)+'.in'
    return out, filename

def gen_sh_input(A,B,W,px,py):
    out = "dim = 2\r\n\
sim = 'FSSH'\r\n\
N = 100000\r\n\
A = " + str(A) + "\r\n\
B = " + str(B) + "\r\n\
W = " + str(W) + "\r\n\
tmax = int(2.5*(3/("+str(px)+"/1000)))+1\r\n\
dt = int(tmax/100)\r\n\
dt_bath=np.round(dt/70,2)\r\n\
xran=[-5,10]\r\n\
rinit=[-3,0]\r\n\
pinit=["+str(px)+","+str(py)+"]\r\n\
include_fmag=True\r\n\
calcdir = str(sim)+'_'+str(include_fmag)+'_'+str(dim)+'_'+str(A)+'_'+str(B)+'_'+str(W)+'_'+str(rinit)+'_'+str(pinit)+'_'+str(dt)+'_'+str(dt_bath)+'_'+str(tmax)"
    filename = 'FSSH_'+str(A)+'_'+str(B)+'_'+str(W)+'_'+str(px)+'_'+str(py)+'.in'
    return out, filename
