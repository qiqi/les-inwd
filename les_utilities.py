from numpy import *

dx = 0.1
dy = 0.1
dz = 0.1
dt = 0.1

def ip(u):
    return u[2:,1:-1,1:-1]

def im(u):
    return u[:-2,1:-1,1:-1]

def jp(u):
    return u[1:-1,2:,1:-1]

def jm(u):
    return u[1:-1,:-2,1:-1]

def kp(u):
    return u[1:-1,1:-1,2:]

def km(u):
    return u[1:-1,1:-1,:-2]

def i(u):
    return u[1:-1,1:-1,1:-1]

def extend_p(p):
    p_ext = zeros([p.shape[0]+2, p.shape[1]+2, p.shape[2]+2])
    p_ext[1:-1,1:-1,1:-1] = p
    p_ext[1:-1,1:-1,0] = p[:,:,-1]
    p_ext[1:-1,1:-1,-1] = p[:,:,0]
    p_ext[1:-1,0,1:-1] = p[:,-1,:]
    p_ext[1:-1,-1,1:-1] = p[:,0,:]
    p_ext[0,1:-1,1:-1] = p[-1,:,:]
    p_ext[-1,1:-1,1:-1] = p[0,:,:]
    return p_ext

def extend_u(u):
    u_ext = zeros([u.shape[0], u.shape[1]+2, u.shape[2]+2, u.shape[3]+2])
    u_ext[:,1:-1,1:-1,1:-1] = u
    u_ext[:,1:-1,1:-1,0] = u[:,:,:,-1]
    u_ext[:,1:-1,1:-1,-1] = u[:,:,:,0]
    u_ext[:,1:-1,0,1:-1] = u[:,:,-1,:]
    u_ext[:,1:-1,-1,1:-1] = u[:,:,0,:]
    u_ext[:,0,1:-1,1:-1] = u[:,-1,:,:]
    u_ext[:,-1,1:-1,1:-1] = u[:,0,:,:]
    return u_ext

