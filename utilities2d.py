__all__ = ['ip', 'im', 'kp', 'km', 'i', 'periodic_M', 'periodic_trP']

from numpy import *

def rollback(x):
    return moveaxis(x, [-2,-1], [0,1])

def rollfwd(x):
    return moveaxis(x, [0,1], [-2,-1])

def ip(u):
    return rollfwd(rollback(u)[2:,1:-1])

def im(u):
    return rollfwd(rollback(u)[:-2,1:-1])

def kp(u):
    return rollfwd(rollback(u)[1:-1,2:])

def km(u):
    return rollfwd(rollback(u)[1:-1,:-2])

def i(u):
    return rollfwd(rollback(u)[1:-1,1:-1])

def periodic_M(M):
    M_ext = zeros([M.shape[0], M.shape[1]+2, M.shape[2]+2], dtype=M.dtype)
    M_ext[:,1:-1,1:-1] = M
    M_ext[:,1:-1,0] = M[:,:,-1]
    M_ext[:,1:-1,-1] = M[:,:,0]
    M_ext[:,0,:] = M_ext[:,-2,:]
    M_ext[:,-1,:] = M_ext[:,1,:]
    return M_ext

def periodic_trP(P):
    P_ext = zeros([P.shape[0]+2, P.shape[1]+2], dtype=P.dtype)
    P_ext[1:-1,1:-1] = P
    P_ext[1:-1,0] = P[:,-1]
    P_ext[1:-1,-1] = P[:,0]
    P_ext[0,:] = P_ext[-2,:]
    P_ext[-1,:] = P_ext[1,:]
    return P_ext
