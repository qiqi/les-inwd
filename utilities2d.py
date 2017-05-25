__all__ = ['ip', 'im', 'kp', 'km', 'i', 'periodic_M', 'periodic_trQ']

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

def periodic_trQ(Q):
    Q_ext = zeros([Q.shape[0]+2, Q.shape[1]+2], dtype=Q.dtype)
    Q_ext[1:-1,1:-1] = Q
    Q_ext[1:-1,0] = Q[:,-1]
    Q_ext[1:-1,-1] = Q[:,0]
    Q_ext[0,:] = Q_ext[-2,:]
    Q_ext[-1,:] = Q_ext[1,:]
    return Q_ext
