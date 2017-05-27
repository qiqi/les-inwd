from __future__ import print_function

import os
import sys
my_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(my_path, '..'))

from numpy import *

from ibl import *

def test_blasius_x(test_dtype, closure):
    nu, M0, H0 = 1E-4, 2E-3, 2.59
    Q0 = M0 * (1 - 1 / H0)

    def extend_M(M):
        M_ext = zeros([M.shape[0], M.shape[1]+2, M.shape[2]+2], dtype=M.dtype)
        M_ext[0] += M0
        M_ext[:,1:-1,1:-1] = M
        M_ext[:,1:-1,0] = M[:,:,0]
        M_ext[:,1:-1,-1] = M[:,:,-1]
        M_ext[:,-1,:] = M_ext[:,-2,:]
        return M_ext

    def extend_trQ(Q):
        Q_ext = zeros([Q.shape[0]+2, Q.shape[1]+2], dtype=Q.dtype) + Q0
        Q_ext[1:-1,1:-1] = Q
        Q_ext[1:-1,0] = Q[:,0]
        Q_ext[1:-1,-1] = Q[:,-1]
        Q_ext[-1,:] = Q_ext[-2,:]
        return Q_ext

    nx, nz, dt = 100, 1, 2E-3
    xgrid = arange(nx+1, dtype=test_dtype) / nx
    zgrid = arange(nz+1, dtype=test_dtype) / nx
    ibl = IBL(xgrid, zgrid, nu, dt, closure,
              extend_M=extend_M, extend_trQ=extend_trQ, dtype=test_dtype)
    Z = zeros([nx,nz], dtype=test_dtype)
    ibl.init(array([M0 + Z, Z]), Q0 + Z)
    Z = zeros([nx+2, nz+2], dtype=test_dtype)
    qe = array([1 + Z, Z])
    pe = Z
    x = ibl.settings.xc[:,0]
    for istep in range(2500):
        ibl.timestep(qe, qe, pe)
        if (istep + 1) % 500 == 0:
            plot(x, ibl.M[1,0,:,0])
    delta_star_analytical = 1.72 * sqrt(nu * (x + x[0]) + (M0 / 1.72)**2)
    plot(x, delta_star_analytical, '--k')
    err_M = ibl.M[1,0,:,0] - delta_star_analytical
    M_qe = (ibl.M[1] * qe[:,1:-1,1:-1]).sum(0)
    ibl.H = M_qe / (M_qe - ibl.trQ[1])
    err_H = ibl.H - H0
    #print(abs(err_M).max(), delta_star_analytical.max())
    #print(abs(err_H).max(), H0)
    assert abs(err_M).max() < 3E-2 * delta_star_analytical.max()
    assert abs(err_H).max() < 3E-2 * H0
    return ibl

def test_blasius_z(test_dtype, closure):
    nu, M0, H0 = 1E-3, 1E-2, 2.59
    Q0 = M0 * (1 - 1 / H0)

    def extend_M(M):
        M_ext = zeros([M.shape[0], M.shape[1]+2, M.shape[2]+2], dtype=M.dtype)
        M_ext[1] += M0
        M_ext[:,1:-1,1:-1] = M
        M_ext[:,0,1:-1] = M[:,0,:]
        M_ext[:,-1,1:-1] = M[:,-1,:]
        M_ext[:,:,-1] = M_ext[:,:,-2]
        return M_ext

    def extend_trQ(Q):
        Q_ext = zeros([Q.shape[0]+2, Q.shape[1]+2], dtype=Q.dtype) + Q0
        Q_ext[1:-1,1:-1] = Q
        Q_ext[0,1:-1] = Q[0,:]
        Q_ext[-1,1:-1] = Q[-1,:]
        Q_ext[:,-1] = Q_ext[:,-2]
        return Q_ext

    nx, nz, dt = 2, 100, 0.1
    xgrid = arange(nx+1, dtype=test_dtype) / nx
    zgrid = arange(nz+1, dtype=test_dtype) / nx
    ibl = IBL(xgrid, zgrid, nu, dt, closure,
              extend_M=extend_M, extend_trQ=extend_trQ, dtype=test_dtype)
    Z = zeros([nx,nz], dtype=test_dtype)
    ibl.init(array([Z, M0 + Z]), Q0 + Z)
    Z = zeros([nx+2, nz+2], dtype=test_dtype)
    qe = array([Z, 1 + Z])
    pe = Z
    z = ibl.settings.zc[0]
    for istep in range(2000):
        ibl.timestep(qe, qe, pe)
        #if (istep + 1) % 250 == 0:
        #    plot(z, ibl.M[1,1,:].T)
    delta_star_analytical = 1.72 * sqrt(nu * (z + z[0]) + (M0 / 1.72)**2)
    #plot(z, delta_star_analytical, '--k')
    err_M = ibl.M[1,1] - delta_star_analytical
    M_qe = (ibl.M[1] * qe[:,1:-1,1:-1]).sum(0)
    ibl.H = M_qe / (M_qe - ibl.trQ[1])
    err_H = ibl.H - H0
    assert abs(err_M).max() < 3E-2 * delta_star_analytical.max()
    assert abs(err_H).max() < 3E-2 * H0
    return ibl

def test_blasius_xz(test_dtype, closure):
    nu, M0, H0 = 1E-6, 1E-3, 2.59
    qe0 = array([1, sqrt(3)]) / 2

    nx, nz, dt = 20, 50, 2E-2
    xgrid = arange(nx+1, dtype=test_dtype) / nx
    zgrid = arange(nz+1, dtype=test_dtype) / nx

    d_x0 = -0.5 / nx * qe0[0] + (arange(nz) + 0.5) / nx * qe0[1]
    d_z0 = -0.5 / nx * qe0[1] + (arange(nx) + 0.5) / nx * qe0[0]
    M_x0 = 1.72 * sqrt(nu * d_x0 + (M0 / 1.72)**2)
    M_z0 = 1.72 * sqrt(nu * d_z0 + (M0 / 1.72)**2)
    Q_x0 = M_x0 * (1 - 1 / H0)
    Q_z0 = M_z0 * (1 - 1 / H0)

    def extend_M(M):
        M_ext = zeros([M.shape[0], M.shape[1]+2, M.shape[2]+2], dtype=M.dtype)
        M_ext[:,1:-1,1:-1] = M
        M_ext[:,0,1:-1] = qe0[:,newaxis] * M_x0
        M_ext[:,1:-1,0] = qe0[:,newaxis] * M_z0
        M_ext[:,0,0] = (M_ext[:,1,0] + M_ext[:,0,1]) / 2
        M_ext[:,-1,:] = M_ext[:,-2,:]
        M_ext[:,:,-1] = M_ext[:,:,-2]
        return M_ext

    def extend_trQ(Q):
        Q_ext = zeros([Q.shape[0]+2, Q.shape[1]+2], dtype=Q.dtype)
        Q_ext[1:-1,1:-1] = Q
        Q_ext[0,1:-1] = Q_x0
        Q_ext[1:-1,0] = Q_z0
        Q_ext[0,0] = (Q_ext[1,0] + Q_ext[0,1]) / 2
        Q_ext[-1,:] = Q_ext[-2,:]
        Q_ext[:,-1] = Q_ext[:,-2]
        return Q_ext

    ibl = IBL(xgrid, zgrid, nu, dt, closure,
              extend_M=extend_M, extend_trQ=extend_trQ, dtype=test_dtype)
    Z = zeros([nx,nz], dtype=test_dtype)
    ibl.init(M0 * qe0[:,newaxis,newaxis] + Z, M0 * (1 - 1 / H0) + Z)
    Z = zeros([nx+2, nz+2], dtype=test_dtype)
    qe = array(qe0[:,newaxis,newaxis] + Z)
    pe = Z
    d = ibl.settings.xc * qe0[0] + ibl.settings.zc * qe0[1]
    for istep in range(500):
        ibl.timestep(qe, qe, pe)
        # if (istep + 1) % 50 == 0:
        #     subplot(2,1,1); plot(d, ibl.M[1,0,:])
        #     subplot(2,1,2); plot(d, ibl.M[1,1,:])
    delta_star_analytical = 1.72 * sqrt(nu * d + (M0 / 1.72)**2)
    # subplot(2,1,1); plot(d, qe0[0] * delta_star_analytical, ':k')
    # subplot(2,1,2); plot(d, qe0[1] * delta_star_analytical, ':k')
    err_M = ibl.M[1] - qe0[:,newaxis,newaxis] * delta_star_analytical
    M_qe = (ibl.M[1] * qe[:,1:-1,1:-1]).sum(0)
    ibl.H = M_qe / (M_qe - ibl.trQ[1])
    err_H = ibl.H - H0
    #print(abs(err_M).max(), 1E-2 * delta_star_analytical.max())
    #print(abs(err_H).max(), 1E-2 * H0)
    assert abs(err_M).max() < 3E-2 * delta_star_analytical.max()
    assert abs(err_H).max() < 3E-2 * H0
    return ibl

if __name__ == '__main__':
    from pylab import *
    ibl = test_blasius_x(float32, ExtendedLaminar2dClosure)
    ibl = test_blasius_z(float32, ExtendedLaminar2dClosure)
    ibl = test_blasius_xz(float32, ExtendedLaminar2dClosure)
    ibl = test_blasius_x(float32, ClassicLaminarSteady2dClosure)
    ibl = test_blasius_z(float32, ClassicLaminarSteady2dClosure)
    ibl = test_blasius_xz(float32, ClassicLaminarSteady2dClosure)
    ibl = test_blasius_x(float32, ClassicLaminar2dClosure)
    ibl = test_blasius_z(float32, ClassicLaminar2dClosure)
    ibl = test_blasius_xz(float32, ClassicLaminar2dClosure)
