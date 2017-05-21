from __future__ import print_function

import os
import sys
my_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(my_path, '..'))

from numpy import *
from ibl import *

def test_falkner_skan_x(a0, dstar0, H0, dt, nsteps=10):
    nu, x0 = 1E-4, 0.4
    nx, nz = 100, 1
    xext = x0 + arange(nx + 2, dtype=float) / nx
    qe0, qe_x = x0**a0, xext**a0
    M0, M_x = dstar0 * sqrt(nu * x0 * qe0), dstar0 * sqrt(nu * xext * qe_x)
    P0 = M0 * (1 - 1 / H0) * qe0

    def extend_M(M):
        M_ext = zeros([M.shape[0], M.shape[1]+2, M.shape[2]+2])
        M_ext[0] += M0
        M_ext[:,1:-1,1:-1] = M
        M_ext[:,1:-1,0] = M[:,:,0]
        M_ext[:,1:-1,-1] = M[:,:,-1]
        M_ext[:,-1,:] = M_ext[:,-2,:]
        return M_ext

    def extend_trP(P):
        P_ext = zeros([P.shape[0]+2, P.shape[1]+2]) + P0
        P_ext[1:-1,1:-1] = P
        P_ext[1:-1,0] = P[:,0]
        P_ext[1:-1,-1] = P[:,-1]
        P_ext[-1,:] = P_ext[-2,:]
        return P_ext

    xgrid = arange(nx+1, dtype=float) / nx
    zgrid = arange(nz+1, dtype=float) / nx
    ibl = IBL(xgrid, zgrid, nu, dt, MyLaminar2dClosure,
              extend_M=extend_M, extend_trP=extend_trP)
    x = ibl.settings.xc[:,0]
    Z = zeros([nx,nz])
    dstar_analytical = dstar0 * sqrt(nu * (x + x[0] + x0) / qe_x[1:-1])
    M_analytical = dstar_analytical * qe_x[1:-1]
    p_analytical = dstar_analytical * qe_x[1:-1]**2 * (1 - 1 / H0)
    ibl.init(array([M0 * 3 + Z, Z]), P0 * 3 + Z)
    #plot(x, ibl.M[1,0,:,0] / qe_x[1:-1])
    Z = zeros([nx+2, nz+2])
    qe = array([transpose([qe_x] * 3), Z])
    pe = -0.5 * (qe**2).sum(0)
    for istep in range(int(nsteps*250)):
        ibl.timestep(qe, qe, pe)
        #if (istep + 1) % 250 == 0:
        #    plot(x, ibl.M[1,0,:,0] / qe_x[1:-1])
    #plot(x, dstar_analytical, '--k')
    err_M = ibl.M[1,0,:,0] / qe_x[1:-1] - dstar_analytical
    err_H = ibl.H - H0
    assert abs(err_M).max() < 1E-2 * dstar_analytical.max()
    assert abs(err_H).max() < 2E-2 * H0
    return ibl

# from Drela, Aerodynamics of Viscous Flows (draft), 2016
a_dstar_H_dt_nsteps_table = array([[2, 0.47648, 2.1882, 1E-3, 10],
                                   [1, 0.6479, 2.21622, 1E-3, 10],
                                   [0.6, 0.7976, 2.24783, 2E-3, 10],
                                   [0.3, 1.01961, 2.30702, 2E-3, 10],
                                   [0.1, 1.34787, 2.42161, 5E-3, 10],
                                   [0, 1.7208, 2.59109, 5E-3, 10],
                                   [-0.05, 2.11777, 2.81815, 5E-3, 10],
                                   [-0.08, 2.67173, 3.22, 1E-2, 20],
                                   [-0.09043, 3.49786, 4.02916, 1E-2, 20],
                                   [-0.087, 4.14726, 4.87975, 1E-2, 20],
                                   [-0.08, 4.7554, 5.89021, 1E-2, 20]])

if __name__ == '__main__':
    for a0, dstar0, H0, dt, nsteps in a_dstar_H_dt_nsteps_table[:8]:
        #figure()
        test_falkner_skan_x(a0, dstar0, H0, dt, nsteps)
        #title('H={}'.format(H0))
