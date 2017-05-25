__all__ = ['IBL', 'ClassicLaminar2dClosure']

import os
import sys
import pdb
import time
from numpy import *

from utilities2d import *

class Settings:
    def __init__(self, x, z, nu, dt):
        assert x.ndim == 1 and z.ndim == 1
        assert (x[1:] > x[:-1]).all()
        assert (z[1:] > z[:-1]).all()
        assert nu > 0 and dt > 0
        self.x, self.z, self.nu, self.dt = x[:,newaxis], z[newaxis,:], nu, dt

    @property
    def nx(self): return self.x.size - 1
    @property
    def nz(self): return self.z.size - 1

    @property
    def dx(self): return diff(self.x, axis=0)
    @property
    def dz(self): return diff(self.z, axis=1)

    @property
    def xc(self): return (self.x[1:,:] + self.x[:-1,:]) / 2
    @property
    def zc(self): return (self.z[:,1:] + self.z[:,:-1]) / 2


class IBL:
    def __init__(self, xgrid, zgrid, nu, dt, closure_cls, dissipation=0.05,
                 extend_M=periodic_M, extend_trQ=periodic_trQ, dtype=float64):
        self.settings = Settings(xgrid, zgrid, nu, dt)
        self.dissipation = dissipation
        self.extend_M = extend_M
        self.extend_trQ = extend_trQ
        self.closure = closure_cls()
        self.dtype = dtype

    def init(self, M, trQ):
        nx, nz = self.settings.nx, self.settings.nz
        if M.ndim == 3:
            M = array([M, M], dtype=self.dtype)
            trQ = array([trQ, trQ], dtype=self.dtype)
        assert M.shape == (2, 2, nx, nz)
        assert trQ.shape == (2, nx, nz)
        self.M = array(M, dtype=self.dtype)
        self.trQ = array(trQ, dtype=self.dtype)

    def ddt(self, flx, u, adv, src, diss):
        '''
        Formula for ddt:
        src - div(flux) - u * grad(adv) + self.dissipation * laplace(diss)
        '''
        dx, dz = self.settings.dx, self.settings.dz
        flx = (ip(flx[0]) - im(flx[0])) / (2*dx) \
            + (kp(flx[1]) - km(flx[1])) / (2*dz)
        adv = u[0] * (ip(adv) - im(adv)) / (2*dx) \
            + u[1] * (kp(adv) - km(adv)) / (2*dz)
        diss = (ip(diss) + im(diss) - 2*i(diss)) / dx \
             + (kp(diss) + km(diss) - 2*i(diss)) / dz
        return i(src) - flx - adv + self.dissipation * diss

    def timestep(self, qe0, qe1, pe):
        '''
        qe0: edge velocity at the current time step
        qe1: edge velocity at the next time step
        pe:  edge pressure in between the current and the next time steps
        '''
        nx, nz, dt = self.settings.nx, self.settings.nz, self.settings.dt
        assert qe0.shape == (2, nx+2, nz+2)
        assert qe1.shape == (2, nx+2, nz+2)
        assert pe.shape == (nx+2, nz+2)

        qe0, qe1 = array([qe0, qe1], dtype=self.dtype)
        pe = array(pe, dtype=self.dtype)

        M = self.extend_M(1.5 * self.M[1] - 0.5 * self.M[0])
        trQ = self.extend_trQ(1.5 * self.trQ[1] - 0.5 * self.trQ[0])
        qe = 0.5 * (qe0 + qe1)
        h0e = (qe[0]**2 + qe[1]**2) / 2 + pe
        Q, tau_w, K, D = self.closure(M, trQ, qe, self.settings.nu)

        flxM = M * qe[:,newaxis] - Q
        advM = qe
        srcM = tau_w
        disM = M
        ddtM = self.ddt(flxM, i(M), advM, srcM, disM)
        self.M[0] = self.M[1]
        self.M[1] += ddtM * dt

        flxQ = K
        advQ = h0e
        srcQ = D
        disQ = (M * qe).sum(0) - trQ / 2
        ddtQ = self.ddt(flxQ, i(M), advQ, srcQ, disQ)
        self.trQ[0] = self.trQ[1]
        dMqe = (self.M[1] * i(qe1) - self.M[0] * i(qe0)).sum(0)
        self.trQ[1] -= (dt * ddtQ - dMqe) * 2

        #pdb.set_trace()

class LaminarUnsteady2dClosure:
    @classmethod
    def __call__(cls, M, trQ, qe, nu):
        qe_mag = sqrt(qe[0]**2 + qe[1]**2)
        qe_by_qe_mag = qe / qe_mag
        qe_by_qe_mag[~isfinite(qe_by_qe_mag)] = sqrt(0.5)
        Q = trQ * qe_by_qe_mag * qe_by_qe_mag[:,newaxis]
        H_Q = (M * qe).sum(0) / trQ
        e_Q = nu * qe / trQ

        K = cls.HQstar_HQ(H_Q, e_Q) * Q * qe
        tau_w = qe * qe_mag / 2 * cls.Cf_H(H) * one_over_Re_theta
        D = qe_mag**3 * cls.Cd_H(H) * one_over_Re_theta
        #pdb.set_trace()
        return Q, tau_w, K, D

class LaminarSteady2dClosure:
    @classmethod
    def __call__(cls, M, trQ, qe, nu):
        qe_mag = sqrt(qe[0]**2 + qe[1]**2)
        qe_by_qe_mag = qe / qe_mag
        qe_by_qe_mag[~isfinite(qe_by_qe_mag)] = sqrt(0.5)
        Q = trQ * qe_by_qe_mag * qe_by_qe_mag[:,newaxis]
        qe2_theta = (M * qe).sum(0) - trQ
        one_over_Re_theta = nu * qe_mag / qe2_theta
        H = (M * qe).sum(0) / qe2_theta
        H[~isfinite(H)] = 2
        H[H<1.01] = 1.01

        Hstar = cls.Hstar_H(H)
        tau_w = qe * qe_mag / 2 * cls.Cf_H(H) * one_over_Re_theta
        K = qe_mag**2 / 2 * M * Hstar / H
        D = qe_mag**3 * cls.Cd_H(H) * one_over_Re_theta
        #pdb.set_trace()
        return Q, tau_w, K, D

class ClassicLaminar2dClosure(LaminarSteady2dClosure):
    @classmethod
    def Hstar_H(cls, H):
        Hstar = 1.525 + (0.076 - 0.065 * (H-2) / H) * (H - 4)**2 / H
        return Hstar

    @classmethod
    def Cd_H(cls, H):
        Cd_m2_oHs = 0.207 - 0.1 * (H - 4)**2 / H**2
        Cd_m2_oHs[H<4] = 0.207 + 0.00205 * (4 - H[H<4])**5.5
        return Cd_m2_oHs / 2 * cls.Hstar_H(H)

    @classmethod
    def Cf_H(cls, H):
        Cf_o2 = -0.066 + 0.066 * (6.2 - H)**2 / (H - 4)**2
        Cf_o2[H<6.2] = -0.066 + 0.066 * (6.2 - H[H<6.2])**1.5 / (H[H<6.2] - 1)
        return Cf_o2 * 2


def test_stokes_layer_const_v():
    dt = 0.1
    xgrid = arange(2, dtype=float)
    zgrid = arange(2, dtype=float)
    M0, H0, nu = 1, 2.413, 1.0
    ibl = IBL(xgrid, zgrid, nu, dt, MyLaminar2dClosure)
    Q0 = M0 * (1 - 1 / H0)
    ibl.init(array([M0, 0], dtype=float).reshape([2,1,1]),
             array(Q0, dtype=float).reshape([1,1]))
    Z = zeros([3,3], dtype=float)
    ue = array([1,0])[:,newaxis,newaxis] + Z
    M_history, H_history = [M0], [H0]
    for istep in range(1000):
        ibl.timestep(ue, ue, Z)
        M_history.append(ibl.M[1,0,0,0])
        H_history.append(ibl.H[0,0])
    M = array(M_history)
    H = array(H_history)
    t = dt * arange(M.size)
    M_analytical = 1.128 * sqrt(nu * t + (M0 / 1.128)**2)
    subplot(2,1,1); plot(t, M); plot(t, M_analytical, '--k')
    subplot(2,1,2); plot(t, H)
    assert abs(M - M_analytical).max() < M_analytical.max() * 1E-2
    assert abs(H - H0).max() < H0 * 2.5E-2
    return ibl

if __name__ == '__main__':
    #test_stokes_layer_const_v()
    dt, omega = 0.1, 0.1
    xgrid = arange(2, dtype=float)
    zgrid = arange(2, dtype=float)
    M0, H0, nu = 1, 2.413, 1.0
    ibl = IBL(xgrid, zgrid, nu, dt, MyLaminar2dClosure)
    Q0 = M0 * (1 - 1 / H0)
    ibl.init(array([M0, 0], dtype=float).reshape([2,1,1]),
             array(Q0, dtype=float).reshape([1,1]))
    Z = zeros([3,3], dtype=float)
    M_history, H_history, qe_history = [M0], [H0], [0]
    ue = zeros([2,1,1]) + Z
    for istep in range(1000):
        ue0 = ue.copy()
        ue[0,:] = sin(istep * dt * omega)
        ibl.timestep(ue0, ue, Z)
        M_history.append(ibl.M[1,0,0,0])
        H_history.append(ibl.H[0,0])
        qe_history.append(ue[0,0,0])
    M = array(M_history)
    H = array(H_history)
    t = dt * arange(M.size)
    M_analytical = sqrt(nu / (2 * omega)) * (sin(omega * t) + cos(omega * t))
    subplot(3,1,1); plot(t, qe_history)
    subplot(3,1,2); plot(t, M); plot(t, M_analytical, '--k')
    subplot(3,1,3); plot(t, H)
