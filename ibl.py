__all__ = ['IBL', 'Laminar2dClosure', 'MyLaminar2dClosure']

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
    def __init__(self, xgrid, zgrid, nu, dt, closure_cls, dissipation=0.02,
                 extend_M=periodic_M, extend_trP=periodic_trP):
        self.settings = Settings(xgrid, zgrid, nu, dt)
        self.dissipation = dissipation
        self.extend_M = extend_M
        self.extend_trP = extend_trP
        self.closure = closure_cls()

    def init(self, M, trP):
        nx, nz = self.settings.nx, self.settings.nz
        if M.ndim == 3:
            M = array([M, M])
            trP = array([trP, trP])
        assert M.shape == (2, 2, nx, nz)
        assert trP.shape == (2, nx, nz)
        self.M = M
        self.trP = trP

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

        M = self.extend_M(1.5 * self.M[1] - 0.5 * self.M[0])
        trP = self.extend_trP(1.5 * self.trP[1] - 0.5 * self.trP[0])
        qe = 0.5 * (qe0 + qe1)
        h0e = (qe[0]**2 + qe[1]**2) / 2 + pe
        P, H, tau_w, K, D = self.closure(M, trP, qe, self.settings.nu)
        self.H = i(H)

        flxM = M * qe[:,newaxis] - P
        advM = qe
        srcM = tau_w
        disM = M
        ddtM = self.ddt(flxM, i(M), advM, srcM, disM)
        self.M[0] = self.M[1]
        self.M[1] += ddtM * dt

        flxP = K
        advP = h0e
        srcP = D
        disP = (M * qe).sum(0) - trP / 2
        ddtP = self.ddt(flxP, i(M), advP, srcP, disP)
        self.trP[0] = self.trP[1]
        dMqe = (self.M[1] * i(qe1) - self.M[0] * i(qe0)).sum(0)
        self.trP[1] -= (dt * ddtP - dMqe) * 2

        # pdb.set_trace()

class Laminar2dClosure:
    @classmethod
    def __call__(cls, M, trP, qe, nu):
        qe_mag = sqrt(qe[0]**2 + qe[1]**2)
        P = trP * qe * qe[:,newaxis] / qe_mag**2
        qe2_theta = (M * qe).sum(0) - trP
        Re_theta = qe2_theta / qe_mag / nu
        H = (M * qe).sum(0) / qe2_theta
        H[~isfinite(H)] = 2
        Re_theta[Re_theta < 0.1] = 0.1

        Hstar = cls.Hstar_H(H)
        tau_w = qe * qe_mag / 2 * cls.Cf_H(H) / Re_theta
        K = qe_mag**2 / 2 * M * Hstar / H
        D = qe_mag**3 * cls.Cd_H(H) / Re_theta
        return P, H, tau_w, K, D

class MyLaminar2dClosure(Laminar2dClosure):
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

if __name__ == '__main__':
    pass
