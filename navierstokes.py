import os
import sys
import time
from numpy import *

from utilities3d import *
from convection import *
from pressure import *

class Settings:
    def __init__(self, x, y, z, nu, dt, tol):
        assert x.ndim == 1 and y.ndim == 1 and z.ndim == 1
        assert (x[1:] > x[:-1]).all()
        assert (y[1:] > y[:-1]).all()
        assert (z[1:] > z[:-1]).all()
        assert nu > 0 and dt > 0 and tol > 0
        self.x, self.y, self.z, self.nu, self.dt, self.tol \
                = x[:,newaxis,newaxis], \
                  y[newaxis,:,newaxis], \
                  z[newaxis,newaxis,:], nu, dt, tol

    @property
    def nx(self): return self.x.size - 1
    @property
    def ny(self): return self.y.size - 1
    @property
    def nz(self): return self.z.size - 1

    @property
    def dx(self): return diff(self.x, axis=0)
    @property
    def dy(self): return diff(self.y, axis=1)
    @property
    def dz(self): return diff(self.z, axis=2)

    @property
    def xc(self): return (self.x[1:,:,:] + self.x[:-1,:,:]) / 2
    @property
    def yc(self): return (self.y[:,1:,:] + self.y[:,:-1,:]) / 2
    @property
    def zc(self): return (self.z[:,:,1:] + self.z[:,:,:-1]) / 2


class NavierStokes:
    def __init__(self, xgrid, ygrid, zgrid, nu, dt, tol=1E-12,
                 f_log=None, extend_u=None, extend_p=None):
        self.settings = Settings(xgrid, ygrid, zgrid, nu, dt, tol)
        if f_log is None: f_log = open(os.devnull, "w")
        if extend_u is None: extend_u = periodic_u
        if extend_p is None: extend_p = periodic_p
        self.f_log, self.extend_u, self.extend_p = f_log, extend_u, extend_p

    def init(self, u, u_bar=None, p=None, p0=None):
        nx, ny, nz = self.settings.nx, self.settings.ny, self.settings.nz
        assert u.shape == (3, nx, ny, nz)
        if u_bar is None:
            ux, uy, uz = self.extend_u(u)
            ux_bar_im = (i(ux) + im(ux)) / 2
            ux_bar_ip = (i(ux) + ip(ux)) / 2
            uy_bar_jm = (i(uy) + jm(uy)) / 2
            uy_bar_jp = (i(uy) + jp(uy)) / 2
            uz_bar_km = (i(uz) + km(uz)) / 2
            uz_bar_kp = (i(uz) + kp(uz)) / 2
            u_bar0 = array([ux_bar_im, ux_bar_ip,
                            uy_bar_jm, uy_bar_jp,
                            uz_bar_km, uz_bar_kp])
            u_bar = array([u_bar0, u_bar0])
        if p is None:
            p = zeros_like(u[0])
        if p0 is None:
            p0 = p
        assert u_bar.shape == (2, 6, nx, ny, nz)
        assert p.shape == (nx, ny, nz)
        self.u, self.u_bar, self.p, self.p0 = u, u_bar, p, p0

    @property
    def p_correct(self):
        u_bar = (self.u_bar[0] * 3 - self.u_bar[1]) / 2
        return correct_pressure(self.settings, self.p, self.p0,
                                u_bar, self.extend_p)

    def timestep(self, momentum_source=0):
        t0 = time.time()
        u_bar_ext = 1.5 * self.u_bar[0] - 0.5 * self.u_bar[1]
        self.u_bar[1] = self.u_bar[0]
        gradp0 = pressure_grad(self.settings, self.p, self.extend_p)

        t1 = time.time()
        u_hat = convection(self.settings, self.u, u_bar_ext, gradp0, momentum_source,
                           f_log=self.f_log, extend_u=self.extend_u)

        t2 = time.time()
        u_tilde = u_hat + self.settings.dt * gradp0
        self.p0 = self.p
        self.p = pressure(self.settings, u_tilde, f_log=self.f_log,
                          extend_u=self.extend_u, extend_p=self.extend_p)

        t3 = time.time()
        gradp = pressure_grad(self.settings, self.p, self.extend_p)
        self.u = u_tilde - self.settings.dt * gradp
        self.u_bar[0] = velocity_mid(self.settings, u_tilde, self.p,
                extend_u=self.extend_u, extend_p=self.extend_p)

        t4 = time.time()
        timing_str = "Navier-Stokes Timing: {0:.1e} {1:.1e} {2:.1e} {3:.1e}\n"
        self.f_log.write(timing_str.format(t1-t0, t2-t1, t3-t2, t4-t3))

    def tecwrite(self, fname):
        with open(fname, 'w') as f:
            tecplot_write(f, self.u, self.p_correct)

    def save(self, fname):
        savez(fname, u=self.u, u_bar=self.u_bar, p=self.p, p0=self.p0)

    def load(self, fname):
        data = load(fname)
        self.init(data['u'], data['u_bar'], data['p'], data['p0'])


if __name__ == '__main__':
    def kinetic_energy(u): return (u**2).sum(0).mean() / 2
    def d_kinetic_energy_dt(u, dudt): return (u*dudt).sum()

    nsave = 5
    ni, nj, nk = 32, 32, 32
    ns = NavierStokes(linspace(-pi, pi, ni+1),
                      linspace(-pi, pi, nj+1),
                      linspace(-pi, pi, nk+1), 1E-1, 0.01, f_log=sys.stdout)
    ns.init(random.random([3, ni, nj, nk]))
    for i in range(100):
        ns.timestep()
        print("u kinetic energy: {0:.1e}\n".format(kinetic_energy(ns.u)))
        if i % nsave == 0:
            ns.tecwrite('sol_{0}.tec'.format(i/nsave))
