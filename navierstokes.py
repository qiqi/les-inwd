import os
import sys
import time
from numpy import *

from settings import *
from utilities import *
from convection import *
from pressure import *

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
        self.f_log.write("Timing: {0:.1e} {1:.1e} {2:.1e} {3:.1e}\n".format(
                             t1-t0, t2-t1, t3-t2, t4-t3))

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
            with open('sol_{0}.tec'.format(i/nsave), 'w') as f:
                tecplot_write(f, ns.u, ns.p)
