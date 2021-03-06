import os
import time
from numpy import *

import settings
import utilities
from utilities import tecplot_write, ip, im, jp, jm, kp, km, i
from convection import convection, velocity_mid
from pressure import pressure, pressure_grad

DEVNULL = open(os.devnull, "w")

def timestep(u0, u_bar, p0, momentum_source=0, f_log=DEVNULL,
             extend_u=None, extend_p=None):
    if extend_u is None: extend_u = utilities.extend_u
    if extend_p is None: extend_p = utilities.extend_p
    t0 = time.time()
    u_bar_ext = 1.5 * u_bar[0] - 0.5 * u_bar[1]
    u_bar[1] = u_bar[0]
    gradp0 = pressure_grad(p0, extend_p)
    t1 = time.time()
    u_hat = convection(u0, u_bar_ext, gradp0, momentum_source,
                       f_log=f_log, extend_u=extend_u)
    t2 = time.time()
    u_tilde = u_hat + settings.dt * gradp0
    p = pressure(u_tilde, f_log=f_log, extend_u=extend_u, extend_p=extend_p)
    t3 = time.time()
    gradp = pressure_grad(p, extend_p)
    u = u_tilde - settings.dt * gradp
    u_bar[0] = velocity_mid(u_tilde, p, extend_u=extend_u, extend_p=extend_p)
    t4 = time.time()
    f_log.write("u kinetic energy: {0:.1e}\n".format(kinetic_energy(u)))
    f_log.write("Timing: {0:.1e} {1:.1e} {2:.1e} {3:.1e}\n".format(
                         t1-t0, t2-t1, t3-t2, t4-t3))
    return u, u_bar, p

def kinetic_energy(u):
    return (u**2).sum(0).mean() / 2

def d_kinetic_energy_dt(u, dudt):
    return (u*dudt).sum()

if __name__ == '__main__':
    nsave = 5
    ni, nj, nk = 32, 32, 32
    u = random.random([3, ni, nj, nk])
    u_bar = random.random([2, 6, ni, nj, nk]) * 0.1
    p = random.random([ni, nj, nk])

    u, u_bar, p = timestep(u, u_bar, p)
    for i in range(100):
        u, u_bar, p = timestep(u, u_bar, p)
        if i % nsave == 0:
            with open('sol_{0}.tec'.format(i/nsave), 'w') as f:
                tecplot_write(f, u, p)
