from __future__ import print_function

import os
import sys
my_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(my_path, '..'))

from navierstokes import *
import settings
from pressure import correct_pressure
from numpy import linspace, pi, cos, sin

if __name__ == '__main__':
    nsave = -1
    T, nu = 1, 1E-1
    dt, n = 0.05, 16
    ni, nj, nk, nt = n, n, 1, int(round(T/dt))
    ns = NavierStokes(linspace(-pi, pi, ni+1),
                      linspace(-pi, pi, nj+1),
                      linspace(-pi, pi, nk+1), nu, dt)

    u0 = zeros([3,ni,nj,nk])
    ux_bar_im = zeros([ni,nj,nk])
    ux_bar_ip = zeros([ni,nj,nk])
    uy_bar_jm = zeros([ni,nj,nk])
    uy_bar_jp = zeros([ni,nj,nk])
    uz_bar_km = zeros([ni,nj,nk])
    uz_bar_kp = zeros([ni,nj,nk])
    p0 = zeros([ni,nj,nk])

    x, y, z = ns.settings.xc, ns.settings.yc, ns.settings.zc
    dx, dy, dz = ns.settings.dx, ns.settings.dy, ns.settings.dz
    u0[0] = +sin(x)*cos(y)
    u0[1] = -cos(x)*sin(y)
    p0[:] = (cos(2*x) + cos(2*y))/4.0
    ux_bar_im[:] = +sin(x-dx/2)*cos(y)
    ux_bar_ip[:] = +sin(x+dx/2)*cos(y)
    uy_bar_jm[:] = -cos(x)*sin(y-dy/2)
    uy_bar_jp[:] = -cos(x)*sin(y+dy/2)

    u_bar0 = array([ux_bar_im, ux_bar_ip,
                    uy_bar_jm, uy_bar_jp,
                    uz_bar_km, uz_bar_kp])
    u_bar = array([u_bar0, u_bar0 * exp(2*nu*dt)])

    ns.init(u0, u_bar, p0)
    for i in range(nt):
        ns.timestep()
        if nsave > 0:
            if i % nsave == 0 or i == nt-1:
                write2file(ns.u, ns.p_correct, i)

    error_u = ns.u - u0*exp(-2*nu*nt*dt)
    error_p = ns.p_correct - p0*exp(-4*nu*(nt-0.5)*dt)
    error_p -= error_p.mean()
    print('ni, nj, nk, nt: ', ni, nj, nk, nt)
    print('Solution error in u and p: ', abs(error_u).max(), abs(error_p).max())

    with open('u_0.tec', 'w') as f:
        tecplot_write(f, u0*exp(-2*nu*nt*dt), p0*exp(-4*nu*(nt-0.5)*dt))
    with open('u_1.tec', 'w') as f:
        tecplot_write(f, ns.u, ns.p_correct)
    with open('u_2.tec', 'w') as f:
        tecplot_write(f, ns.u-u0*exp(-2*nu*nt*dt), ns.p_correct-p0*exp(-4*nu*(nt-0.5)*dt))
