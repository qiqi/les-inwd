import os
import sys
my_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(my_path, '..'))

from les import *
import settings
from pressure import correct_pressure
from numpy import linspace, pi, cos, sin

if __name__ == '__main__':
    nsave = -1
    nt = 40
    ni, nj, nk = 64,64,1
    settings.x = linspace(-pi, pi, ni+1)
    settings.y = linspace(-pi, pi, nj+1)
    settings.z = linspace(-pi, pi, nk+1)
    dt = settings.dt
    mu = settings.mu
    u0 = zeros([3,ni,nj,nk])
    ux_bar_im = zeros([ni,nj,nk])
    ux_bar_ip = zeros([ni,nj,nk])
    uy_bar_jm = zeros([ni,nj,nk])
    uy_bar_jp = zeros([ni,nj,nk])
    uz_bar_km = zeros([ni,nj,nk])
    uz_bar_kp = zeros([ni,nj,nk])
    p0 = zeros([ni,nj,nk])

    dx = diff(settings.x)[:,newaxis,newaxis]
    dy = diff(settings.y)[newaxis,:,newaxis]
    dz = diff(settings.z)[newaxis,newaxis,:]

    x = (settings.x[1:] + settings.x[:-1])[:,newaxis,newaxis] / 2
    y = (settings.y[1:] + settings.y[:-1])[newaxis,:,newaxis] / 2
    z = (settings.z[1:] + settings.z[:-1])[newaxis,newaxis,:] / 2

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
    u_bar = array([u_bar0, u_bar0 * exp(2*mu*dt)])
    u = u0
    p = p0
    for i in range(nt):
        pm = p
        u, u_bar, p = timestep(u, u_bar, p)
        if nsave > 0:
            if i % nsave == 0:
                write2file(u, p, i)
            elif nsave == 0:
                if i == nt-1:
                    write2file(u, p, i)

    p = correct_pressure(p, pm, (u_bar[0] * 3 - u_bar[1]) / 2)
    error_u = u-u0*exp(-2*mu*nt*dt)
    error_p = p-p0*exp(-4*mu*(nt-0.5)*dt)
    error_p -= error_p.mean()
    print('ni, nj, nk, nt: ', ni, nj, nk, nt)
    print('Solution error in u and p: ', abs(error_u).max(), abs(error_p).max())

    with open('u_0.tec', 'w') as f:
        tecplot_write(f, u0*exp(-2*mu*nt*dt), p0*exp(-4*mu*(nt-0.5)*dt))
    with open('u_1.tec', 'w') as f:
        tecplot_write(f, u, p)
    with open('u_2.tec', 'w') as f:
        tecplot_write(f, u-u0*exp(-2*mu*nt*dt), p-p0*exp(-4*mu*(nt-0.5)*dt))
