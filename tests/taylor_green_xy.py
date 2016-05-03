import os
import sys
my_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(my_path, '..'))

from les import *
from les_utilities import lx, ly, lz, mu, dt
from pressure import correct_pressure

if __name__ == '__main__':
    nsave = -1
    nt = 40
    ni, nj, nk = 32,32,1
    u0 = zeros([3,ni,nj,nk])
    ux_bar_im = zeros([ni,nj,nk])
    ux_bar_ip = zeros([ni,nj,nk])
    uy_bar_jm = zeros([ni,nj,nk])
    uy_bar_jp = zeros([ni,nj,nk])
    uz_bar_km = zeros([ni,nj,nk])
    uz_bar_kp = zeros([ni,nj,nk])
    p0 = zeros([ni,nj,nk])

    dx = lx/float(ni)
    dy = ly/float(nj)
    dz = lz/float(nk)

    for k in range(nk):
        for j in range(nj):
            y = j*dy
            for i in range(ni):
                x = i*dx
                u0[0,i,j,k] =  sin(x)*cos(y)
                u0[1,i,j,k] = -cos(x)*sin(y)
                p0[i,j,k] = (cos(2*x) + cos(2*y))/4.0
                ux_bar_im[i,j,k] = sin(x-dx/2)*cos(y)
                ux_bar_ip[i,j,k] = sin(x+dx/2)*cos(y)
                uy_bar_jm[i,j,k] = -cos(x)*sin(y-dy/2)
                uy_bar_jp[i,j,k] = -cos(x)*sin(y+dy/2)

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
    error_u = amax(abs(u-u0*exp(-2*mu*nt*dt)))
    error_p = amax(abs(p-p0*exp(-4*mu*(nt-0.5)*dt)))
    print('ni, nj, nk, nt: ', ni, nj, nk, nt)
    print('Solution error in u and p: ', error_u, error_p)
