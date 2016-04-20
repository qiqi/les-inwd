from les import *
from les_utilities import lx, ly, lz, mu, dt
from pressure import correct_pressure

if __name__ == '__main__':
    nsave = -1
    nt = 40
    ni, nj, nk = 128,1,128
    u0 = zeros([3,ni,nj,nk])
    u_bar = zeros([3,ni+1,nj+1,nk+1])
    p0 = zeros([ni,nj,nk])

    dx = lx/float(ni)
    dy = ly/float(nj)
    dz = lz/float(nk)
        
    for k in range(nk):
        z = k*dz
        for j in range(nj):
            u_bar[0,0,j,k] = sin(-dx/2)*cos(z)
            for i in range(ni):
                x = i*dx
                u0[0,i,j,k] =  sin(x)*cos(z)
                u0[2,i,j,k] = -cos(x)*sin(z)
                p0[i,j,k] = (cos(2*x) + cos(2*z))/4.0
                u_bar[0,i+1,j,k] =  sin(x+dx/2)*cos(z)
                u_bar[2,i,j,k+1] = -cos(x)*sin(z+dz/2)

    for j in range(nj):
        for i in range(ni):
            x = i*dx
            u_bar[2,i,j,0] = -cos(x)*sin(-dz/2)

    u = u0
    p = p0
    u_barm = u_bar * exp(2*mu*dt)
    for i in range(nt):
        pm = p
        u_bar_ext = 1.5 * u_bar - 0.5 * u_barm
        u_barm = u_bar
        u, u_bar, p = timestep(u, u_bar_ext, p)
        if nsave > 0:
            if i % nsave == 0:
                write2file(u, p, i)
            elif nsave == 0:
                if i == nt-1:
                    write2file(u, p, i)
        
    p = correct_pressure(p, pm, u_bar_ext)
    error_u = amax(abs(u-u0*exp(-2*mu*nt*dt)))
    error_p = amax(abs(p-p0*exp(-4*mu*(nt-0.5)*dt)))
    print error_u, error_p
