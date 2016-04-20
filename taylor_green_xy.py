from les import *
from les_utilities import lx, ly, lz, mu, dt
from pressure import correct_pressure

if __name__ == '__main__':
    nsave = -1
    nt = 80
    ni, nj, nk = 256,256,1
    u0 = zeros([3,ni,nj,nk])
    u_bar = zeros([3,ni+1,nj+1,nk+1])
    p0 = zeros([ni,nj,nk])

    dx = lx/float(ni)
    dy = ly/float(nj)
    dz = lz/float(nk)
        
    for k in range(nk):
        for j in range(nj):
            y = j*dy
            u_bar[0,0,j,k] = sin(-dx/2)*cos(y)
            for i in range(ni):
                x = i*dx
                u0[0,i,j,k] =  sin(x)*cos(y)
                u0[1,i,j,k] = -cos(x)*sin(y)
                p0[i,j,k] = (cos(2*x) + cos(2*y))/4.0
                u_bar[0,i+1,j,k] =  sin(x+dx/2)*cos(y)
                u_bar[1,i,j+1,k] = -cos(x)*sin(y+dy/2)
        for i in range(ni):
            x = i*dx
            u_bar[1,i,0,k] = -cos(x)*sin(-dy/2)

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
