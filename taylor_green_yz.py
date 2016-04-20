from les import *
from les_utilities import lx, ly, lz

if __name__ == '__main__':
    nsave = -1
    nt = 80
    ni, nj, nk = 1,256,256

    u0 = zeros([3,ni,nj,nk])
    u_tilde = zeros([3,ni,nj,nk])
    p0 = zeros([ni,nj,nk])

    dx = lx/float(ni)
    dy = ly/float(nj)
    dz = lz/float(nk)

    for k in range(nk):
        z = k*dz
        for j in range(nj):
            y = j*dy
            for i in range(ni):
                x = i*dx
                u0[1,i,j,k] =  sin(y)*cos(z)
                u0[2,i,j,k] = -cos(y)*sin(z)
                u_tilde[1,i,j,k] = sin(y)*cos(z) +\
                                   dt*(-2*sin(2*y) + cos(2*z))/4.0
                u_tilde[2,i,j,k] = -cos(y)*sin(z) +\
                                   dt*(cos(2*y) - 2*sin(2*z))/4.0
                p0[i,j,k] = (cos(2*y) + cos(2*z))/4.0

    u = u0
    p = p0
    for i in range(nt):
        u, u_tilde, p = timestep(u, u_tilde, p)
        if nsave > 0:
            if i % nsave == 0:
                write2file(u, p, i)
        elif nsave == 0:
            if i == nt-1:
                write2file(u, p, i)
    
    error_u = amax(abs(u-u0))
    error_p = amax(abs(p-p0))
    print error_u, error_p
