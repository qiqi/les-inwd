import os
import sys
my_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(my_path, '..'))

from les import *
from les_utilities import lx, ly, lz

if __name__ == '__main__':
    nsave = 1
    ni, nj, nk = 32,32,32

    u = zeros([3,ni,nj,nk])
    u_tilde = zeros([3,ni,nj,nk])
    p = zeros([ni,nj,nk])

    dx = lx/float(ni)
    dy = ly/float(nj)
    dz = lz/float(nk)

    for k in range(nk):
        z = k*dz
        for j in range(nj):
            y = j*dy
            for i in range(ni):
                x = i*dx
                u[0,i,j,k] =  sin(x)*cos(y)*cos(z)
                u[1,i,j,k] = -cos(x)*sin(y)*cos(z)
                u_tilde[0,i,j,k] = sin(x)*cos(y)*cos(z) +\
                                   dt*(-2*sin(2*x) + cos(2*y))*(cos(2*z) + 2.0)/16.0
                u_tilde[1,i,j,k] = -cos(x)*sin(y)*cos(z) +\
                                   dt*(cos(2*x) - 2*sin(2*y))*(cos(2*z) + 2.0)/16.0
                u_tilde[2,i,j,k] = -dt*(cos(2*x) + cos(2*y))*sin(2*z)/8.0
                p[i,j,k] = (cos(2*x) + cos(2*y))*(cos(2*z) + 2.0)/16.0
                  
    for i in range(50):
        u, u_tilde, p = timestep(u, u_tilde, p)
        if i % nsave == 0:
             write2file(u, p, i)
