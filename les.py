from numpy import *

from les_utilities import dt
from convection import convection
from pressure import pressure, pressure_grad

def timestep(u0, u_tilde0, p0):
    dpdx0 = pressure_grad(p0)
    u_hat = convection(u0, u_tilde0, p0, dpdx0)
    u_tilde = u_hat + dt * dpdx0
    p = pressure(u_tilde)
    dpdx = pressure_grad(p)
    u = u_tilde - dt * dpdx
    print("u kinetic energy: ", kinetic_energy(u))
    return u, u_tilde, p

def kinetic_energy(u):
    return (u**2).sum() / 2

def tecplot_write(f, u, p):
    f.write('TITLE = "LES"\n')
    f.write('VARIABLES = "X", "Y", "Z", "U-X", "U-Y", "U-Z", "P"\n')
    f.write('ZONE I={0}, J={1}, K={2}, F=BLOCK\n'.format(*p.shape))
    x, y, z = meshgrid(*tuple(range(i) for i in p.shape), indexing='ij')
    for v in [x, y, z, u[0], u[1], u[2], p]:
        f.write('\n')
        v = ravel(v.T)
        for i in range(v.size):
            f.write('{0} '.format(v[i]))

if __name__ == '__main__':
    ni, nj, nk = 32,32,32
    u = random.randn(3, ni, nj, nk)
    u_tilde = random.randn(3, ni, nj, nk)
    p = random.randn(ni, nj, nk)
    u, u_tilde, p = timestep(u, u_tilde, p)
    for i in range(100):
        u, u_tilde, p = timestep(u, u_tilde, p)
    with open('les.tec', 'wt') as f:
        tecplot_write(f, u, p)
