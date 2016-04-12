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
    return u, u_tilde, p

if __name__ == '__main__':
    ni, nj, nk = 4,5,6
    u = random.random([3, ni, nj, nk])
    u_tilde = random.random([3, ni, nj, nk])
    p = random.random([ni, nj, nk])
    u, u_tilde, p = timestep(u, u_tilde, p)
