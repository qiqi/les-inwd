from numpy import *

from les_utilities import dt
from convection import convection
from pressure import pressure, pressure_grad

def timestep(u0, u_tilde0, p0):
    dpdx0 = pressure_grad(p0)
    print("u0: ", kinetic_energy(u0))
    u_hat = convection(u0, u_tilde0, p0, dpdx0)
    print("u_tilde0: ", kinetic_energy(u_hat))
    u_tilde = u_hat + dt * dpdx0
    p = pressure(u_tilde)
    dpdx = pressure_grad(p)
    u = u_tilde - dt * dpdx
    print("u: ", kinetic_energy(u))
    return u, u_tilde, p

def kinetic_energy(u):
    return (u**2).sum() / 2

def d_kinetic_energy_dt(u, dudt):
    return (u*dudt).sum()

if __name__ == '__main__':
    ni, nj, nk = 16,16,128
    u = random.random([3, ni, nj, nk])
    u_tilde = random.random([3, ni, nj, nk])
    p = random.random([ni, nj, nk])
    u, u_tilde, p = timestep(u, u_tilde, p)
    for i in range(100):
        u, u_tilde, p = timestep(u, u_tilde, p)
