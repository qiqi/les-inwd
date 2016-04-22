from numpy import *

from les_utilities import lx, ly, lz, dt, mu, gmrestol
from les_utilities import tecplot_write, ip, im, jp, jm, kp, km, i
from les_utilities import extend_u, extend_p, velocity_mid
from convection import convection
from pressure import pressure, pressure_grad

def timestep(u0, u_barph, p0):
    gradp0 = pressure_grad(p0)
    print("u0: ", kinetic_energy(u0))
    u_hat = convection(u0, u_barph, gradp0)
    print("u_tilde0: ", kinetic_energy(u_hat))
    u_tilde = u_hat + dt * gradp0
    p = pressure(u_tilde)
    gradp = pressure_grad(p)
    u = u_tilde - dt * gradp
    u_bar = velocity_mid(u_tilde, p)
    print("u: ", kinetic_energy(u))
    return u, u_bar, p
    
    
def kinetic_energy(u):
    return (u**2).sum() / 2

def d_kinetic_energy_dt(u, dudt):
    return (u*dudt).sum()
   
if __name__ == '__main__':
    nsave = 5
    ni, nj, nk = 32, 32, 32
    u = random.random([3, ni, nj, nk])
    u_bar = random.random([3, ni+1, nj+1, nk+1])
    p = random.random([ni, nj, nk])
    
    u, u_bar, p = timestep(u, u_bar, p)
    for i in range(100):
        u, u_bar, p = timestep(u, u_bar, p)
        if i % nsave == 0:
            with open('sol_{0}.tec'.format(i/nsave), 'w') as f:
                tecplot_write(f, u, p)
