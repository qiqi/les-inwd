from convection import convection
from pressure import pressure, correction

def timestep(u0, u_tilde0, p0, dt):
    dpdx0 = pressure_grad(p0)
    u_hat = convection(u0, u_tilde0, p0, dpdx0, dt)
    u_tilde = u_hat + dt * dpdx0
    p = pressure(u_tilde)
    dpdx = pressure_grad(p)
    u = u_tilde - dt * dpdx
    return u, u_tilde, p, dpdx
