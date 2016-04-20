from numpy import *
from scipy.sparse.linalg import LinearOperator, gmres

from les_utilities import lx, ly, lz, dt, extend_u, extend_p, gmrestol, mu
from les_utilities import ip, im, jp, jm, kp, km, i

def residual(p, ux, uy, uz):

    nx, ny, nz = p.shape

    dx = lx/float(nx)
    dy = ly/float(ny)
    dz = lz/float(nz)

    p = extend_p(p)
    ux_ip = (i(ux) + ip(ux)) / 2 - (ip(p) - i(p)) / dx * dt
    ux_im = (i(ux) + im(ux)) / 2 + (im(p) - i(p)) / dx * dt
    uy_jp = (i(uy) + jp(uy)) / 2 - (jp(p) - i(p)) / dy * dt
    uy_jm = (i(uy) + jm(uy)) / 2 + (jm(p) - i(p)) / dy * dt
    uz_kp = (i(uz) + kp(uz)) / 2 - (kp(p) - i(p)) / dz * dt
    uz_km = (i(uz) + km(uz)) / 2 + (km(p) - i(p)) / dz * dt
    return (ux_ip - ux_im) / dx + \
           (uy_jp - uy_jm) / dy + \
           (uz_kp - uz_km) / dz

def pressure(u):
    '''
    Residual is Ax - b
    when x=0, residual = -b, so b = -residual(0, u, um, dpdx)
    when x is not zero, Ax = residual + b
    '''
    ux, uy, uz = extend_u(u)
    p = zeros(u[0].shape)
    b = -ravel(residual(p, ux, uy, uz))
    def linear_op(p):
        p = p.reshape(u[0].shape)
        res = residual(p, ux, uy, uz)
        return ravel(res) + b
    A = LinearOperator((p.size, p.size), linear_op, dtype='float64')
    p, _ = gmres(A, b, tol=gmrestol, maxiter=200)
    return p.reshape(u[0].shape)

def pressure_grad(p):

    nx, ny, nz = p.shape

    dx = lx/float(nx)
    dy = ly/float(ny)
    dz = lz/float(nz)

    p = extend_p(p)   
    dpdx = (ip(p) - im(p)) / (2 * dx)
    dpdy = (jp(p) - jm(p)) / (2 * dy)
    dpdz = (kp(p) - km(p)) / (2 * dz)
    return array([dpdx, dpdy, dpdz])

def correct_pressure(p, p0, u_bar):

    nx, ny, nz = p.shape

    dx = lx/float(nx)
    dy = ly/float(ny)
    dz = lz/float(nz)

    ux_bar, uy_bar, uz_bar = u_bar
    u_bar_ip = ux_bar[1:,:-1,:-1]
    u_bar_im = ux_bar[:-1,:-1,:-1]
    u_bar_jp = uy_bar[:-1,1:,:-1]
    u_bar_jm = uy_bar[:-1,:-1,:-1]
    u_bar_kp = uz_bar[:-1,:-1,1:]
    u_bar_km = uz_bar[:-1,:-1,:-1]

    p_ext = extend_p(p - p0)
    lap_p = (ip(p_ext) + im(p_ext)) / dx**2 +\
            (jp(p_ext) + jm(p_ext)) / dy**2 +\
            (kp(p_ext) + km(p_ext)) / dz**2 -\
            2.0 * (1.0/dx**2 + 1.0/dy**2 + 1.0/dz**2) * i(p_ext)
    
    u_gradp = (u_bar_ip * (i(p_ext) + ip(p_ext)) / 2 -
               u_bar_im * (i(p_ext) + im(p_ext)) / 2) / dx \
            + (u_bar_jp * (i(p_ext) + jp(p_ext)) / 2 -
               u_bar_jm * (i(p_ext) + jm(p_ext)) / 2) / dy \
            + (u_bar_kp * (i(p_ext) + kp(p_ext)) / 2 -
               u_bar_km * (i(p_ext) + km(p_ext)) / 2) / dz
    
    return p - lap_p * dt * mu / 2.0 + u_gradp * dt / 2.0
