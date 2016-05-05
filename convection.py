from numpy import *
from scipy.sparse.linalg import LinearOperator, gmres

import settings
from utilities import extend_u, extend_p
from utilities import ip, im, jp, jm, kp, km, i

def residual(u_hat, u, u_bar, gradp, source):

    nx, ny, nz = u.shape[1:]

    dx = settings.lx/float(nx)
    dy = settings.ly/float(ny)
    dz = settings.lz/float(nz)

    dudt = (u_hat - u) / settings.dt

    ux_bar_ip, ux_bar_im, uy_bar_jp, uy_bar_jm, uz_bar_kp, uz_bar_km = u_bar
    u = (u_hat + u) / 2
    ux, uy, uz = extend_u(u)
    conv_x = (ux_bar_ip * (i(ux) + ip(ux)) / 2 -
              ux_bar_im * (i(ux) + im(ux)) / 2) / dx \
           + (uy_bar_jp * (i(ux) + jp(ux)) / 2 -
              uy_bar_jm * (i(ux) + jm(ux)) / 2) / dy \
           + (uz_bar_kp * (i(ux) + kp(ux)) / 2 -
              uz_bar_km * (i(ux) + km(ux)) / 2) / dz
    conv_y = (ux_bar_ip * (i(uy) + ip(uy)) / 2 -
              ux_bar_im * (i(uy) + im(uy)) / 2) / dx \
           + (uy_bar_jp * (i(uy) + jp(uy)) / 2 -
              uy_bar_jm * (i(uy) + jm(uy)) / 2) / dy \
           + (uz_bar_kp * (i(uy) + kp(uy)) / 2 -
              uz_bar_km * (i(uy) + km(uy)) / 2) / dz
    conv_z = (ux_bar_ip * (i(uz) + ip(uz)) / 2 -
              ux_bar_im * (i(uz) + im(uz)) / 2) / dx \
           + (uy_bar_jp * (i(uz) + jp(uz)) / 2 -
              uy_bar_jm * (i(uz) + jm(uz)) / 2) / dy \
           + (uz_bar_kp * (i(uz) + kp(uz)) / 2 -
              uz_bar_km * (i(uz) + km(uz)) / 2) / dz
    conv = array([conv_x, conv_y, conv_z])

    visc_x = (ip(ux) + im(ux)) / dx**2 +\
             (jp(ux) + jm(ux)) / dy**2 +\
             (kp(ux) + km(ux)) / dz**2 -\
             2.0 * i(ux) * (1.0/dx**2 + 1.0/dy**2 + 1.0/dz**2)

    visc_y = (ip(uy) + im(uy)) / dx**2 +\
             (jp(uy) + jm(uy)) / dy**2 +\
             (kp(uy) + km(uy)) / dz**2 -\
             2.0 * i(uy) * (1.0/dx**2 + 1.0/dy**2 + 1.0/dz**2)

    visc_z = (ip(uz) + im(uz)) / dx**2 +\
             (jp(uz) + jm(uz)) / dy**2 +\
             (kp(uz) + km(uz)) / dz**2 -\
             2.0 * i(uz) * (1.0/dx**2 + 1.0/dy**2 + 1.0/dz**2)

    visc = settings.mu * array([visc_x, visc_y, visc_z])

    res = dudt + conv + gradp - visc - source
    return res

def convection(u, u_bar, gradp, source):
    '''
    Residual is Ax - b
    when x=0, residual = -b, so b = -residual(0, u, um, dpdx)
    when x is not zero, Ax = residual + b
    '''

    u_hat = zeros(u.shape)
    b = -ravel(residual(u_hat, u, u_bar, gradp, source))
    def linear_op(u_hat):
        u_hat = u_hat.reshape(u.shape)
        res = residual(u_hat, u, u_bar, gradp, source)
        return ravel(res) + b
    A = LinearOperator((u.size, u.size), linear_op, dtype='float64')
    u_hat, _ = gmres(A, b, x0=ravel(u.copy()), tol=settings.tol, maxiter=50)
    return u_hat.reshape(u.shape)

def velocity_mid(u_tilde, p):

    nx, ny, nz = p.shape

    dx = settings.lx/float(nx)
    dy = settings.ly/float(ny)
    dz = settings.lz/float(nz)

    ux_t, uy_t, uz_t = extend_u(u_tilde)
    p = extend_p(p)

    ux_bar_ip = (i(ux_t) + ip(ux_t)) / 2 - (ip(p) - i(p)) / dx * settings.dt
    ux_bar_im = (i(ux_t) + im(ux_t)) / 2 + (im(p) - i(p)) / dx * settings.dt
    uy_bar_jp = (i(uy_t) + jp(uy_t)) / 2 - (jp(p) - i(p)) / dy * settings.dt
    uy_bar_jm = (i(uy_t) + jm(uy_t)) / 2 + (jm(p) - i(p)) / dy * settings.dt
    uz_bar_kp = (i(uz_t) + kp(uz_t)) / 2 - (kp(p) - i(p)) / dz * settings.dt
    uz_bar_km = (i(uz_t) + km(uz_t)) / 2 + (km(p) - i(p)) / dz * settings.dt

    return array([ux_bar_ip, ux_bar_im,
                  uy_bar_jp, uy_bar_jm,
                  uz_bar_kp, uz_bar_km])
