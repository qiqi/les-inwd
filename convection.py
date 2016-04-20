from numpy import *
from scipy.sparse.linalg import LinearOperator, gmres

from les_utilities import lx, ly, lz, dt, mu, gmrestol
from les_utilities import extend_u, extend_p
from les_utilities import ip, im, jp, jm, kp, km, i

def residual(u_hat, u, u_bar, gradp):

    nx, ny, nz = u.shape[1:]
    
    dx = lx/float(nx)
    dy = ly/float(ny)
    dz = lz/float(nz)
    
    dudt = (u_hat - u) / dt

    ux_bar, uy_bar, uz_bar = u_bar
    u_bar_ip = ux_bar[1:,:-1,:-1]
    u_bar_im = ux_bar[:-1,:-1,:-1]
    u_bar_jp = uy_bar[:-1,1:,:-1]
    u_bar_jm = uy_bar[:-1,:-1,:-1]
    u_bar_kp = uz_bar[:-1,:-1,1:]
    u_bar_km = uz_bar[:-1,:-1,:-1]
    
    u = (u_hat + u) / 2
    ux, uy, uz = extend_u(u)
    conv_x = (u_bar_ip * (i(ux) + ip(ux)) / 2 -
              u_bar_im * (i(ux) + im(ux)) / 2) / dx \
           + (u_bar_jp * (i(ux) + jp(ux)) / 2 -
              u_bar_jm * (i(ux) + jm(ux)) / 2) / dy \
           + (u_bar_kp * (i(ux) + kp(ux)) / 2 -
              u_bar_km * (i(ux) + km(ux)) / 2) / dz
    conv_y = (u_bar_ip * (i(uy) + ip(uy)) / 2 -
              u_bar_im * (i(uy) + im(uy)) / 2) / dx \
           + (u_bar_jp * (i(uy) + jp(uy)) / 2 -
              u_bar_jm * (i(uy) + jm(uy)) / 2) / dy \
           + (u_bar_kp * (i(uy) + kp(uy)) / 2 -
              u_bar_km * (i(uy) + km(uy)) / 2) / dz
    conv_z = (u_bar_ip * (i(uz) + ip(uz)) / 2 -
              u_bar_im * (i(uz) + im(uz)) / 2) / dx \
           + (u_bar_jp * (i(uz) + jp(uz)) / 2 -
              u_bar_jm * (i(uz) + jm(uz)) / 2) / dy \
           + (u_bar_kp * (i(uz) + kp(uz)) / 2 -
              u_bar_km * (i(uz) + km(uz)) / 2) / dz
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
    
    visc = mu * array([visc_x, visc_y, visc_z])

    res = dudt + conv + gradp - visc
    return res

def convection(u, u_bar, gradp):
    '''
    Residual is Ax - b
    when x=0, residual = -b, so b = -residual(0, u, um, dpdx)
    when x is not zero, Ax = residual + b
    '''

    u_hat = zeros(u.shape)
    b = -ravel(residual(u_hat, u, u_bar, gradp))
    def linear_op(u_hat):
        u_hat = u_hat.reshape(u.shape)
        res = residual(u_hat, u, u_bar, gradp)
        return ravel(res) + b
    A = LinearOperator((u.size, u.size), linear_op, dtype='float64')
    u_hat, _ = gmres(A, b, x0=ravel(u.copy()), tol=gmrestol, maxiter=50)
    return u_hat.reshape(u.shape)
