from numpy import *
from scipy.sparse.linalg import LinearOperator, gmres

from les_utilities import dx, dy, dz, dt, extend_u, extend_p
from les_utilities import ip, im, jp, jm, kp, km, i

def residual(u_hat, u, u_tilde, p, dpdx):
    dudt = (u_hat - u) / dt
    ux_t, uy_t, uz_t = extend_u(u_tilde)
    p = extend_p(p)
    u_bar_ip = (i(ux_t) + ip(ux_t)) / 2 + (ip(p) - i(p)) / dx * dt
    u_bar_im = (i(ux_t) + im(ux_t)) / 2 - (im(p) - i(p)) / dx * dt
    u_bar_jp = (i(uy_t) + jp(uy_t)) / 2 + (jp(p) - i(p)) / dy * dt
    u_bar_jm = (i(uy_t) + jm(uy_t)) / 2 - (jm(p) - i(p)) / dy * dt
    u_bar_kp = (i(uz_t) + kp(uz_t)) / 2 + (kp(p) - i(p)) / dz * dt
    u_bar_km = (i(uz_t) + km(uz_t)) / 2 - (km(p) - i(p)) / dz * dt
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
    res = dudt + conv # + dpdx
    return res

def convection(u, u_tilde, p, dpdx):
    '''
    Residual is Ax - b
    when x=0, residual = -b, so b = -residual(0, u, um, dpdx)
    when x is not zero, Ax = residual + b
    '''

    u_hat = zeros(u.shape)
    b = -ravel(residual(u_hat, u, u_tilde, p, dpdx))
    def linear_op(u_hat):
        u_hat = u_hat.reshape(u.shape)
        res = residual(u_hat, u, u_tilde, p, dpdx)
        return ravel(res) + b
    A = LinearOperator((u.size, u.size), linear_op)
    u_hat, _ = gmres(A, b, x0=ravel(u.copy()), tol=1E-10)
    return u_hat.reshape(u.shape)
