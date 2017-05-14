__all__ = ['pressure', 'pressure_grad', 'correct_pressure']

from numpy import *
from scipy.sparse.linalg import LinearOperator, cg

import utilities
from utilities import ip, im, jp, jm, kp, km, i

def _residual(settings, p, ux, uy, uz, extend_p):
    dx, dy, dz, dt = settings.dx, settings.dy, settings.dz, settings.dt
    p = extend_p(p)
    ux_ip = (i(ux) + ip(ux)) / 2 - (ip(p) - i(p)) / dx * dt
    ux_im = (i(ux) + im(ux)) / 2 + (im(p) - i(p)) / dx * dt
    uy_jp = (i(uy) + jp(uy)) / 2 - (jp(p) - i(p)) / dy * dt
    uy_jm = (i(uy) + jm(uy)) / 2 + (jm(p) - i(p)) / dy * dt
    uz_kp = (i(uz) + kp(uz)) / 2 - (kp(p) - i(p)) / dz * dt
    uz_km = (i(uz) + km(uz)) / 2 + (km(p) - i(p)) / dz * dt
    res = (ux_ip - ux_im) / dx + \
          (uy_jp - uy_jm) / dy + \
          (uz_kp - uz_km) / dz
    res[0,0,0] = p[0,0,0]
    return res

def pressure(settings, u, f_log, extend_u, extend_p):
    '''
    Residual is Ax - b
    when x=0, residual = -b, so b = -residual(0, ...)
    when x is not zero, Ax = residual + b
    '''
    ux, uy, uz = extend_u(u)
    p = zeros(u[0].shape)
    b = -ravel(_residual(settings, p, ux, uy, uz, extend_p))
    def linear_op(p):
        p = p.reshape(u[0].shape)
        res = _residual(settings, p, ux, uy, uz, extend_p)
        return ravel(res) + b
    A = LinearOperator((p.size, p.size), linear_op, dtype='float64')
    p, info = cg(A, b, tol=settings.tol, maxiter=500)
    res = _residual(settings, p.reshape(u[0].shape), ux, uy, uz, extend_p)
    f_log.write("pressure CG returns {0}, residual={1}\n".format(
                info, linalg.norm(ravel(res))))
    return p.reshape(u[0].shape)

def pressure_grad(settings, p, extend_p):
    dx, dy, dz = settings.dx, settings.dy, settings.dz
    p = extend_p(p)
    dpdx = (ip(p) - im(p)) / (2 * dx)
    dpdy = (jp(p) - jm(p)) / (2 * dy)
    dpdz = (kp(p) - km(p)) / (2 * dz)
    return array([dpdx, dpdy, dpdz])

def correct_pressure(settings, p, p0, u_bar, extend_p):
    dx, dy, dz, dt = settings.dx, settings.dy, settings.dz, settings.dt
    ux_bar_ip, ux_bar_im, uy_bar_jp, uy_bar_jm, uz_bar_kp, uz_bar_km = u_bar

    p_ext = extend_p(p - p0)
    lap_p = (ip(p_ext) + im(p_ext)) / dx**2 +\
            (jp(p_ext) + jm(p_ext)) / dy**2 +\
            (kp(p_ext) + km(p_ext)) / dz**2 -\
            2.0 * (1.0/dx**2 + 1.0/dy**2 + 1.0/dz**2) * i(p_ext)

    u_gradp = (ux_bar_ip * (i(p_ext) + ip(p_ext)) / 2 -
               ux_bar_im * (i(p_ext) + im(p_ext)) / 2) / dx \
            + (uy_bar_jp * (i(p_ext) + jp(p_ext)) / 2 -
               uy_bar_jm * (i(p_ext) + jm(p_ext)) / 2) / dy \
            + (uz_bar_kp * (i(p_ext) + kp(p_ext)) / 2 -
               uz_bar_km * (i(p_ext) + km(p_ext)) / 2) / dz

    return p - (lap_p * settings.nu - u_gradp) * dt / 2.0

