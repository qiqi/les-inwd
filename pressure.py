import time
from numpy import *
from scipy.sparse.linalg import LinearOperator, gmres, cg

from les_utilities import dx, dy, dz, dt, extend_u, extend_p
from les_utilities import ip, im, jp, jm, kp, km, i

def residual(p, ux, uy, uz):
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
    t0 = time.time()
    p, _ = cg(A, b, tol=1E-8, maxiter=500)
    print('pressure solver: {0}/{1} after {2:4.1f}s'.format(
        linalg.norm(A * p - b), linalg.norm(b), time.time() - t0
    ))
    return p.reshape(u[0].shape)

def pressure_grad(p):
    p = extend_p(p)
    dpdx = (ip(p) - im(p)) / (2 * dx)
    dpdy = (jp(p) - jm(p)) / (2 * dy)
    dpdz = (kp(p) - km(p)) / (2 * dz)
    return array([dpdx, dpdy, dpdz])

