__all__ = ['Settings']

from numpy import *

class Settings:
    def __init__(self, x, y, z, nu, dt, tol):
        assert x.ndim == 1 and y.ndim == 1 and z.ndim == 1
        assert (x[1:] > x[:-1]).all()
        assert (y[1:] > y[:-1]).all()
        assert (z[1:] > z[:-1]).all()
        assert nu > 0 and dt > 0 and tol > 0
        self.x, self.y, self.z, self.nu, self.dt, self.tol \
                = x[:,newaxis,newaxis], \
                  y[newaxis,:,newaxis], \
                  z[newaxis,newaxis,:], nu, dt, tol

    @property
    def nx(self): return self.x.size - 1
    @property
    def ny(self): return self.y.size - 1
    @property
    def nz(self): return self.z.size - 1

    @property
    def dx(self): return diff(self.x, axis=0)
    @property
    def dy(self): return diff(self.y, axis=1)
    @property
    def dz(self): return diff(self.z, axis=2)

    @property
    def xc(self): return (self.x[1:,:,:] + self.x[:-1,:,:]) / 2
    @property
    def yc(self): return (self.y[:,1:,:] + self.y[:,:-1,:]) / 2
    @property
    def zc(self): return (self.z[:,:,1:] + self.z[:,:,:-1]) / 2
