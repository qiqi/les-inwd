from numpy import *

def trapezoidal(f, y):
    fm = (f[:-1] + f[1:]) / 2.
    dy = y[1:] - y[:-1]
    
    return fm.dot(dy)    

def trapezoidal_y(f, y):
    nx, ny, nz = f.shape
    f = f.transpose((0,2,1))

    fm = (f[:,:,:-1] + f[:,:,1:]) / 2.
    dy = y[1:] - y[:-1]
    
    return fm.dot(dy)    