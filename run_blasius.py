from numpy import *
from inwd import *

nx, ny, nz = 128, 20, 2
xmin = 0.1
xmax = 1.0
nu = 1e-3

if __name__ == '__main__':
    
    dx = (xmax-xmin)/(nx+1.)
    dz = dx
    dt = dx
    
    s0 = sqrt(nu*xmax)*ones((nx,nz))
    
    y = linspace(0., 1., ny)
            
    param = {'dx':dx, 'dz':dz, 'dt':dt}
    OW = {'u':ones((nx,ny,nz)), 'v':zeros((nx,ny,nz)), 'w':zeros((nx,ny,nz)), \
          'p':0.5*ones((nx,ny,nz)), 'y':y}
    OW0 = OW
    ref = {'q':1., 'rho':1., 'nu':nu}
    
    for i in range(int(10./dt)):
        s = timestep(s0, OW, OW0, ref, param)
        s0 = s
    
    x = linspace(xmin + dx, xmax - dx, nx)
    st = sqrt(nu*x)
    error = amax(abs(s[:,0]-st))
    print error
