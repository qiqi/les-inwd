from numpy import *

gmrestol = 1e-10
lx = 0.025
ly = 2*pi
lz = 2*pi
dt = 0.025

def ip(u):
    return u[2:,1:-1,1:-1]

def im(u):
    return u[:-2,1:-1,1:-1]

def jp(u):
    return u[1:-1,2:,1:-1]

def jm(u):
    return u[1:-1,:-2,1:-1]

def kp(u):
    return u[1:-1,1:-1,2:]

def km(u):
    return u[1:-1,1:-1,:-2]

def i(u):
    return u[1:-1,1:-1,1:-1]

def extend_p(p):

    p_ext = zeros([p.shape[0]+2, p.shape[1]+2, p.shape[2]+2])
    p_ext[1:-1,1:-1,1:-1] = p
    
    p_ext[1:-1,1:-1,0] = p[:,:,-1]
    p_ext[1:-1,1:-1,-1] = p[:,:,0]
    p_ext[1:-1,0,1:-1] = p[:,-1,:]
    p_ext[1:-1,-1,1:-1] = p[:,0,:]
    p_ext[0,1:-1,1:-1] = p[-1,:,:]
    p_ext[-1,1:-1,1:-1] = p[0,:,:]

    return p_ext

def extend_u(u):
    u_ext = zeros([u.shape[0], u.shape[1]+2, u.shape[2]+2, u.shape[3]+2])
    u_ext[:,1:-1,1:-1,1:-1] = u
    u_ext[:,1:-1,1:-1,0] = u[:,:,:,-1]
    u_ext[:,1:-1,1:-1,-1] = u[:,:,:,0]
    u_ext[:,1:-1,0,1:-1] = u[:,:,-1,:]
    u_ext[:,1:-1,-1,1:-1] = u[:,:,0,:]
    u_ext[:,0,1:-1,1:-1] = u[:,-1,:,:]
    u_ext[:,-1,1:-1,1:-1] = u[:,0,:,:]

    return u_ext

def write2file(u, p, i):

    nx, ny, nz = p.shape

    dx = lx/float(nx)
    dy = ly/float(ny)
    dz = lz/float(nz)
      
    if i == 0:
        out = open('solution.dat','w',0)
        out.write('TTILE = "LES Integral Near Wall Defficit solution"\n')
        out.write('FILETYPE = FULL\n')
        out.write('VARIABLES = "X", "Y", "Z", "U", "V", "W", "P"\n')
        out.write('ZONE T = "t = {:10.3E}"\n'.format(i*dt))
        out.write('STRANDID = 1\n'.format(i*dt))
        out.write('SOLUTIONTIME = {:10.3E}\n'.format(i*dt))
        out.write('I = {:d}, J = {:d}, K = {:d}\n'.format(nx,ny,nz))
        out.write('ZONETYPE = Ordered, DATAPACKING = BLOCK\n')
        for j in range(nz):
            for k in range(ny):
                for l in range(nx):
                    out.write('{:10.3E}\n'.format(dx*l))
        for j in range(nz):
            for k in range(ny):
                for l in range(nx):
                    out.write('{:10.3E}\n'.format(dy*k))
        for j in range(nz):
            for k in range(ny):
                for l in range(nx):
                    out.write('{:10.3E}\n'.format(dz*j))
    else:
        out = open('solution.dat','a',0)
        out.write('ZONE T = "t = {:10.3E}"\n'.format(i*dt))
        out.write('STRANDID = 1\n'.format(i*dt))
        out.write('SOLUTIONTIME = {:10.3E}\n'.format(i*dt))
        out.write('I = {:d}, J = {:d}, K = {:d}\n'.format(nx,ny,nz))
        out.write('ZONETYPE = Ordered, DATAPACKING = BLOCK\n')
        out.write('VARSHARELIST=([1-3]=1)\n')

    ux, uy, uz = u
    ux = ravel(ux)
    uy = ravel(uy)
    uz = ravel(uz)
    for j in range(nx*ny*nz):
        out.write('{:10.3E}\n'.format(ux[j]))
    for j in range(nx*ny*nz):
        out.write('{:10.3E}\n'.format(uy[j]))
    for j in range(nx*ny*nz):
        out.write('{:10.3E}\n'.format(uz[j]))
    p = ravel(p)
    for j in range(p.shape[0]):
        out.write('{:10.3E}\n'.format(p[j]))
    
    out.close()
    
