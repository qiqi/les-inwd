from numpy import *

#----------------------------------------------------------

""" Peridic boundary conditions for 2-D array """

#----------------------------------------------------------

def ip_per_2d(x):
    ipx = zeros(x.shape)
    ipx[:-1,:] = x[1:,:]
    ipx[-1,:] = x[0,:]
    return ipx
    
#----------------------------------------------------------

def im_per_2d(x):
    imx = zeros(x.shape)
    imx[1:,:] = x[:-1,:]
    imx[0,:] = x[-1,:]
    return imx

#----------------------------------------------------------

def imm_per_2d(x):
    immx = zeros(x.shape)
    immx[2:,:] = x[:-2,:]
    immx[1,:] = x[-1,:]
    immx[0,:] = x[-2,:]
    return immx

#----------------------------------------------------------

def kp_per_2d(x):
    kpx = zeros(x.shape)
    kpx[:,:-1] = x[:,1:]
    kpx[:,-1]  = x[:,0]
    return kpx

#----------------------------------------------------------

def km_per_2d(x):
    kmx = zeros(x.shape)
    kmx[:,1:] = x[:,:-1]
    kmx[:,0]  = x[:,-1]
    return kmx
        
#----------------------------------------------------------

""" Peridic boundary conditions for 3-D array """

#----------------------------------------------------------

def ip_per_3d(x):
    ipx = zeros(x.shape)
    ipx[:-1,:,:] = x[1:,:,:]
    ipx[-1,:,:] = x[0,:,:]
    return ipx

#----------------------------------------------------------

def im_per_3d(x):
    imx = zeros(x.shape)
    imx[1:,:,:] = x[:-1,:,:]
    imx[0,:,:] = x[-1,:,:]
    return imx

#----------------------------------------------------------

def imm_per_3d(x):
    immx = zeros(x.shape)
    immx[2:,:,:] = x[:-2,:,:]
    immx[1,:,:] = x[-1,:,:]
    immx[0,:,:] = x[-2,:,:]
    return immx

#----------------------------------------------------------

def kp_per_3d(x):
    kpx = zeros(x.shape)
    kpx[:,:,:-1] = x[:,:,1:]
    kpx[:,:,-1] = x[:,:,0]
    return kpx

#----------------------------------------------------------

def km_per_3d(x):
    kmx = zeros(x.shape)
    kmx[:,:,1:] = x[:,:,:-1]
    kmx[:,:,0] = x[:,:,-1]
    return kmx

#----------------------------------------------------------

""" Dirichlet boundary conditions for 2-D array """

#----------------------------------------------------------

def ip_dir_2d(x, value):
    ipx = zeros(x.shape)
    ipx[:-1,:] = x[1:,:]
    ipx[-1,:] = value
    return ipx
    
#----------------------------------------------------------

def im_dir_2d(x, value):
    imx = zeros(x.shape)
    imx[1:,:] = x[:-1,:]
    imx[0,:] = value
    return imx

#----------------------------------------------------------

def imm_dir_2d(x, value):
    immx = zeros(x.shape)
    immx[2:,:] = x[:-2,:]
    immx[1,:] = value
    immx[0,:] = 3.*value - 3.*x[0,:] + x[1,:]
    return immx

#----------------------------------------------------------

def kp_dir_2d(x, value):
    kpx = zeros(x.shape)
    kpx[:,:-1] = x[:,1:]
    kpx[:,-1] = value
    return kpx
    
#----------------------------------------------------------

def km_dir_2d(x, value):
    kmx = zeros(x.shape)
    kmx[:,1:] = x[:,:-1]
    kmx[:,0] = value
    return kmx

#----------------------------------------------------------

""" Dirichlet boundary conditions for 3-D array """

#----------------------------------------------------------

def ip_dir_3d(x, value):
    ipx = zeros(x.shape)
    ipx[:-1,:,:] = x[1:,:,:]
    ipx[-1,:,:] = value
    return ipx
    
#----------------------------------------------------------

def im_dir_3d(x, value):
    imx = zeros(x.shape)
    imx[1:,:,:] = x[:-1,:,:]
    imx[0,:,:] = value
    return imx

#----------------------------------------------------------

def imm_dir_3d(x, value):
    immx = zeros(x.shape)
    immx[2:,:,:] = x[:-2,:,:]
    immx[1,:,:] = value
    immx[0,:,:] = 3.*value - 3.*x[0,:,:] + x[1,:,:]
    return immx

#----------------------------------------------------------

def kp_dir_3d(x, value):
    kpx = zeros(x.shape)
    kpx[:,:,:-1] = x[:,:,1:]
    kpx[:,:,-1] = value
    return kpx
    
#----------------------------------------------------------

def km_dir_3d(x, value):
    kmx = zeros(x.shape)
    kmx[:,:,1:] = x[:,:,:-1]
    kmx[:,:,0] = value
    return kmx

#----------------------------------------------------------

""" Neumann boundary conditions for 2-D array """

#----------------------------------------------------------

def ip_neu_2d(x, param, value):

    dx = param['dx']

    ipx = zeros(x.shape)
    ipx[:-1,:] = x[1:,:]
    ipx[-1,:] = 2.*dx*value + x[-2,:]
    return ipx
    
#----------------------------------------------------------

def im_neu_2d(x, param, value):
    
    dx = param['dx']
    
    imx = zeros(x.shape)
    imx[1:,:] = x[:-1,:]
    imx[0,:] = x[1,:] - 2.*dx*value
    return imx

#----------------------------------------------------------

def imm_neu_2d(x, param, value):
    
    dx = param['dx']

    immx = zeros(x.shape)
    immx[2:,:] = x[:-2,:]
    immx[1,:] = x[1,:] - 2.*dx*value
    immx[0,:] = -3.*x[0,:] + 4.*x[1,:] - 6.*dx*value
    return immx

#----------------------------------------------------------

def kp_neu_2d(x, param, value):

    dx = param['dx']

    kpx = zeros(x.shape)
    kpx[:,:-1] = x[:,1:]
    kpx[:,-1] = 2.*dx*value + x[:,-2]
    return kpx
    
#----------------------------------------------------------

def km_neu_2d(x, param, value):

    dx = param['dx']
    
    kmx = zeros(x.shape)
    kmx[:,1:] = x[:,:-1]
    kmx[:,0] = x[:,1] - 2.*dx*value
    return kmx

#----------------------------------------------------------

""" Neumann boundary conditions for 3-D array """

#----------------------------------------------------------

def ip_neu_3d(x, param, value):

    dx = param['dx']

    ipx = zeros(x.shape)
    ipx[:-1,:,:] = x[1:,:,:]
    ipx[-1,:,:] = 2.*dx*value + x[-2,:,:]
    return ipx
    
#----------------------------------------------------------

def im_neu_3d(x, param, value):
    
    dx = param['dx']
    
    imx = zeros(x.shape)
    imx[1:,:,:] = x[:-1,:,:]
    imx[0,:,:] = x[1,:,:] - 2.*dx*value
    return imx

#----------------------------------------------------------

def imm_neu_3d(x, param, value):
    
    dx = param['dx']

    immx = zeros(x.shape)
    immx[2:,:,:] = x[:-2,:,:]
    immx[1,:,:] = x[1,:,:] - 2.*dx*value
    immx[0,:,:] = -3.*x[0,:,:] + 4.*x[1,:,:] - 6.*dx*value
    return immx

#----------------------------------------------------------

def kp_neu_3d(x, param, value):

    dx = param['dx']

    kpx = zeros(x.shape)
    kpx[:,:,:-1] = x[:,:,1:]
    kpx[:,:,-1] = 2.*dx*value + x[:,:,-2]
    return kpx
    
#----------------------------------------------------------

def km_neu_3d(x, param, value):

    dx = param['dx']
    
    kmx = zeros(x.shape)
    kmx[:,:,1:] = x[:,:,:-1]
    kmx[:,:,0] = x[:,:,1] - 2.*dx*value
    return kmx
