from numpy import *
from blasius_functions_les import *
from newton import *

#----------------------------------------------------------

def timestep(s0, OW, OW0, ref, param):
    
    def F(s):
        return momentum(s, s0, OW, OW0, ref, param)
    def gradF(ds, s):
        return jacobian_momentum(ds, s, OW, ref, param)
        
    return newton(F,gradF,s0)    
    
#----------------------------------------------------------

def momentum(s, s0, OW, OW0, ref, param):
    
    dt = param['dt']
            
    deltaXi  = g_deltaXi(s, ref, OW)
    deltaXi0 = g_deltaXi(s0, ref, OW0)
    
    DdeltaXiDt = (deltaXi - deltaXi0) / dt
            
    conv = ( convection(s, ref, OW, param) \
           + convection(s0, ref, OW0, param))/2.
    
    visc = ( diffusion(s, ref, OW, param) \
           + diffusion(s0, ref, OW0, param))/2.
    
    gradp = ( pressure_gradient(s, ref, OW, param) \
            + pressure_gradient(s0, ref, OW0, param))/2.

    return DdeltaXiDt + conv + gradp - visc
    
#----------------------------------------------------------

def convection(s, ref, OW, param):

    dx = param['dx']
    dz = param['dz']

    thetaXi = g_thetaXi(s, ref, OW)
    thetaZeta = g_thetaZeta(s, ref, OW)
    vInf = g_vInf(s, ref, param)
    uInf = g_uXiInf(s, ref)
    
    DthetaXiDx = ( imm_thXi(s, ref, OW) \
                 - 4.*im_thXi(s, ref, OW) \
                 + 3.*thetaXi) / (2.*dx)
    DthetaXiDz = ( kp_thXi(s, ref, OW) \
                 - km_thXi(s, ref, OW)) / (2.*dz)
    DthetaZetaDx = ( imm_thZeta(s, ref, OW) \
                   - 4.*im_thZeta(s, ref, OW) \
                   + 3.*thetaZeta) / (2.*dx)
    DthetaZetaDz = ( kp_thZeta(s, ref, OW) \
                   - km_thZeta(s, ref, OW)) / (2.*dz)

    qr = ref['q']

    uOW = OW['u']
    wOW = OW['w']
    uXiOW = streamwise(uOW, wOW)

    vOW = OW['v']
    
    DthetaXiDXi = streamwise(uOW, wOW, DthetaXiDx, DthetaXiDz)
    DthetaZetaDZeta = transverse(uOW, wOW, DthetaZetaDx, DthetaZetaDz)

    conv = qr*( DthetaXiDXi + DthetaZetaDZeta ) \
         + (uXiOW[:,-1,:]*vOW[:,-1,:] - uXiOW[:,0,:]*vOW[:,0,:]) / qr \
         - uInf*vInf / qr

    return conv
    
#----------------------------------------------------------

def pressure_gradient(s, ref, OW, param):

    dx = param['dx']
    dz = param['dz']
    
    uOW = OW['u']
    wOW = OW['w']
    
    qr = ref['q']

    pi = g_pi(s, ref, OW)

    DpiDx = (imm_pi(s, ref, OW) - 4.*im_pi(s, ref, OW) + 3.*pi) / (2.*dx)
    DpiDz = (kp_pi(s, ref, OW) - km_pi(s, ref, OW)) / (2.*dz)
    DpiDXi = streamwise(uOW, wOW, DpiDx, DpiDz)
    
    return qr*DpiDXi/2.

#----------------------------------------------------------

def diffusion(s, ref, OW, param):

    dx = param['dx']
    dz = param['dz']

    nu = ref['nu']
    qr = ref['q']
    
    tauXi = g_tauXi(s, ref, OW, param)
    deltaXi = g_deltaXi(s, ref, OW)
    
    DdeltaXiDxx = ( ip_d(s, ref, OW, param) - 2.*deltaXi + im_d(s, ref, OW) ) / (dx**2)
    DdeltaXiDzz = ( kp_d(s, ref, OW) - 2.*deltaXi + km_d(s, ref, OW) ) / (dz**2)
    
#    visc = nu*(DdeltaXiDxx + DdeltaXiDzz) - qr*tauXi/2.
    visc = -qr*tauXi/2.
    
    return visc

#----------------------------------------------------------

def jacobian_momentum(ds, s, OW, ref, param):
    
    dt = param['dt']
            
    JdeltaXi = Jg_deltaXi(ds, s, ref, OW)
    
    DdeltaXiDt = JdeltaXi / dt
            
    conv = jacobian_convection(ds, s, ref, OW, param)/2.    
    visc = jacobian_diffusion(ds, s, ref, OW, param)/2.
    gradp = jacobian_pressure_gradient(ds, s, ref, OW, param)/2.

    return DdeltaXiDt + conv + gradp - visc

#----------------------------------------------------------

def jacobian_convection(ds, s, ref, OW, param):

    dx = param['dx']
    dz = param['dz']

    vInf = g_vInf(s, ref, param)
    uXiInf = g_uXiInf(s, ref)

    JthetaXi = Jg_thetaXi(ds, s, ref, OW)
    JvInf = Jg_vInf(ds, s, ref, param)
    JuXiInf = Jg_uXiInf(ds, s, ref)
    
    JDthetaXiDx = ( Jimm_thXi(ds, s, ref, OW) \
                  - 4.*Jim_thXi(ds, s, ref, OW) \
                  + 3.*JthetaXi) / (2.*dx)
    JDthetaXiDz = ( Jkp_thXi(ds, s, ref, OW) \
                  - Jkm_thXi(ds, s, ref, OW)) / (2.*dz)
    JDthetaZetaDx = ( Jimm_thZeta(ds, s, ref, OW) \
                    - 4.*Jim_thZeta(ds, s, ref, OW) \
                    + 3.*JthetaXi) / (2.*dx)
    JDthetaZetaDz = ( Jkp_thZeta(ds, s, ref, OW) \
                    - Jkm_thZeta(ds, s, ref, OW)) / (2.*dz)

    qr = ref['q']

    uOW = OW['u']
    wOW = OW['w']

    JDthetaXiDXi = streamwise(uOW, wOW, JDthetaXiDx, JDthetaXiDz)
    JDthetaZetaDZeta = transverse(uOW, wOW, JDthetaZetaDx, JDthetaZetaDz)

    conv = qr*( JDthetaXiDXi + JDthetaZetaDZeta ) \
         - ( uXiInf*JvInf + JuXiInf*vInf )/ qr


    return conv

#----------------------------------------------------------

def jacobian_pressure_gradient(ds, s, ref, OW, param):

    dx = param['dx']
    dz = param['dz']
    
    qr = ref['q']
    
    uOW = OW['u']
    wOW = OW['w']

    Jpi = Jg_pi(ds, s, ref, OW)
    JDpiDx = (Jimm_pi(ds, s, ref, OW) - 4.*Jim_pi(ds, s, ref, OW) + 3.*Jpi) / (2.*dx)
    JDpiDz = (Jkp_pi(ds, s, ref, OW) - Jkm_pi(ds, s, ref, OW)) / (2.*dz)
    JDpiDXi = streamwise(uOW, wOW, JDpiDx, JDpiDz)

    return qr*JDpiDXi/2.

#----------------------------------------------------------

def jacobian_diffusion(ds, s, ref, OW, param):

    dx = param['dx']
    dz = param['dz']

    qr = ref['q']
    nu = ref['nu']
    
    JtauXi = Jg_tauXi(ds, s, ref, OW)
    JdeltaXi = Jg_deltaXi(ds, s, ref, OW)
    
    JDdeltaXiDxx = ( Jip_d(ds, s, ref, OW, param) - 2.*JdeltaXi + Jim_d(ds, s, ref, OW) ) / (dx**2)
    JDdeltaXiDzz = ( Jkp_d(ds, s, ref, OW) - 2.*JdeltaXi + Jkm_d(ds, s, ref, OW) ) / (dz**2)
    
#    visc = nu*(JDdeltaXiDxx + JDdeltaXiDzz) - qr*JtauXi/2.
    visc = -qr*JtauXi/2.
    
    return visc
