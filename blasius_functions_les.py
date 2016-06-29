from run_blasius import xmin, xmax, nu
from quadrature import *
from interpolation import *
from neighbors import *
from numpy import *

#----------------------------------------------------------

eta_ref = linspace(0,8.8,45)
f_ref = array([0.000000000E+00,6.640999715E-03,2.655988402E-02,5.973463750E-02,\
               1.061082208E-01,1.655717258E-01,2.379487173E-01,3.229815738E-01,\
               4.203207655E-01,5.295180377E-01,6.500243699E-01,7.811933370E-01,\
               9.222901256E-01,1.072505977E+00,1.230977302E+00,1.396808231E+00,\
               1.569094960E+00,1.746950094E+00,1.929525170E+00,2.116029817E+00,\
               2.305746418E+00,2.498039663E+00,2.692360938E+00,2.888247990E+00,\
               3.085320655E+00,3.283273665E+00,3.481867612E+00,3.680919063E+00,\
               3.880290678E+00,4.079881939E+00,4.279620923E+00,4.479457297E+00,\
               4.679356615E+00,4.879295811E+00,5.079259772E+00,5.279238811E+00,\
               5.479226847E+00,5.679220147E+00,5.879216466E+00,6.079214481E+00,\
               6.279213431E+00,6.479212887E+00,6.679212609E+00,6.879212471E+00,\
               7.079212403E+00])

fp_ref = array([0.000000000E+00,6.640779210E-02,1.327641608E-01,1.989372524E-01,\
                2.647091387E-01,3.297800312E-01,3.937761044E-01,4.562617647E-01,\
                5.167567844E-01,5.747581439E-01,6.297657365E-01,6.813103772E-01,\
                7.289819351E-01,7.724550211E-01,8.115096232E-01,8.460444437E-01,\
                8.760814552E-01,9.017612214E-01,9.233296659E-01,9.411179967E-01,\
                9.555182298E-01,9.669570738E-01,9.758708321E-01,9.826835008E-01,\
                9.877895262E-01,9.915419002E-01,9.942455354E-01,9.961553040E-01,\
                9.974777682E-01,9.983754937E-01,9.989728724E-01,9.993625417E-01,\
                9.996117017E-01,9.997678702E-01,9.998638190E-01,9.999216041E-01,\
                9.999557173E-01,9.999754577E-01,9.999866551E-01,9.999928812E-01,\
                9.999962745E-01,9.999980875E-01,9.999990369E-01,9.999995242E-01,\
                9.999997695E-01])

fpp_ref = array([3.320573362E-01,3.319838371E-01,3.314698442E-01,3.300791276E-01,\
                 3.273892701E-01,3.230071167E-01,3.165891911E-01,3.078653918E-01,\
                 2.966634615E-01,2.829310173E-01,2.667515457E-01,2.483509132E-01,\
                 2.280917607E-01,2.064546268E-01,1.840065939E-01,1.613603195E-01,\
                 1.391280556E-01,1.178762461E-01,9.808627878E-02,8.012591814E-02,\
                 6.423412109E-02,5.051974749E-02,3.897261085E-02,2.948377201E-02,\
                 2.187118635E-02,1.590679869E-02,1.134178897E-02,7.927659815E-03,\
                 5.431957680E-03,3.648413667E-03,2.402039844E-03,1.550170691E-03,\
                 9.806151170E-04,6.080442648E-04,3.695625701E-04,2.201689553E-04,\
                 1.285698072E-04,7.359298339E-05,4.129031111E-05,2.270775140E-05,\
                 1.224092624E-05,6.467978611E-06,3.349939753E-06,1.700667989E-06,\
                 8.462841214E-07])

#----------------------------------------------------------

def f(eta):
    return cubic_interpolation(eta_ref,f_ref,fp_ref,eta)

#----------------------------------------------------------

def fp(eta):
    return cubic_interpolation(eta_ref,fp_ref,fpp_ref,eta)
                
#----------------------------------------------------------

def fpp(eta):
    return linear_interpolation(eta_ref,fpp_ref,eta)

#----------------------------------------------------------

int_fp = trapezoidal(fp_ref,eta_ref)
int_fp2 = trapezoidal(fp_ref**2,eta_ref)
int_fpp_eta = trapezoidal(fpp_ref*eta_ref,eta_ref)
int_fp_fpp_eta = trapezoidal(fp_ref*fpp_ref*eta_ref,eta_ref)
vc = eta_ref[-1]*fp_ref[-1] - f_ref[-1]

#----------------------------------------------------------

def streamwise(u, w, x = [], z = []):

    uInf = u[:,-1,:]
    wInf = w[:,-1,:]
    qInf = sqrt(uInf**2 + wInf**2)
    
    ny = size(u,1)

    xDotXi = uInf / qInf
    zDotXi = wInf / qInf

    if size(x) == 0:
        x = u
        xDotXi = extend_y(xDotXi, ny)
    if size(z) == 0:
        z = w
        zDotXi = extend_y(zDotXi, ny)

    
    return xDotXi*x + zDotXi*z

#----------------------------------------------------------

def transverse(u, w, x = [], z = []):

    uInf = u[:,-1,:]
    wInf = w[:,-1,:]
    qInf = sqrt(uInf**2 + wInf**2)
    
    ny = size(u,1)

    xDotZeta = -wInf / qInf
    zDotZeta =  uInf / qInf
    if size(x) == 0:
        x = u
        xDotZeta = extend_y(xDotZeta,ny)
    if size(z) == 0:
        z = w
        zDotZeta = extend_y(zDotZeta,ny)
        
    return xDotZeta*x + zDotZeta*z

#----------------------------------------------------------

def g_deltaXi(s, ref, OW):

    qr = ref['q']

    uOW = OW['u']
    wOW = OW['w']
    y = OW['y']            
    
    uXiOW = streamwise(uOW, wOW)
    ye = eta_ref[-1]*s

    return trapezoidal_y(uXiOW / qr,y) - (int_fp*s + y[-1] - ye)
        
#----------------------------------------------------------

def g_thetaXi(s, ref, OW):

    uOW = OW['u']
    wOW = OW['w']
    y = OW['y']
            
    qr = ref['q']
        
    uXiOW = streamwise(uOW, wOW)        
    ye = eta_ref[-1]*s
                
    return trapezoidal_y((uXiOW / qr)**2,y) - (int_fp2*s + y[-1] - ye) 

#----------------------------------------------------------

def g_thetaZeta(s, ref, OW):

    uOW = OW['u']
    wOW = OW['w']
    y = OW['y']
            
    qr = ref['q']
        
    uZetaOW = transverse(uOW, wOW)        
                
    return trapezoidal_y((uZetaOW / qr)**2,y)

#----------------------------------------------------------

def g_vInf(s, ref, param):

    dx = param['dx']
    qr = ref['q']
        
    DsDx = (imm_s(s) - 4.*im_s(s) + 3.*s) / (2.*dx)

    return vc*qr*DsDx

#----------------------------------------------------------

def g_uXiInf(s, ref):
    qr = ref['q']
    return qr

#----------------------------------------------------------

def g_tauXi(s, ref, OW, param):

    dx = param['dx']
    
    nu = ref['nu']
    qr = ref['q']

    uOW = OW['u']
    wOW = OW['w']
    uXiOW = streamwise(uOW, wOW)

    DuXiOWDy = (-3.*uXiOW[:,0,:] + 4.*uXiOW[:,1,:] - uXiOW[:,2,:]) / (2.*dx)

    DuXiDy = qr*fpp_ref[0] / s
        
    return 2.*nu*(DuXiOWDy - DuXiDy) / (qr**2)

#----------------------------------------------------------

def g_pi(s, ref, OW):
    
    rho = ref['rho']
    qr = ref['q']
    
    pOW = OW['p']
    y = OW['y']
    
    return trapezoidal_y(2.*pOW / (rho*(qr**2)) - 1.,y)

#----------------------------------------------------------

def Jg_deltaXi(ds, s, ref, OW):
    return int_fpp_eta*ds

#----------------------------------------------------------

def Jg_thetaXi(ds, s, ref, OW):
    return 2.*int_fp_fpp_eta*ds
    
#----------------------------------------------------------

def Jg_thetaZeta(ds, s, ref, OW):
    return 0.

#----------------------------------------------------------

def Jg_vInf(ds, s, ref, param):
    dx = param['dx']  
    qr = ref['q']  
    JDsDx = (Jimm_s(ds) - 4.*Jim_s(ds) + 3.*ds) / (2.*dx)    
    return vc*qr*JDsDx

#----------------------------------------------------------

def Jg_uXiInf(ds, s, ref):
    return 0.

#----------------------------------------------------------

def Jg_tauXi(ds, s, ref, OW):

    nu = ref['nu']    
    qr = ref['q']
        
    DuXiDys = -qr*fpp_ref[0] / (s**2)
        
    return -2.*nu*DuXiDys * ds / (qr**2)

#----------------------------------------------------------

def Jg_pi(ds, s, ref, OW):    
    return 0.

#----------------------------------------------------------

def ip_s(s):
    return ip_dir_2d(s, sqrt(xmax*nu))

#----------------------------------------------------------

def im_s(s):
    return im_dir_2d(s, sqrt(xmin*nu))

#----------------------------------------------------------

def imm_s(s):
    return imm_dir_2d(s, sqrt(xmin*nu))

#----------------------------------------------------------

def kp_s(s):
    return kp_per_2d(s)

#----------------------------------------------------------

def km_s(s):
    return km_per_2d(s)

#----------------------------------------------------------

def ip_d(s, ref, OW, param):
    
    uOW = OW['u']
    wOW = OW['w']
    y = OW['y']
    
    uOWp = ip_u(uOW, param)
    wOWp = ip_w(wOW)
    sp = ip_s(s)
    OWp = {'u': uOWp, 'w': wOWp, 'y':y}
    
    return g_deltaXi(sp, ref, OWp)

#----------------------------------------------------------

def im_d(s, ref, OW):
    
    uOW = OW['u']
    wOW = OW['w']
    y = OW['y']
    
    uOWm = im_u(uOW, ref)
    wOWm = im_w(wOW)
    sm = im_s(s)
    OWm = {'u': uOWm, 'w': wOWm, 'y':y}
    
    return g_deltaXi(sm, ref, OWm)

#----------------------------------------------------------

def imm_d(s, ref, OW):
    
    uOW = OW['u']
    wOW = OW['w']
    y = OW['y']
    
    uOWmm = imm_u(uOW, ref)
    wOWmm = imm_w(wOW)
    smm = imm_s(s)
    OWmm = {'u': uOWmm, 'w': wOWmm, 'y':y}
    
    return g_deltaXi(smm, ref, OWmm)

#----------------------------------------------------------

def kp_d(s, ref, OW):
    
    uOW = OW['u']
    wOW = OW['w']
    y = OW['y']
    
    uOWp = kp_u(uOW)
    wOWp = kp_w(wOW)
    sp = kp_s(s)
    OWp = {'u': uOWp, 'w': wOWp, 'y':y}
    
    return g_deltaXi(sp, ref, OWp)

#----------------------------------------------------------

def km_d(s, ref, OW):
    
    uOW = OW['u']
    wOW = OW['w']
    y = OW['y']
    
    uOWm = km_u(uOW)
    wOWm = km_w(wOW)
    sm = km_s(s)
    OWm = {'u': uOWm, 'w': wOWm, 'y':y}
    
    return g_deltaXi(sm, ref, OWm)

#----------------------------------------------------------

def ip_thXi(s, ref, OW, param):
    
    uOW = OW['u']
    wOW = OW['w']
    y = OW['y']
    
    uOWp = ip_u(uOW, param)
    wOWp = ip_w(wOW)
    sp = ip_s(s)
    OWp = {'u': uOWp, 'w': wOWp, 'y':y}
    
    return g_thetaXi(sp, ref, OWp)

#----------------------------------------------------------

def im_thXi(s, ref, OW):
    
    uOW = OW['u']
    wOW = OW['w']
    y = OW['y']
    
    uOWm = im_u(uOW, ref)
    wOWm = im_w(wOW)
    sm = im_s(s)
    OWm = {'u': uOWm, 'w': wOWm, 'y':y}
    
    return g_thetaXi(sm, ref, OWm)

#----------------------------------------------------------

def imm_thXi(s, ref, OW):
    
    uOW = OW['u']
    wOW = OW['w']
    y = OW['y']
    
    uOWmm = imm_u(uOW, ref)
    wOWmm = imm_w(wOW)
    smm = imm_s(s)
    OWmm = {'u': uOWmm, 'w': wOWmm, 'y':y}
    
    return g_thetaXi(smm, ref, OWmm)

#----------------------------------------------------------

def kp_thXi(s, ref, OW):
    
    uOW = OW['u']
    wOW = OW['w']
    y = OW['y']
    
    uOWp = kp_u(uOW)
    wOWp = kp_w(wOW)
    sp = kp_s(s)
    OWp = {'u': uOWp, 'w': wOWp, 'y':y}
    
    return g_thetaXi(sp, ref, OWp)

#----------------------------------------------------------

def km_thXi(s, ref, OW):
    
    uOW = OW['u']
    wOW = OW['w']
    y = OW['y']
    
    uOWm = km_u(uOW)
    wOWm = km_w(wOW)
    sm = km_s(s)
    OWm = {'u': uOWm, 'w': wOWm, 'y':y}
    
    return g_thetaXi(sm, ref, OWm)

#----------------------------------------------------------

def ip_thZeta(s, ref, OW, param):
    
    uOW = OW['u']
    wOW = OW['w']
    y = OW['y']
    
    uOWp = ip_u(uOW, param)
    wOWp = ip_w(wOW)
    sp = ip_s(s)
    OWp = {'u': uOWp, 'w': wOWp, 'y':y}
    
    return g_thetaZeta(sp, ref, OWp)

#----------------------------------------------------------

def im_thZeta(s, ref, OW):
    
    uOW = OW['u']
    wOW = OW['w']
    y = OW['y']
    
    uOWm = im_u(uOW, ref)
    wOWm = im_w(wOW)
    sm = im_s(s)
    OWm = {'u': uOWm, 'w': wOWm, 'y':y}
    
    return g_thetaZeta(sm, ref, OWm)

#----------------------------------------------------------

def imm_thZeta(s, ref, OW):
    
    uOW = OW['u']
    wOW = OW['w']
    y = OW['y']
    
    uOWmm = imm_u(uOW, ref)
    wOWmm = imm_w(wOW)
    smm = imm_s(s)
    OWmm = {'u': uOWmm, 'w': wOWmm, 'y':y}
    
    return g_thetaZeta(smm, ref, OWmm)

#----------------------------------------------------------

def kp_thZeta(s, ref, OW):
    
    uOW = OW['u']
    wOW = OW['w']
    y = OW['y']
    
    uOWp = kp_u(uOW)
    wOWp = kp_w(wOW)
    sp = kp_s(s)
    OWp = {'u': uOWp, 'w': wOWp, 'y':y}
    
    return g_thetaZeta(sp, ref, OWp)

#----------------------------------------------------------

def km_thZeta(s, ref, OW):
    
    uOW = OW['u']
    wOW = OW['w']
    y = OW['y']
    
    uOWm = km_u(uOW)
    wOWm = km_w(wOW)
    sm = km_s(s)
    OWm = {'u': uOWm, 'w': wOWm, 'y':y}
    
    return g_thetaZeta(sm, ref, OWm)

#----------------------------------------------------------

def ip_pi(s, ref, OW):
    
    pOW = OW['u']
    y = OW['y']
    
    pOWp = ip_p(pOW, ref)
    sp = ip_s(s)
    OWp = {'p': pOWp, 'y':y}
    
    return g_pi(sp, ref, OWp)

#----------------------------------------------------------

def im_pi(s, ref, OW):
    
    pOW = OW['p']
    y = OW['y']
    
    pOWm = im_p(pOW, ref)
    sm = im_s(s)
    OWm = {'p': pOWm, 'y':y}
    
    return g_pi(sm, ref, OWm)

#----------------------------------------------------------

def imm_pi(s, ref, OW):
    
    pOW = OW['p']
    y = OW['y']
    
    pOWmm = imm_p(pOW, ref)
    smm = imm_s(s)
    OWmm = {'p': pOWmm, 'y':y}
    
    return g_pi(smm, ref, OWmm)

#----------------------------------------------------------

def kp_pi(s, ref, OW):
    
    pOW = OW['p']
    y = OW['y']
    
    pOWp = kp_p(pOW)
    sp = kp_s(s)
    OWp = {'p': pOWp, 'y':y}
    
    return g_pi(sp, ref, OWp)

#----------------------------------------------------------

def km_pi(s, ref, OW):
    
    pOW = OW['p']
    y = OW['y']
    
    pOWm = km_u(pOW)
    sm = km_s(s)
    OWm = {'p': pOWm, 'y':y}
    
    return g_pi(sm, ref, OWm)

#----------------------------------------------------------

def ip_u(u, param):
    return ip_neu_3d(u, param, 0.)

#----------------------------------------------------------

def im_u(u, ref):
    qr = ref['q']
    return im_dir_3d(u, qr)

#----------------------------------------------------------

def imm_u(u, ref):
    qr = ref['q']
    return imm_dir_3d(u, qr)

#----------------------------------------------------------

def kp_u(u):
    return kp_per_3d(u)

#----------------------------------------------------------

def km_u(u):
    return km_per_3d(u)

#----------------------------------------------------------

def ip_w(w):
    return ip_dir_3d(w, 0.)

#----------------------------------------------------------

def im_w(w):
    return im_dir_3d(w, 0.)

#----------------------------------------------------------

def imm_w(w):
    return imm_dir_3d(w, 0.)

#----------------------------------------------------------

def kp_w(w):
    return kp_per_3d(w)

#----------------------------------------------------------

def km_w(w):
    return km_per_3d(w)

#----------------------------------------------------------

def ip_p(p, ref):
    qr = ref['q']
    rho = ref['rho']
    pref = 0.5*rho*(qr**2)
    return ip_dir_3d(p, pref)

#----------------------------------------------------------

def im_p(p, ref):
    qr = ref['q']
    rho = ref['rho']
    pref = 0.5*rho*(qr**2)
    return im_dir_3d(p, pref)

#----------------------------------------------------------

def imm_p(p, ref):
    qr = ref['q']
    rho = ref['rho']
    pref = 0.5*rho*(qr**2)
    return imm_dir_3d(p, pref)

#----------------------------------------------------------

def kp_p(p):
    return kp_per_3d(p)

#----------------------------------------------------------

def km_p(p):
    return km_per_3d(p)

#----------------------------------------------------------

def Jip_s(ds):
    return ip_dir_2d(ds, 0.)

#----------------------------------------------------------

def Jim_s(ds):
    return im_dir_2d(ds, 0.)

#----------------------------------------------------------

def Jimm_s(ds):
    return imm_dir_2d(ds, 0.)

#----------------------------------------------------------

def Jkp_s(ds):
    return kp_per_2d(ds)

#----------------------------------------------------------

def Jkm_s(ds):
    return km_per_2d(ds)

#----------------------------------------------------------

def Jip_d(ds, s, ref, OW, param):
    
    uOW = OW['u']
    wOW = OW['w']
    y = OW['y']
    
    uOWp = ip_u(uOW, param)
    wOWp = ip_w(wOW)
    sp = ip_s(s)
    dsp = Jip_s(ds)
    OWp = {'u': uOWp, 'w': wOWp, 'y':y}
    
    return Jg_deltaXi(dsp, sp, ref, OWp)

#----------------------------------------------------------

def Jim_d(ds, s, ref, OW):
    
    uOW = OW['u']
    wOW = OW['w']
    y = OW['y']
    
    uOWm = im_u(uOW, ref)
    wOWm = im_w(wOW)
    sm = im_s(s)
    dsm = Jim_s(ds)
    OWm = {'u': uOWm, 'w': wOWm, 'y':y}
    
    return Jg_deltaXi(dsm, sm, ref, OWm)

#----------------------------------------------------------

def Jimm_d(ds, s, ref, OW):
    
    uOW = OW['u']
    wOW = OW['w']
    y = OW['y']
    
    uOWmm = imm_u(uOW, ref)
    wOWmm = imm_w(wOW)
    smm = imm_s(s)
    dsmm = Jimm_s(ds)
    OWmm = {'u': uOWmm, 'w': wOWmm, 'y':y}
    
    return g_deltaXi(dsmm, smm, ref, OWmm)

#----------------------------------------------------------

def Jkp_d(ds, s, ref, OW):
    
    uOW = OW['u']
    wOW = OW['w']
    y = OW['y']
    
    uOWp = kp_u(uOW)
    wOWp = kp_w(wOW)
    sp = kp_s(s)
    dsp = Jkp_s(ds)
    OWp = {'u': uOWp, 'w': wOWp, 'y':y}
    
    return Jg_deltaXi(dsp, sp, ref, OWp)

#----------------------------------------------------------

def Jkm_d(ds, s, ref, OW):
    
    uOW = OW['u']
    wOW = OW['w']
    y = OW['y']
    
    uOWm = km_u(uOW)
    wOWm = km_w(wOW)
    sm = km_s(s)
    dsm = Jkm_s(ds)
    OWm = {'u': uOWm, 'w': wOWm, 'y':y}
    
    return Jg_deltaXi(dsm, sm, ref, OWm)

#----------------------------------------------------------

def Jip_thXi(ds, s, ref, OW, param):
    
    uOW = OW['u']
    wOW = OW['w']
    y = OW['y']
    
    uOWp = ip_u(uOW, param)
    wOWp = ip_w(wOW)
    sp = ip_s(s)
    dsp = Jip_s(ds)
    OWp = {'u': uOWp, 'w': wOWp, 'y':y}
    
    return Jg_thetaXi(dsp, sp, ref, OWp)

#----------------------------------------------------------

def Jim_thXi(ds, s, ref, OW):
    
    uOW = OW['u']
    wOW = OW['w']
    y = OW['y']
    
    uOWm = im_u(uOW, ref)
    wOWm = im_w(wOW)
    sm = im_s(s)
    dsm = Jim_s(ds)
    OWm = {'u': uOWm, 'w': wOWm, 'y':y}
    
    return Jg_thetaXi(dsm, sm, ref, OWm)

#----------------------------------------------------------

def Jimm_thXi(ds, s, ref, OW):
    
    uOW = OW['u']
    wOW = OW['w']
    y = OW['y']
    
    uOWmm = imm_u(uOW, ref)
    wOWmm = imm_w(wOW)
    smm = imm_s(s)
    dsmm = Jimm_s(ds)
    OWmm = {'u': uOWmm, 'w': wOWmm, 'y':y}
    
    return Jg_thetaXi(dsmm, smm, ref, OWmm)

#----------------------------------------------------------

def Jkp_thXi(ds, s, ref, OW):
    
    uOW = OW['u']
    wOW = OW['w']
    y = OW['y']
    
    uOWp = kp_u(uOW)
    wOWp = kp_w(wOW)
    sp = kp_s(s)
    dsp = Jkp_s(ds)
    OWp = {'u': uOWp, 'w': wOWp, 'y':y}
    
    return Jg_thetaXi(dsp, sp, ref, OWp)

#----------------------------------------------------------

def Jkm_thXi(ds, s, ref, OW):
    
    uOW = OW['u']
    wOW = OW['w']
    y = OW['y']
    
    uOWm = km_u(uOW)
    wOWm = km_w(wOW)
    sm = km_s(s)
    dsm = Jkm_s(ds)
    OWm = {'u': uOWm, 'w': wOWm, 'y':y}
    
    return Jg_thetaXi(dsm, sm, ref, OWm)

#----------------------------------------------------------

def Jip_thZeta(ds, s, ref, OW, param):
    
    uOW = OW['u']
    wOW = OW['w']
    y = OW['y']
    
    uOWp = ip_u(uOW, param)
    wOWp = ip_w(wOW)
    sp = ip_s(s)
    dsp = Jip_s(ds)
    OWp = {'u': uOWp, 'w': wOWp, 'y':y}
    
    return Jg_thetaZeta(dsp, sp, ref, OWp)

#----------------------------------------------------------

def Jim_thZeta(ds, s, ref, OW):
    
    uOW = OW['u']
    wOW = OW['w']
    y = OW['y']
    
    uOWm = im_u(uOW, ref)
    wOWm = im_w(wOW)
    sm = im_s(s)
    dsm = Jim_s(ds)
    OWm = {'u': uOWm, 'w': wOWm, 'y':y}
    
    return Jg_thetaZeta(dsm, sm, ref, OWm)

#----------------------------------------------------------

def Jimm_thZeta(ds, s, ref, OW):
    
    uOW = OW['u']
    wOW = OW['w']
    y = OW['y']
    
    uOWmm = imm_u(uOW, ref)
    wOWmm = imm_w(wOW)
    smm = imm_s(s)
    dsmm = Jimm_s(ds)
    OWmm = {'u': uOWmm, 'w': wOWmm, 'y':y}
    
    return Jg_thetaZeta(dsmm, smm, ref, OWmm)

#----------------------------------------------------------

def Jkp_thZeta(ds, s, ref, OW):
    
    uOW = OW['u']
    wOW = OW['w']
    y = OW['y']
    
    uOWp = kp_u(uOW)
    wOWp = kp_w(wOW)
    sp = kp_s(s)
    dsp = Jkp_s(ds)
    OWp = {'u': uOWp, 'w': wOWp, 'y':y}
    
    return Jg_thetaZeta(dsp, sp, ref, OWp)

#----------------------------------------------------------

def Jkm_thZeta(ds, s, ref, OW):
    
    uOW = OW['u']
    wOW = OW['w']
    y = OW['y']
    
    uOWm = km_u(uOW)
    wOWm = km_w(wOW)
    sm = km_s(s)
    dsm = Jkm_s(ds)
    OWm = {'u': uOWm, 'w': wOWm, 'y':y}
    
    return Jg_thetaZeta(dsm, sm, ref, OWm)

#----------------------------------------------------------

def Jip_pi(ds, s, ref, OW, param):
    
    uOW = OW['u']
    wOW = OW['w']
    y = OW['y']
    
    uOWp = ip_u(uOW, param)
    wOWp = ip_w(wOW)
    sp = ip_s(s)
    dsp = Jip_s(ds)
    OWp = {'u': uOWp, 'w': wOWp, 'y':y}
    
    return Jg_pi(dsp, sp, ref, OWp)

#----------------------------------------------------------

def Jim_pi(ds, s, ref, OW):
    
    uOW = OW['u']
    wOW = OW['w']
    y = OW['y']
    
    uOWm = im_u(uOW, ref)
    wOWm = im_w(wOW)
    sm = im_s(s)
    dsm = Jim_s(ds)
    OWm = {'u': uOWm, 'w': wOWm, 'y':y}
    
    return Jg_pi(dsm, sm, ref, OWm)

#----------------------------------------------------------

def Jimm_pi(ds, s, ref, OW):
    
    uOW = OW['u']
    wOW = OW['w']
    y = OW['y']
    
    uOWmm = imm_u(uOW, ref)
    wOWmm = imm_w(wOW)
    smm = imm_s(s)
    dsmm = Jimm_s(ds)
    OWmm = {'u': uOWmm, 'w': wOWmm, 'y':y}
    
    return Jg_pi(dsmm, smm, ref, OWmm)

#----------------------------------------------------------

def Jkp_pi(ds, s, ref, OW):
    
    uOW = OW['u']
    wOW = OW['w']
    y = OW['y']
    
    uOWp = kp_u(uOW)
    wOWp = kp_w(wOW)
    sp = kp_s(s)
    dsp = Jkp_s(ds)
    OWp = {'u': uOWp, 'w': wOWp, 'y':y}
    
    return Jg_pi(dsp, sp, ref, OWp)

#----------------------------------------------------------

def Jkm_pi(ds, s, ref, OW):
    
    uOW = OW['u']
    wOW = OW['w']
    y = OW['y']
    
    uOWm = km_u(uOW)
    wOWm = km_w(wOW)
    sm = km_s(s)
    dsm = Jkm_s(ds)
    OWm = {'u': uOWm, 'w': wOWm, 'y':y}
    
    return Jg_pi(dsm, sm, ref, OWm)

#----------------------------------------------------------

def extend_y(u, ny):
    return transpose(tile(u,(ny,1,1)),(1,0,2))