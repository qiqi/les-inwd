from numpy import *

def linear_interpolation(x,y,xe):
    
    xe = array(xe)
    xev = ravel(xe)
    xv = ravel(x)
    yv = ravel(y)

    nxe = xev.size
    ye = zeros(nxe)

    for i in range(nxe):
        indx = argmin(abs(xev[i] - xv))
        if xev[i] > xv[indx]:
            xi = (xev[i] - xv[indx])/(xv[indx+1] - xv[indx])
            ye[i] = yv[indx]*(1 + xi) - yv[indx+1]*xi
        else:
            xi = (xev[i] - xv[indx-1])/(xv[indx] - xv[indx-1])
            ye[i] = yv[indx-1]*(1 + xi) - yv[indx]*xi

    return reshape(ye,xe.shape)


def cubic_interpolation(x,y,yp,xe):
    
    xe = array(xe)
    xev = ravel(xe)
    xv = ravel(x)
    yv = ravel(y)
    ypv = ravel(yp)

    nxe = xe.size
    ye = zeros(nxe)

    for i in range(nxe):
        indx = argmin(abs(xev[i] - xv))
        if xev[i] > xv[indx]:
            xi = (xev[i] - xv[indx])/(xv[indx+1] - xv[indx])
            ye[i] = yv[indx]*p0(xi) + ypv[indx]*m0(xi) \
                  + yv[indx+1]*p1(xi) + ypv[indx+1]*m1(xi)
        else:
            xi = (xev[i] - xv[indx-1])/(xv[indx] - xv[indx-1])
            ye[i] = yv[indx-1]*p0(xi) + ypv[indx-1]*m0(xi) \
                  + yv[indx]*p1(xi) + ypv[indx]*m1(xi)


    return reshape(ye,xe.shape)
    

def p0(x):
    return 2.*(x**3) - 3.*(x**2) + 1.
    
    
def p1(x):
    return -2.*(x**3) + 3.*(x**2)
    
    
def m0(x):
    return (x**3) - 2.*(x**2) + x


def m1(x):
    return (x**3) - (x**2)