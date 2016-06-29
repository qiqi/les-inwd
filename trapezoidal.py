from numpy import *

def trapezoidal_y(f, x):
    fm = (f[:,:-1,:] + f[:,1:,:]) / 2.