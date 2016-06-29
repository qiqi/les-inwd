from numpy import *
from utilities_blasius import *
from scipy.sparse.linalg import LinearOperator, gmres
from numpy.linalg import *

gmresTol = 1e-10
gmresMaxIter = 500
newtonTol = 1e-10
newtonMaxIter = 100

def newton(F, gradF, x0):

    x = x0
    b = -ravel(F(x))
    for i in range(newtonMaxIter):
        dx = zeros(x0.shape)
        b0 = -ravel(gradF(dx, x))
        def linear_op(dx):
            dx = dx.reshape(x0.shape)
            gradFdx = gradF(dx, x)
            return ravel(gradFdx) + b0

        A = LinearOperator((x.size, x.size), linear_op, dtype='float64')
        dx, _ = gmres(A, b+b0, x0=ravel(dx.copy()), tol=gmresTol, maxiter=gmresMaxIter)
        x += dx.reshape(x0.shape)
        b = -ravel(F(x))
        if norm(b) < newtonTol:
            break    

    return x
