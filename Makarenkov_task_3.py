import numpy as np
import numpy.linalg as ln
import scipy as sp
import scipy.optimize


# Function
def f(x):
    n = len(x)
    res = 0
    for i in range(n - 1):
        res += (x[i]**2 - 2)**2
    res1 = 0
    for i in range(n):
        res1 += x[i]**2
    res1 -= 0.5
    res += res1 ** 2
    return res


# Derivative
def f1(x):
    n = len(x)
    res = [0] * n
    for i in range(n - 1):
        res[i] += 2*(x[i]**2 - 2)*2*x[i]
    res1 = 0
    for i in range(n):
        res1 += x[i] ** 2
    res1 -= 0.5
    for i in range(n):
        res[i] += 2*res1*2*x[i]
    return np.array(res)


def bfgs(f, fprime, x0, eps=0.001):

    k = 0
    gfk = fprime(x0)
    N = len(x0)
    Identity = np.eye(N, dtype=int)
    Hk = Identity
    xk = x0

    while ln.norm(gfk) > eps:
        pk = -np.dot(Hk, gfk)
        wolf_search = sp.optimize.line_search(f, f1, xk, pk)
        alpha = wolf_search[0]

        xkp = xk + alpha * pk
        sk = xkp - xk
        xk = xkp

        gfkp1 = fprime(xkp)
        yk = gfkp1 - gfk
        gfk = gfkp1

        k += 1

        ro = 1.0 / (np.dot(yk, sk))
        A1 = Identity - ro * sk[:, np.newaxis] * yk[np.newaxis, :]
        A2 = Identity - ro * yk[:, np.newaxis] * sk[np.newaxis, :]
        Hk = np.dot(A1, np.dot(Hk, A2)) + (ro * sk[:, np.newaxis] *
                                           sk[np.newaxis, :])
    return (xk, k)


res, num = bfgs(f, f1, np.array([1]*10))

print('Final point: %s' % (res))
print('Num of interation: %s' % (num))
print('Value of the function: %s' % (f(res)))