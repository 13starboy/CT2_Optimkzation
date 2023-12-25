import numpy as np

# Function
def func(x):
  n = len(x)
  res = 0
  for i in range(n - 1):
    res += 4 * ((x[i] ** 2 + x[i + 1] ** 2) ** 2)
  for i in range(n - 1):
    res += (-4 * x[i] + 3)
  return res

# Derivative
def grad(x):
  n = len(x)
  res = [0] * n
  for i in range(n - 1):
    res[i] += 4 * 2 * (x[i] ** 2 + x[i + 1] ** 2) * 2 * x[i]
    res[i + 1] += 4 * 2 * (x[i] ** 2 + x[i + 1] ** 2) * 2 * x[i + 1]

  for i in range(n - 1):
    res[i] += -4

  return np.array(res)


def norm(vec):
    res = 0
    for v in vec:
        res += v * v
    return res ** 0.5


def gold(func, grad, x_k, d, max_alpha=1, rho=0.001, t=2):
    phi_0 = func(x_k)
    dphi_0 = np.dot(grad(x_k), d)
    a = 0
    b = max_alpha
    k = 0
    np.random.seed(42)
    alpha = np.random.rand()*max_alpha
    max_iter = 1000
    while k < max_iter:
        phi = func(x_k + d*alpha)
        if phi_0 + rho*alpha*dphi_0 >= phi:
            if phi_0 + (1-rho)*alpha*dphi_0 <= phi:
                break
            else:
                a = alpha
                if b >= max_alpha:
                    alpha = t*alpha
                    k += 1
                    continue
        else:
            b = alpha
        alpha = 0.5*(a+b)
        k += 1
    return alpha


def fletch_reev(func, grad, x0, epsilon):
    k = 0
    p, x, alpha, beta = [-grad(x0)], [x0], [], ['_']
    while norm(grad(x[k])) > epsilon:
        alpha += [gold(func, grad, x[k], p[k])]
        x += [x[k] + alpha[k] * p[k]]
        beta += [norm(grad(x[k + 1])) ** 2 / norm(grad(x[k])) ** 2]
        p += [-grad(x[k + 1]) + beta[k + 1] * p[k]]
        k += 1
    return (x[k], func(x[k]), k)


x0 = np.array([2] * 10)
point, res, num= fletch_reev(func, grad, x0, epsilon=0.001)

print('Final point: %s' % (point))
print('Num of interation: %s' % (num))
print('Value of the function: %s' % (res))