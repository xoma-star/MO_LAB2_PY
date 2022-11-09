import numpy as np

def target_func(x):
    return np.power(x[0],2) + 2 * np.power(x[1],2)- 10 * np.power(x[2],2)

def H(x):
    return x[0] - np.power(x[1],2) + x[2] - 5

def G(x):
    return x[0] + 5 * x[1] + x[2] - 7

def fi(x,r):# Метод внешних штрафов
    return r/2 * ( np.power(H(x),2) + np.power(G(x),2))

def P(x,r):
    return target_func(x) + fi(x,r)

def grad_target_func(x):
    return np.array([2*x[0], 4*x[1], 20*x[2]])

def grad_H(x):
    return np.array([1, 2*x[1], 1])

def grad_G():
    return np.array([1, 5, 1])

def grad_P(x,r):
    return grad_target_func(x) + ( r*H(x)*grad_H(x) + r*G(x)*grad_G() )

eps_1 = 0.001
eps_2 = 0.01
M = 500 # max iters

C = 4
r = 0.1

while True:
    x = np.array([0,0,0])
    x_prev = np.array([1,1,1])

    t_k = 1
    dots = []

    while (np.linalg.norm(x - x_prev) > eps_2) and (len(dots) < M):

        x_prev = x
        x = x - t_k*grad_P(x,r)
        if P(x,r) - P(x_prev,r) > 0:
            t_k = t_k/2
        
        dots.append(x)

    if abs(fi(x,r)) <= eps_1:
        break
    else:
        r = r * C


print('iter num: ', dots.__len__())
print('Решение: ', x)
print('Значние функции в точке x:', target_func(x))