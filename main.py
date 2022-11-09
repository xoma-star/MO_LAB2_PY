import numpy as np
import sympy



def gradientDescent(functions, x0=(2, 2, 2), M=10000, eps1=0.001, alfa=0.01, C=1.03, r=0.01):
    x1, x2, x3 = x0

    dY_divide_dX1, dY_divide_dX2, dY_divide_dX3, _, _, phi, p, y = functions()
    dY_divide_dX1, dY_divide_dX2, dY_divide_dX3, phi, p, y = (sympy.lambdify([sympy.Symbol('x1'), sympy.Symbol('x2'), sympy.Symbol('x3'), sympy.Symbol('r')], i) for i in (dY_divide_dX1, dY_divide_dX2, dY_divide_dX3, phi, p, y))

    numberOfIteration = 0
    rNext = r
    while(numberOfIteration < M):
        f = y(x1, x2, x3, r)
        phi1 = phi(x1, x2, x3, r)
        p1 = p(x1, x2, x3, r)

        x1Next = x1 - alfa * dY_divide_dX1(x1, x2, x3, rNext)
        x2Next = x2 - alfa * dY_divide_dX2(x1, x2, x3, rNext)
        x3Next = x3 - alfa * dY_divide_dX3(x1, x2, x3, rNext)

        fNext = y(x1Next, x2Next, x3Next, r)
        phi2 = phi(x1Next, x2Next, x3Next, rNext)
        p2 = p(x1Next, x2Next, x3Next, rNext)

        if(phi1 < eps1):
            print(f'phi1 = {phi1} p2 = {p2}')
            break
        else:
            x1 = x1Next
            x2 = x2Next
            x3 = x3Next
            rNext = C * rNext
            # alfa *= 0.95
            numberOfIteration += 1

    return x1Next, x2Next, x3Next, fNext, y(*x0, r)

def derivativesFunction():
    x1 = sympy.Symbol('x1')
    x2 = sympy.Symbol('x2')
    x3 = sympy.Symbol('x3')

    r = sympy.Symbol('r')

    # defenitions
    y = x1 ** 2 + 2 * x2 ** 2 + 10 * (x3 ** 2)
    cond1 = x1 - x2 ** 2 + x3 - 5
    cond2 = x1 + 5 * x2 + x3 - 7

    phi = r / 2 * (cond1 ** 2 + cond2 ** 2)
    p = y + phi

    # print(dY_divide_dX1)
    # print(dY_divide_dX2)
    # print(dY_divide_dX3)

    return p.diff(x1), p.diff(x2), p.diff(x3), cond1, cond2, phi, p, y

def task1Variant10(functions=derivativesFunction):

    return gradientDescent(functions, x0=(0, 0, 0))


x1min, x2min, x3min, fmin, fold = task1Variant10()

print(f'x1min = {x1min}')
print(f'x2min = {x2min}')
print(f'x3min = {x3min}')
print(f'fmin = {fmin}')
print(f'fold = {fold}')