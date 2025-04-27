import numpy as np
from scipy import optimize
from jax import grad

# Se define la función a implementar:
# 3(x_1-1)^2 + 2(x_2-3)^2
def f(X):
    x_1, x_2 = X
    return (3*(x_1-1)**2 + 2*(x_2-3)**2)

# Se define la restricción:
# x_1 + x_2 < 5
def constraint(X):
    x_1, x_2 = X
    return x_1 + x_2 - 2

# Se calcula el gradiente de la función objetivo:
gradient_f = grad(f)

# Se optimiza con SLSQP:
res = optimize.minimize(f,
                        method='SLSQP',
                        x0=[1,12],
                        jac=gradient_f,
                        constraints=[{'type': 'ineq', 'fun': constraint}])

print(f"Resultado con SLSQP")
print(f"--------------------")
print(f"Resultado de la optimización = {res.message}")
print(f"Valor óptimo de f = {res.fun}")
print(f"[x_1,x_2] = {res.x}")
