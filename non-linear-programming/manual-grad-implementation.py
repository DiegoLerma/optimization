import numpy as np
from scipy import optimize

#Librería usada para calcular el gradiente.
from jax import grad

# Se define la función a implementar:
# 3(x_1-1)^2 + 2(x_2-3)^2
def f(X):
    x_1, x_2 = X
    return (3*(x_1-1)**2 + 2*(x_2-3)**2)

# Se define la restricción:
# x_1 + x_2 = 20
def constraint_eq(X):
    x_1, x_2 = X
    return x_1 + x_2 - 20

# Se calcula el Lagrangeano:
def L(X):
    x_1, x_2, _lambda = X
    return f((x_1, x_2)) - _lambda * constraint_eq((x_1, x_2))

# Se calcula el gradiente del Lagrangeano:
gradient_L = grad(L)

# Se resuelve buscando las raices (Los puntos donde el gradiente es 0):
res = optimize.root(gradient_L, [0.0,0.0,0.0])

# Se imprimen los resultados:
print(f"Resultado con calculo manual del gradiente")
print(f"------------------------------------------")
print(f"Resultado de la optimización = {res.message}")
print(f"Valor óptimo de f = {f(res.x[:-1])}")
print(f"[x_1,x_2] = {res.x[:-1]}")
print(f"Valor de lambda = {res.x[-1]}")

# 