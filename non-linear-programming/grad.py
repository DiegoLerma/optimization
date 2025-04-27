import numpy as np
from scipy import optimize
# Librería usada para calcular el gradiente.
from jax import grad

# Se define la función a implementar:
# 3(x_1-1)^2 + 2(x_2-3)^2
def f(X):
    x_1, x_2 = X
    return (3*(x_1-1)**2 + 2*(x_2-3)**2)

# Se calcula el gradiente:
gradient_f = grad(f)

# Se optimiza mediante una variedad del método Newton: BFGS.
res_newton = optimize.minimize(f, method='BFGS', x0=[1,1], jac=gradient_f)
print(f"Resultados con 'BFGS'")
print(f"--------------------")
print(f"Resultado de la optimización = {res_newton.message}")
print(f"Valor óptimo de f = {res_newton.fun}")
print(f"[x_1,x_2] = {res_newton.x}")

# Se optimiza mediante una variedad del método Newton: Nelder-Mead.
res_neld = optimize.minimize(f, method='Nelder-Mead', x0=[1,1], jac=gradient_f)
print(f"Resultados con 'Nelder-Mead'")
print(f"-----------------------------")
print(f"Resultado de la optimización = {res_neld.message}")
print(f"Valor óptimo de f = {res_neld.fun}")
print(f"[x_1,x_2] = {res_neld.x}")
