import numpy as np
from jax import grad

# Se define la función objetivo:
def f(x_1, x_2):
    return (3*(x_1-1)**2 + 2*(x_2-3)**2)

# Se calcula el gradiente para ambos argumentos:
gradL = grad(f, argnums=[0,1])

# Se define cual sera el metodo de actualización de x:
def minGD(x) : return x - 0.1 * np.array(gradL(x[0], x[1]))

# Se genera el primer valor al azar:
x = np.random.random(2)

# Se itera 150 veces:
for epoch in range(150):
    x = minGD(x)

print(f"Resultado con gradiente descendente")
print(f"--------------------------------")
print(f"Valor óptimo de f = {f(x[0], x[1])}")
print(f"[x_1,x_2] = {x}")

# 