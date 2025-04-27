from scipy import optimize
import numpy as np

# Funcion a optimizar
def f(x):
    return np.cos(2*np.pi*x)*np.exp(-x)+np.sin(3*np.pi*x)*np.exp(-x)

# Implementacion de simulated annealing
res_annealing = optimize.dual_annealing(f, [(0,100)], maxiter=1000)
print(f"Resultado con simulated annealing")
print(f"x = {res_annealing.x}")
print(f"f(x) = {res_annealing.fun}")
print(f"[x_1,x_2] = {res_annealing.x}")

print(f"\n--------------------------------\n")

# Implementacion con algoritmos geneticos
res_genetic = optimize.differential_evolution(f, [(0,100)], maxiter=1000)
print(f"Resultado con algoritmos geneticos")
print(f"Resultado de la optimizacion = {res_genetic.message}")
print(f"Valor optimo de f = {res_genetic.fun}")
print(f"[x_1,x_2] = {res_genetic.x}")