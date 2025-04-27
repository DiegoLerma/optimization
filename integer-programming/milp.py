import numpy as np
from scipy.optimize import milp
from scipy.optimize import LinearConstraint

# MILP está implementado para minimizar, por lo que habrá
# que transformar max f(x) en -min(-f(x).

# ¡Importante: MILP solo está disponible a partir de la versión 1.9 de SciPy.

# Se introducen los coeficientes de f(x) = -17x_1 + 12x_2.
# La introducción de los elementos se puede hacer tanto con listas
# de Python como de arrays de NumPy, se recomienda usar NumPy, ya que
# ofrece mejor rendimiento por estar optimizado para análisis numérico.
c = np.array([-17, 12])

# Se introducen los coeficientes de las restricciones:
# 10x_1 + 7x_2 < 40
# 1x_1 + 1x_2 < 5
A = np.array([[10, 7], [1, 1]])
b_u = np.array([40, 5])
b_l = np.array([-np.inf, -np.inf])
constraints = LinearConstraint(A, b_l, b_u)

# Todas las variables van a ser enteras:
integrality = np.array([1, 1])

# Se realiza la optimización:
res = milp(c=c, constraints=constraints, integrality=integrality)

# Con el atributo fun se tiene el valor z:
print(f"z = {res.fun}")

# Con el atributo x, los valores de las variables decisoras:
print(f"[x_1, x_2] = {res.x}")
