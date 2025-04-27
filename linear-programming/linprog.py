from scipy.optimize import linprog

# Linprog está optimizado para minimizar una función.
# Si queremos maximizar, debemos cambiar el signo de la función objetivo.
# También, debemos proporcionar las desigualdades en forma de <=, y las igualdades en forma de ==.

# Ejemplo:
# Minimizar: x + y
# Sujeto a: x + y >= 1, x >= 0, y >= 0

# Convertimos el problema de maximización a minimización cambiando el signo de la función objetivo.
# Maximizar: -x -y
# Sujeto a: -x -y <= -1, x >= 0, y >= 0

# Definimos los coeficientes de la función objetivo.
c=[-2,-3]

# Definimos los coeficientes de las desigualdades.
A= [[1,2.8],[1,1]]
b=[20_000,12_000]

# Las variables deben ser >= 0.
x_0_bounds=(0,None)
x_1_bounds=(0,None)

# Ejecutamos el algoritmo.
res = linprog(c, A_ub=A, b_ub=b, bounds=[x_0_bounds, x_1_bounds], method='highs')

# Con el atributo fun obtenemos el valor de Z.
print(f"Z = {res.fun}")

# Con el atributo x obtenemos el valor de las variables.
print(f"x = {res.x}")

# Con el atributo slack obtenemos el valor de los slack variables.
print(f"slack = {res.slack}")

# 