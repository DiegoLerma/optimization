# Proyecto de Optimización con Python

Este repositorio contiene ejemplos de técnicas de optimización en Python, organizados en tres categorías:
- Programación lineal
- Programación no lineal
- Programación entera

## Requisitos

- Python 3.7 o superior
- NumPy
- SciPy (>= 1.7)
- JAX

## Instalación de dependencias

Se recomienda usar [mamba](https://mamba.readthedocs.io/en/latest/), un gestor de paquetes rápido compatible con conda.

### Instalación con mamba

1. Si no tienes mamba, instálalo con:

   ```bash
   conda install -c conda-forge mamba
   ```

2. Crea un entorno y activa:

   ```bash
   mamba create -n optimizacion python=3.9 numpy scipy jaxlib jax -c conda-forge
   conda activate optimizacion
   ```

   > **Nota:** Si usas Apple Silicon o GPU, revisa la [documentación de JAX](https://github.com/google/jax#installation) para instrucciones específicas.

### Alternativa: instalación con pip

```bash
pip install numpy scipy jax
```

## Estructura de directorios

- `linear-programming/`: ejemplos de programación lineal usando `scipy.optimize.linprog`.
- `non-linear-programming/`: ejemplos de optimización no lineal usando SciPy y JAX.
- `integer-programming/`: ejemplos de problemas de programación entera (MILP) con SciPy.

## 1. Programación lineal

### Archivo

`linear-programming/linprog.py`

### Descripción

Resuelve problemas de optimización lineal (minimización o maximización) usando `scipy.optimize.linprog`.

### Ejecución

```bash
python linear-programming/linprog.py
```

## 2. Programación no lineal

### Archivos y métodos

- `grad.py`: optimización con métodos BFGS y Nelder-Mead (SciPy + JAX).
- `grad-desc.py`: descenso de gradiente explícito usando `jax.grad`.
- `manual-grad-implementation.py`: método de Lagrange y resolución de raíces con `scipy.optimize.root`.
- `slsqp.py`: optimización con SLSQP y restricciones de desigualdad.

### Ejecución

```bash
python non-linear-programming/grad.py
python non-linear-programming/grad-desc.py
python non-linear-programming/manual-grad-implementation.py
python non-linear-programming/slsqp.py
```

## 3. Programación entera

### Archivo

`integer-programming/milp.py`

### Descripción

Resuelve problemas de programación entera mixta (MILP) usando `scipy.optimize.milp` y `LinearConstraint`.

### Ejecución

```bash
python integer-programming/milp.py
```

## Licencia

Este proyecto está disponible bajo la licencia MIT. 