"""VEM 3D: boundary."""

import numpy as np


def detectar_borde(nodos):
    """
    Nodos sobre el contorno del dominio prismático.
    La tolerancia es relativa al tamaño del dominio para no depender de que
    el generador escriba exactamente el valor de la cara.
    """
    x, y, z = nodos[:, 3], nodos[:, 4], nodos[:, 5]
    Lx = x.max() - x.min()
    Ly = y.max() - y.min()
    Lz = z.max() - z.min()
    tol = 1e-9 * max(Lx, Ly, Lz)
    return (
        (x <= x.min() + tol)
        | (x >= x.max() - tol)
        | (y <= y.min() + tol)
        | (y >= y.max() - tol)
        | (z <= z.min() + tol)
        | (z >= z.max() - tol)
    )


def indices_gdl(nodos):
    """Índices globales de los gdl x, y, z de cada nodo (columnas 0,1,2)."""
    return (nodos[:, 0].astype(int), nodos[:, 1].astype(int), nodos[:, 2].astype(int))


def campo_lineal(nodos, exx, eyy, ezz, exy, exz, eyz):
    """
    Campo de desplazamiento lineal asociado a una deformación uniforme,
    evaluado en TODOS los nodos y devuelto en el orden global de gdl:

        u = exx*x + exy*y + exz*z
        v = exy*x + eyy*y + eyz*z
        w = exz*x + eyz*y + ezz*z
    """
    x, y, z = nodos[:, 3], nodos[:, 4], nodos[:, 5]
    ix, iy, iz = indices_gdl(nodos)

    U = np.zeros(3 * len(nodos))
    U[ix] = exx * x + exy * y + exz * z
    U[iy] = exy * x + eyy * y + eyz * z
    U[iz] = exz * x + eyz * y + ezz * z
    return U
