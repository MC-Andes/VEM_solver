"""VEM k=1 con gradiente medio y estabilización del complemento lineal.

Proyección definida por gradiente medio y promedio nodal, como en (12), (20)
de Beirão da Veiga, Lovadina y Mora: https://arxiv.org/abs/1503.02042.
"""

import numpy as np

from .geometry import polygon_geometry


def element_stiffness(points, C):
    area, gradient, _ = polygon_geometry(points)
    n = len(points)
    B = np.zeros((3, 2 * n))
    B[0, :n], B[1, n:] = gradient[0], gradient[1]
    B[2, :n], B[2, n:] = gradient[1], gradient[0]
    centered = points - points.mean(axis=0)
    h = np.ptp(points, axis=0).max()
    D = np.column_stack((np.ones(n), centered / h))
    P = np.vstack((np.full(n, 1 / n), h * gradient))
    projection = D @ P
    # (I-DP)^T(I-DP), evaluado con rango 3: O(n²), no O(n³).
    S = P.T @ (D.T @ D) @ P - projection - projection.T
    S.flat[:: n + 1] += 1
    K = area * (B.T @ C @ B)
    alpha = np.trace(K) / n
    K[:n, :n] += alpha * S
    K[n:, n:] += alpha * S
    return (K + K.T) / 2, B, area
