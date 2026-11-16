"""Geometría de polígonos simples; cálculo trasladado y escalado."""

import numpy as np


def polygon_geometry(points):
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3 or not np.isfinite(points).all():
        raise ValueError("Un polígono necesita al menos tres vértices 2D finitos")
    scale = np.ptp(points, axis=0).max()
    if scale <= 0:
        raise ValueError("Polígono degenerado")
    p = (points - points[0]) / scale
    edge = np.roll(p, -1, axis=0) - p
    tol = 1e-13
    if np.any(np.linalg.norm(edge, axis=1) <= tol):
        raise ValueError("Polígono con arista de longitud nula")

    def cross(a, b):
        return a[0] * b[1] - a[1] * b[0]

    def on_segment(a, b, q):
        return (
            abs(cross(b - a, q - a)) <= tol
            and np.all(q >= np.minimum(a, b) - tol)
            and np.all(q <= np.maximum(a, b) + tol)
        )

    # Los polígonos pueden ser cóncavos, pero no autointersectarse.
    for i in range(len(p)):
        a, b = p[i], p[(i + 1) % len(p)]
        for j in range(i + 2, len(p)):
            if i == 0 and j == len(p) - 1:
                continue
            c, d = p[j], p[(j + 1) % len(p)]
            signs = (
                cross(b - a, c - a),
                cross(b - a, d - a),
                cross(d - c, a - c),
                cross(d - c, b - c),
            )
            proper = signs[0] * signs[1] < -(tol**2) and signs[2] * signs[3] < -(tol**2)
            if (
                proper
                or on_segment(a, b, c)
                or on_segment(a, b, d)
                or on_segment(c, d, a)
                or on_segment(c, d, b)
            ):
                raise ValueError("Polígono con aristas que se cruzan o se solapan")
    twice_area = np.sum(p[:, 0] * np.roll(p[:, 1], -1) - p[:, 1] * np.roll(p[:, 0], -1))
    if abs(twice_area) <= tol:
        raise ValueError("Polígono con área nula o numéricamente degenerada")
    area = abs(twice_area) * scale**2 / 2
    # Integral del gradiente de cada función nodal, mediante las dos aristas vecinas.
    normals = np.column_stack((edge[:, 1], -edge[:, 0])) * np.sign(twice_area)
    gradient = ((normals + np.roll(normals, 1, axis=0)) / (abs(twice_area) * scale)).T
    return float(area), gradient, bool(twice_area > 0)
