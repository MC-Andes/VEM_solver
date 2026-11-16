"""Elasticidad isotrópica en tensión plana y deformación plana."""

import numpy as np


def constitutive(E, nu, hypothesis="tension-plana"):
    if not np.isfinite(E) or E <= 0 or not np.isfinite(nu) or not -1 < nu < 0.5:
        raise ValueError("Cada material requiere E > 0 y -1 < nu < 0.5, finitos")
    mu = E / (2 * (1 + nu))
    if hypothesis == "tension-plana":
        lam = E * nu / (1 - nu**2)
    elif hypothesis == "deformacion-plana":
        lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    else:
        raise ValueError(f"Hipótesis desconocida: {hypothesis}")
    return np.array([[lam + 2 * mu, lam, 0], [lam, lam + 2 * mu, 0], [0, 0, mu]])


def isotropic_fit(C, hypothesis):
    # Ajuste Frobenius en Voigt ingenieril: C = lambda*A + mu*B.
    a = np.array([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    b = np.diag([2.0, 2.0, 1.0])
    lam, mu = np.linalg.lstsq(np.column_stack([a.ravel(), b.ravel()]), C.ravel(), rcond=None)[0]
    if mu <= 0 or lam + mu <= 0:
        raise ValueError("La matriz equivalente no admite un ajuste isotrópico estable")
    if hypothesis == "tension-plana":
        nu = lam / (lam + 2 * mu)
    elif hypothesis == "deformacion-plana":
        nu = lam / (2 * (lam + mu))
    else:
        raise ValueError(f"Hipótesis desconocida: {hypothesis}")
    E = 2 * mu * (1 + nu)
    fit = lam * a + mu * b
    return float(E), float(nu), float(np.linalg.norm(C - fit) / np.linalg.norm(C))
