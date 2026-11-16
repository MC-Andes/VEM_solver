"""VEM 3D: materials."""

import numpy as np
from scipy.optimize import least_squares


def C_isotropica_3D(E, nu):
    """Tensor constitutivo isotrópico 3D en notación de Voigt (ingenieril)."""
    f = E / ((1 + nu) * (1 - 2 * nu))
    G = E / (2 * (1 + nu))
    C = np.zeros((6, 6))
    C[0, 0] = C[1, 1] = C[2, 2] = f * (1 - nu)
    C[0, 1] = C[0, 2] = C[1, 2] = f * nu
    C[1, 0] = C[2, 0] = C[2, 1] = f * nu
    C[3, 3] = C[4, 4] = C[5, 5] = G
    return C


def extraer_E_nu_directo(C):
    """
    Extracción directa que usaba el script 3D anterior:
        nu = C12 / (C11 + C12),   E = 2*C44*(1 + nu)
    con C11, C12 y C44 promediados sobre las entradas equivalentes.
    Solo es exacta si C es isotrópica; se reporta para comparar.
    """
    C11 = np.mean([C[0, 0], C[1, 1], C[2, 2]])
    C12 = np.mean([C[0, 1], C[0, 2], C[1, 2], C[1, 0], C[2, 0], C[2, 1]])
    C44 = np.mean([C[3, 3], C[4, 4], C[5, 5]])
    nu = C12 / (C11 + C12) if (C11 + C12) != 0 else 0.0
    E = 2 * C44 * (1 + nu)
    return float(E), float(nu)


def ajustar_E_nu(C):
    """
    Ajusta por mínimos cuadrados el par (E, nu) cuya C isotrópica mejor
    aproxima la C calculada, usando las 36 entradas de la matriz.
    """

    def residuo(p):
        return (C_isotropica_3D(p[0], p[1]) - C).ravel()

    E0, nu0 = extraer_E_nu_directo(C)
    if not np.isfinite(E0) or E0 <= 0:
        E0 = max(abs(C[0, 0]), 1.0)
    nu0 = float(np.clip(nu0, -0.3, 0.45))
    sol = least_squares(residuo, [E0, nu0], bounds=([1e-6, -0.95], [np.inf, 0.49]))
    return float(sol.x[0]), float(sol.x[1])


def anisotropia(C):
    """
    Medida escalar de la desviación respecto a la mejor C isotrópica:
        ||C - C_iso(E_eq, nu_eq)||_max / max|C|
    No es un índice estándar; sirve para detectar mallas cuya C efectiva
    no admite una lectura isotrópica razonable.
    """
    E, nu = ajustar_E_nu(C)
    escala = np.abs(C).max()
    if escala == 0:
        return 0.0
    return float(np.abs(C - C_isotropica_3D(E, nu)).max() / escala)
