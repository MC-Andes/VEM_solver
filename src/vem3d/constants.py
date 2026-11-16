"""Convenciones de materiales y cargas: Voigt ingenieril."""

import numpy as np

# Código de material de la matriz. Todo código distinto se cuenta como agregado.
COD_MATRIZ = 0

# Propiedades del caso homogéneo de verificación
E_HOM = 1.0e6
NU_HOM = 0.3

R_mat = np.array([[0, 1], [-1, 0]])

# (nombre, exx, eyy, ezz, exy, exz, eyz)  -- componentes TENSORIALES
CASOS_CARGA = [
    ("exx=1", 1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    ("eyy=1", 0.0, 1.0, 0.0, 0.0, 0.0, 0.0),
    ("ezz=1", 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
    ("gxy=1", 0.0, 0.0, 0.0, 0.5, 0.0, 0.0),
    ("gxz=1", 0.0, 0.0, 0.0, 0.0, 0.5, 0.0),
    ("gyz=1", 0.0, 0.0, 0.0, 0.0, 0.0, 0.5),
]

COMP_SIGMA = ["sxx", "syy", "szz", "txy", "txz", "tyz"]
COMP_EPS = ["exx", "eyy", "ezz", "gxy", "gxz", "gyz"]
