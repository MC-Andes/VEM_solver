"""VEM 3D: solvers."""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

from .boundary import indices_gdl

try:
    import pyamg

    HAY_PYAMG = True
except ImportError:
    HAY_PYAMG = False
UMBRAL_GDL_LU = 40_000
TOL_AMG = 1e-12
AMG_SMOOTH = "jacobi"
AMG_STRENGTH = "symmetric"
AMG_MAX_COARSE = 500


def _validar_indices_amg(K):
    """PyAMG 5.3 no compila los kernels necesarios para índices int64."""
    if (K.indptr.dtype == np.int64 or K.indices.dtype == np.int64) and (
        np.lib.NumpyVersion(pyamg.__version__) < "5.4.1"
    ):
        raise RuntimeError(
            "Esta matriz requiere índices CSR int64. Instala pyamg>=5.4.1; "
            "PyAMG 5.3.0 sólo admite int32 en los kernels usados. "
            "No convertir los índices a int32: se perderían no ceros."
        )


def factorizar_particionado(part, solver="auto"):
    """
    Prepara el operador del solver sobre un sistema ya particionado por
    `ensamblar_particionado`. Es la mitad de `factorizar` que no depende de
    como se haya construido K_ll, de modo que ambos caminos comparten solver
    y los tiempos siguen siendo comparables.
    """
    K_ll = part["K_ll"]
    n_l = K_ll.shape[0]

    if solver == "auto":
        solver = "lu" if n_l < UMBRAL_GDL_LU else "amg"
    if solver == "amg" and not HAY_PYAMG:
        raise RuntimeError("solver='amg' requiere pyamg (pip install pyamg).")
    if solver not in ("lu", "amg"):
        raise ValueError(f"solver desconocido: {solver!r}")

    sistema = {
        "solver": solver,
        "K_lf": part["K_lf"],
        "K_f": part["K_f"],
        "idx_l": part["idx_l"],
        "idx_f": part["idx_f"],
        "ndof": part["ndof"],
        "iters": [],
        "res_rel": [],
        "carpeta_tmp": part.get("carpeta_tmp"),
    }

    if n_l == 0:
        sistema["op"] = None
        return sistema

    if solver == "lu":
        sistema["op"] = splu(sp.csr_matrix(K_ll).tocsc())
    else:
        _validar_indices_amg(K_ll)
        sistema["op"] = pyamg.smoothed_aggregation_solver(
            K_ll, max_coarse=AMG_MAX_COARSE, smooth=AMG_SMOOTH, strength=AMG_STRENGTH
        )
        sistema["K_ll"] = K_ll
    return sistema


def factorizar(K, nodos, borde_mask, solver="auto"):
    """
    Particiona los gdl en libres (l) y prescritos (f) y prepara UNA sola vez
    el operador con el que se resuelven los seis casos de carga. Los seis
    fijan EXACTAMENTE los mismos gdl (los del contorno) y solo cambian los
    valores prescritos, de modo que K_ll es común a los seis: con LU se
    reutiliza la factorización y con AMG se reutiliza la jerarquía.

    solver:
      "lu"   -> LU disperso (SuperLU). Exacto, pero el relleno crece muy
                rápido en 3D: una malla de 116 304 gdl libres necesitó más
                de 6 GB y fue abortada por falta de memoria.
      "amg"  -> multigrid algebraico por agregación (pyamg) con CG.
                Para esa misma malla: 2.8 GB de pico, 42-43 iteraciones por
                caso y residuo relativo ~1e-12.
      "auto" -> LU si hay pocos gdl libres (< UMBRAL_GDL_LU) y AMG si no,
                requiere pyamg para sistemas grandes.
    """
    ndof = 3 * len(nodos)
    ix, iy, iz = indices_gdl(nodos)

    fijo = np.zeros(ndof, dtype=bool)
    fijo[ix[borde_mask]] = True
    fijo[iy[borde_mask]] = True
    fijo[iz[borde_mask]] = True

    idx_l = np.where(~fijo)[0]
    idx_f = np.where(fijo)[0]

    if solver == "auto":
        if len(idx_l) < UMBRAL_GDL_LU:
            solver = "lu"
        else:
            solver = "amg"
    if solver == "amg" and not HAY_PYAMG:
        raise RuntimeError("solver='amg' requiere pyamg (pip install pyamg).")
    if solver not in ("lu", "amg"):
        raise ValueError(f"solver desconocido: {solver!r}")

    # Se guardan tambien las FILAS prescritas, que son las unicas que hacen
    # falta para las reacciones del contorno. Asi el llamador puede liberar
    # la K completa en cuanto esta funcion retorna, en vez de mantenerla
    # viva junto con K_ll durante los seis casos de carga.
    K_lf = K[idx_l][:, idx_f].tocsr()
    sistema = {
        "solver": solver,
        "K_lf": K_lf,
        "K_f": K[idx_f].tocsr(),
        "idx_l": idx_l,
        "idx_f": idx_f,
        "ndof": ndof,
        "iters": [],
        "res_rel": [],
    }

    if len(idx_l) == 0:
        sistema["op"] = None
        return sistema

    if solver == "lu":
        sistema["op"] = splu(K[idx_l][:, idx_l].tocsc())
    else:
        K_ll = K[idx_l][:, idx_l].tocsr()
        _validar_indices_amg(K_ll)
        sistema["op"] = pyamg.smoothed_aggregation_solver(
            K_ll, max_coarse=AMG_MAX_COARSE, smooth=AMG_SMOOTH, strength=AMG_STRENGTH
        )
        sistema["K_ll"] = K_ll
    return sistema


def resolver(sistema, u_f, x0=None):
    """Resuelve K_ll u_l = -K_lf u_f reutilizando el operador preparado."""
    U = np.zeros(sistema["ndof"])
    U[sistema["idx_f"]] = u_f
    idx_l = sistema["idx_l"]
    if len(idx_l) == 0:
        return U

    b = -(sistema["K_lf"] @ u_f)
    if not np.any(b):
        # El sistema libre es SPD: con RHS cero la solución es cero, aunque x0 no lo sea.
        if sistema["solver"] == "amg":
            sistema["iters"].append(0)
            sistema["res_rel"].append(0.0)
        return U

    if sistema["solver"] == "lu":
        U[idx_l] = sistema["op"].solve(b)
        return U

    residuos = []
    u_l = sistema["op"].solve(b, x0=x0, tol=TOL_AMG, accel="cg", residuals=residuos, maxiter=500)
    nb = np.linalg.norm(b)
    rel = np.linalg.norm(b - sistema["K_ll"] @ u_l) / nb if nb > 0 else 0.0
    if not np.isfinite(rel) or not np.all(np.isfinite(u_l)) or rel > TOL_AMG:
        raise RuntimeError(f"AMG no convergió: residuo relativo {rel:.3e} > {TOL_AMG:.3e}")
    sistema["iters"].append(max(len(residuos) - 1, 0))
    sistema["res_rel"].append(float(rel))
    U[idx_l] = u_l
    return U
