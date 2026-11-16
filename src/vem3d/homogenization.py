"""VEM 3D: homogenization."""

import numpy as np

from .boundary import campo_lineal, indices_gdl
from .constants import CASOS_CARGA, COD_MATRIZ, COMP_SIGMA
from .solvers import resolver


def post_proceso(U, elementos, info):
    """Deformación y esfuerzo constantes por elemento (proyección VEM)."""
    ne = len(elementos)
    eps = np.zeros((ne, 6))
    sig = np.zeros((ne, 6))
    for i in range(ne):
        ue = U[info[i]["gdl"]]
        operator = info[i].get("strain")
        if operator is None:
            operator = info[i]["dpi"] @ info[i]["piast"]
        e = operator @ ue
        eps[i] = e
        sig[i] = info[i]["C"] @ e
    return eps, sig


def fraccion_agregado(info, codigos):
    """
    Fracción de volumen ocupada por agregado.
    El código COD_MATRIZ (= 0) es la matriz; cualquier otro es agregado.
    """
    V_tot = sum(info[i]["V"] for i in range(len(info)))
    V_agg = sum(info[i]["V"] for i in range(len(info)) if int(codigos[i]) != COD_MATRIZ)
    return (V_agg / V_tot if V_tot else 0.0), V_tot


def esfuerzo_por_fuerzas(sistema, U, nodos, borde_mask, V):
    """
    Esfuerzo medio a partir de las reacciones en el contorno:

        <sigma_ij> = (1/V) * sum_(nodos borde) F_i * x_j

    Es la otra lectura del teorema del esfuerzo medio. Debe coincidir con el
    promedio volumétrico; se calculan las dos para poder contrastarlas.
    Las componentes cortantes se simetrizan.
    """
    # Solo hacen falta las reacciones en los gdl prescritos (los del
    # contorno), que son los unicos que intervienen en las sumas de abajo;
    # por eso basta con las filas prescritas de K y no la K completa.
    F = np.zeros(sistema["ndof"])  # no hay fuerzas de cuerpo
    F[sistema["idx_f"]] = sistema["K_f"] @ U
    ix, iy, iz = indices_gdl(nodos)
    Fx, Fy, Fz = F[ix], F[iy], F[iz]
    x, y, z = nodos[:, 3], nodos[:, 4], nodos[:, 5]

    b = borde_mask
    sxx = np.dot(Fx[b], x[b]) / V
    syy = np.dot(Fy[b], y[b]) / V
    szz = np.dot(Fz[b], z[b]) / V
    sxy = 0.5 * (np.dot(Fx[b], y[b]) + np.dot(Fy[b], x[b])) / V
    sxz = 0.5 * (np.dot(Fx[b], z[b]) + np.dot(Fz[b], x[b])) / V
    syz = 0.5 * (np.dot(Fy[b], z[b]) + np.dot(Fz[b], y[b])) / V
    return np.array([sxx, syy, szz, sxy, sxz, syz])


def homogenizar(
    nodos, elementos, info, sistema, borde_mask, V_bbox, guardar_campos=True, inicio_afin=True
):
    """
    Resuelve los seis casos de carga y arma la C 6x6 por dos vías
    independientes: promedio volumétrico de esfuerzos y reacciones de borde.
    Recibe el `sistema` ya preparado por `factorizar`, no la K completa:
    asi el llamador puede liberar K antes de resolver y no se mantienen K y
    K_ll vivas a la vez.

    Devuelve (C_vol, C_fuerzas, resultados_por_caso, info_solver), donde
    info_solver recoge el solver usado y, con AMG, las iteraciones y el
    residuo relativo de cada caso, para dejarlos registrados en el resumen.
    """
    C = np.zeros((6, 6))
    C_fuer = np.zeros((6, 6))
    resultados = {}

    vols = np.array([info[i]["V"] for i in range(len(info))])
    V_tot = vols.sum()

    idx_f = sistema["idx_f"]

    for col, (nombre, exx, eyy, ezz, exy, exz, eyz) in enumerate(CASOS_CARGA):
        u_teo = campo_lineal(nodos, exx, eyy, ezz, exy, exz, eyz)
        # El campo macroscópico es una aproximación física disponible en ambos modos.
        # Se comprueba el residuo real después de resolver; no se relaja la tolerancia.
        initial = u_teo[sistema["idx_l"]] if inicio_afin else None
        U = resolver(sistema, u_teo[idx_f], x0=initial)
        eps, sig = post_proceso(U, elementos, info)

        # Vía 1: promedio de esfuerzos ponderado por volumen
        sigma_prom = (sig * vols[:, None]).sum(axis=0) / V_tot
        C[:, col] = sigma_prom

        # Vía 2: reacciones en el contorno
        sigma_fuer = esfuerzo_por_fuerzas(sistema, U, nodos, borde_mask, V_bbox)
        C_fuer[:, col] = sigma_fuer

        # Desviación respecto al campo lineal impuesto
        denom = np.abs(u_teo).max()
        err_rel = np.abs(U - u_teo).max() / denom if denom > 0 else 0.0

        resultados[nombre] = {
            "stats": estadisticas_esfuerzo(sig),
            "sigma_prom": sigma_prom,
            "sigma_fuerzas": sigma_fuer,
            "err_rel": err_rel,
        }
        if guardar_campos:
            resultados[nombre].update(U=U, eps=eps, sig=sig)

    info_solver = {
        "nombre": sistema["solver"],
        "n_gdl_libres": int(len(sistema["idx_l"])),
        "iters_max": (max(sistema["iters"]) if sistema["iters"] else 0),
        "res_rel_max": (max(sistema["res_rel"]) if sistema["res_rel"] else 0.0),
    }
    return C, C_fuer, resultados, info_solver


def estadisticas_esfuerzo(sig):
    """Estadísticas reducidas por componente; no retiene los campos completos."""
    return {
        comp: {
            "min": float(sig[:, k].min()),
            "max": float(sig[:, k].max()),
            "mean": float(sig[:, k].mean()),
            "std": float(sig[:, k].std()),
        }
        for k, comp in enumerate(COMP_SIGMA)
    }


def _estadisticas(resultados):
    """min/max/media/desviación de cada componente de esfuerzo, por caso."""
    stats = {}
    for nombre, r in resultados.items():
        reduced = r.get("stats")
        if reduced is None:
            reduced = estadisticas_esfuerzo(r["sig"])
        for comp, values in reduced.items():
            for statistic, value in values.items():
                stats[f"{nombre}_{comp}_{statistic}"] = value
    return stats
