"""VEM 3D: workflow."""

import os
import time

import numpy as np

from .assembly import ensamblar_particionado, liberar_sistema
from .boundary import detectar_borde
from .constants import E_HOM, NU_HOM
from .geometry import construir_infocaras
from .homogenization import _estadisticas, fraccion_agregado, homogenizar
from .materials import C_isotropica_3D, ajustar_E_nu, anisotropia, extraer_E_nu_directo
from .mesh import (
    _buscar_archivo,
    asignar_propiedades,
    asignar_propiedades_uniformes,
    leer_aristas,
    leer_caras,
    leer_elementos,
    leer_nodos,
    leer_propiedades,
)
from .resources import memoria_pico_gb
from .results import guardar_resultados_malla
from .solvers import factorizar_particionado


def correr_malla(
    carpeta_vem,
    carpeta_salida,
    nombre_malla,
    modo,
    tol_C=1e-6,
    guardar_campos=True,
    solver="auto",
    base_tmp=None,
    inicio_afin=True,
):
    """
    Corre la homogenización de una réplica.

    modo: "homogeneo"   -> E=1e6, nu=0.3 en todos los elementos y se
                           verifica C contra la teórica.
          "heterogeneo" -> se leen las propiedades del Propiedades.txt de
                           la propia malla.

    Devuelve un dict con el resultado (ok / error y los datos del resumen).
    """
    t0 = time.perf_counter()
    part = sistema = None
    try:
        if modo not in ("homogeneo", "heterogeneo"):
            raise ValueError(f"Modo desconocido: {modo}")
        # ── lectura ──────────────────────────────────────────────────
        t = time.perf_counter()
        f_nod = _buscar_archivo(carpeta_vem, "Input_nodos.txt", "input_nodos.txt")
        f_ari = _buscar_archivo(carpeta_vem, "Input_aristas.txt", "input_aristas.txt")
        f_car = _buscar_archivo(carpeta_vem, "Input_caras.txt", "input_caras.txt")
        f_ele = _buscar_archivo(
            carpeta_vem,
            "input_elements.txt",
            "Input_elements.txt",
            "input_elementos.txt",
            "Input_elementos.txt",
        )

        nodos = leer_nodos(f_nod)
        aristas = leer_aristas(f_ari)
        caras = leer_caras(f_car)
        elementos, codigos = leer_elementos(f_ele)

        if modo == "homogeneo":
            asignar_propiedades_uniformes(elementos, E_HOM, NU_HOM)
        else:
            f_pro = _buscar_archivo(
                carpeta_vem, "Propiedades.txt", "propiedades.txt", "propiedades2.txt"
            )
            asignar_propiedades(elementos, leer_propiedades(f_pro), codigos)
        t_lectura = time.perf_counter() - t
        print(f"[{nombre_malla}] lectura: {t_lectura:.2f} s", flush=True)

        # ── geometría de caras ───────────────────────────────────────
        t = time.perf_counter()
        infocaras = construir_infocaras(nodos, aristas, caras)
        t_infocaras = time.perf_counter() - t
        print(f"[{nombre_malla}] geometría: {t_infocaras:.2f} s", flush=True)

        # ── ensamblaje ───────────────────────────────────────────────
        # Se impone el contorno DURANTE el ensamblaje: nunca se construye la
        # K completa, solo los bloques que el solver necesita.
        borde_mask = detectar_borde(nodos)
        t = time.perf_counter()
        info = {}
        part = ensamblar_particionado(
            nodos,
            aristas,
            caras,
            elementos,
            infocaras,
            info,
            borde_mask,
            base_tmp=base_tmp,
            compact_info=True,
        )
        t_ensamblaje = time.perf_counter() - t
        print(f"[{nombre_malla}] ensamblaje: {t_ensamblaje:.2f} s", flush=True)
        del infocaras

        # ── homogenización ───────────────────────────────────────────
        x, y, z = nodos[:, 3], nodos[:, 4], nodos[:, 5]
        Lx = x.max() - x.min()
        Ly = y.max() - y.min()
        Lz = z.max() - z.min()
        V_bbox = Lx * Ly * Lz

        t = time.perf_counter()
        sistema = factorizar_particionado(part, solver=solver)
        part = None
        C, C_fuer, resultados, info_solver = homogenizar(
            nodos,
            elementos,
            info,
            sistema,
            borde_mask,
            V_bbox,
            guardar_campos=guardar_campos,
            inicio_afin=inicio_afin,
        )
        liberar_sistema(sistema)
        sistema = None
        t_solver = time.perf_counter() - t
        print(f"[{nombre_malla}] solver y homogenización: {t_solver:.2f} s", flush=True)

        E_eq, nu_eq = ajustar_E_nu(C)
        E_dir, nu_dir = extraer_E_nu_directo(C)
        frac_agg, V_tot = fraccion_agregado(info, codigos)

        escala_C = np.abs(C).max()
        dif_vias = float(np.abs(C - C_fuer).max() / escala_C) if escala_C > 0 else 0.0
        dif_vol = (abs(V_tot - V_bbox) / V_bbox) if V_bbox > 0 else 0.0

        meta = {
            "malla": nombre_malla,
            "modo": modo,
            "n_nodos": len(nodos),
            "n_aristas": len(aristas),
            "n_caras": len(caras),
            "n_elementos": len(elementos),
            "n_dofs": 3 * len(nodos),
            "n_borde": int(borde_mask.sum()),
            "vol_total": V_tot,
            "vol_bbox": V_bbox,
            "dif_volumen": dif_vol,
            "Lx": Lx,
            "Ly": Ly,
            "Lz": Lz,
            "frac_agg": frac_agg,
            "E_eq": E_eq,
            "nu_eq": nu_eq,
            "E_dir": E_dir,
            "nu_dir": nu_dir,
            "anisotropia": anisotropia(C),
            "dif_vias": dif_vias,
            "tol_C": tol_C,
            "t_lectura": t_lectura,
            "t_infocaras": t_infocaras,
            "t_ensamblaje": t_ensamblaje,
            "t_solver": t_solver,
            "solver": info_solver["nombre"],
            "n_gdl_libres": info_solver["n_gdl_libres"],
            "solver_iters_max": info_solver["iters_max"],
            "solver_res_rel": info_solver["res_rel_max"],
            "volumenes": np.array([info[i]["V"] for i in range(len(info))]),
        }

        if modo == "homogeneo":
            Ct = C_isotropica_3D(E_HOM, NU_HOM)
            escala = np.abs(Ct).max()
            err_max = float(np.abs(C - Ct).max() / escala)
            meta["C_teorica"] = Ct
            meta["err_C_max"] = err_max
            meta["verif_ok"] = err_max <= tol_C

        meta["memoria_pico_gb"] = memoria_pico_gb()
        meta["tiempo_s"] = time.perf_counter() - t0
        guardar_resultados_malla(carpeta_salida, nodos, C, C_fuer, resultados, meta, guardar_campos)

        meta.pop("volumenes")  # no viaja entre procesos
        return {
            "ok": True,
            "malla": nombre_malla,
            "C": C,
            "C_fuerzas": C_fuer,
            "meta": meta,
            "err_patch": max(r["err_rel"] for r in resultados.values()),
            "stats": _estadisticas(resultados),
            "tiempo_s": meta["tiempo_s"],
        }

    except Exception as ex:
        import traceback

        tb = traceback.format_exc()
        try:
            os.makedirs(carpeta_salida, exist_ok=True)
            with open(os.path.join(carpeta_salida, "ERROR.txt"), "w", encoding="utf-8") as f:
                f.write(tb)
        except Exception:
            pass
        return {
            "ok": False,
            "malla": nombre_malla,
            "error": str(ex),
            "traceback": tb,
            "tiempo_s": time.perf_counter() - t0,
        }

    finally:
        liberar_sistema(sistema)
        liberar_sistema(part)
