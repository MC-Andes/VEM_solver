"""VEM 3D: results."""

import csv
import os

import numpy as np

from .boundary import indices_gdl
from .constants import CASOS_CARGA, COD_MATRIZ, COMP_EPS, COMP_SIGMA, E_HOM, NU_HOM
from .homogenization import estadisticas_esfuerzo


def guardar_resultados_malla(carpeta, nodos, C, C_fuer, resultados, meta, guardar_campos=True):
    """
    Escribe los .txt de una réplica:
      C_equivalente.txt, C_equivalente_fuerzas.txt, resumen.txt,
      U_<caso>.txt, deformaciones_<caso>.txt, esfuerzos_<caso>.txt
    """
    os.makedirs(carpeta, exist_ok=True)
    ix, iy, iz = indices_gdl(nodos)

    if guardar_campos:
        for nombre, r in resultados.items():
            U = r["U"]
            np.savetxt(
                os.path.join(carpeta, f"U_{nombre}.txt"),
                np.column_stack([U[ix], U[iy], U[iz]]),
                fmt="%.12e",
                header="ux  uy  uz",
                comments="# ",
            )
            np.savetxt(
                os.path.join(carpeta, f"deformaciones_{nombre}.txt"),
                np.column_stack([np.arange(len(r["eps"])), meta["volumenes"], r["eps"]]),
                fmt=["%.0f"] + ["%.12e"] * 7,
                header="elem  V  " + "  ".join(COMP_EPS),
                comments="# ",
            )
            np.savetxt(
                os.path.join(carpeta, f"esfuerzos_{nombre}.txt"),
                np.column_stack([np.arange(len(r["sig"])), meta["volumenes"], r["sig"]]),
                fmt=["%.0f"] + ["%.12e"] * 7,
                header="elem  V  " + "  ".join(COMP_SIGMA),
                comments="# ",
            )

    etiq_c = [c[0] for c in CASOS_CARGA]

    # ── C_equivalente.txt (promedio volumétrico) ─────────────────────
    with open(os.path.join(carpeta, "C_equivalente.txt"), "w", encoding="utf-8") as f:
        f.write("# Matriz constitutiva homogenizada C (Pa) - VEM 3D\n")
        f.write("# Obtenida por promedio de esfuerzos ponderado por volumen\n")
        f.write("# Voigt: sigma = [sxx, syy, szz, txy, txz, tyz]\n")
        f.write("#        eps   = [exx, eyy, ezz, gxy, gxz, gyz]\n")
        f.write("# Columnas: " + ", ".join(etiq_c) + "\n")
        for i in range(6):
            f.write("  ".join(f"{C[i, j]:+.12e}" for j in range(6)) + "\n")

    # ── C_equivalente_fuerzas.txt (reacciones de borde) ──────────────
    with open(os.path.join(carpeta, "C_equivalente_fuerzas.txt"), "w", encoding="utf-8") as f:
        f.write("# Matriz constitutiva homogenizada C (Pa) - VEM 3D\n")
        f.write("# Obtenida por reacciones en el contorno: sigma_ij = (1/V) sum F_i x_j\n")
        f.write(f"# Volumen usado (caja): {meta['vol_bbox']:.12e}\n")
        f.write("# Columnas: " + ", ".join(etiq_c) + "\n")
        for i in range(6):
            f.write("  ".join(f"{C_fuer[i, j]:+.12e}" for j in range(6)) + "\n")

    # ── resumen.txt ──────────────────────────────────────────────────
    L = []
    L.append("=" * 100)
    L.append(f"RESUMEN VEM 3D - {meta['malla']}")
    L.append("=" * 100)
    L.append("")
    L.append(f"Modo                 : {meta['modo']}")
    L.append(f"Nodos                : {meta['n_nodos']:,}")
    L.append(f"Aristas              : {meta['n_aristas']:,}")
    L.append(f"Caras                : {meta['n_caras']:,}")
    L.append(f"Elementos            : {meta['n_elementos']:,}")
    L.append(f"Grados de libertad   : {meta['n_dofs']:,}")
    L.append(f"Nodos en borde       : {meta['n_borde']:,}")
    L.append(f"  gdl prescritos     : {3 * meta['n_borde']:,}")
    L.append(f"  gdl libres         : {meta['n_gdl_libres']:,}")
    L.append("")
    L.append(f"Solver               : {meta['solver']}")
    if meta["solver"] == "amg":
        L.append(f"  iteraciones (max)  : {meta['solver_iters_max']}")
        L.append(f"  residuo rel. (max) : {meta['solver_res_rel']:.3e}")
    L.append("")
    L.append(f"Volumen (suma elem.) : {meta['vol_total']:.6e}")
    L.append(f"Volumen (caja)       : {meta['vol_bbox']:.6e}")
    L.append(f"  discrepancia rel.  : {meta['dif_volumen']:.3e}")
    L.append(f"Dominio Lx, Ly, Lz   : {meta['Lx']:.6e}, {meta['Ly']:.6e}, {meta['Lz']:.6e}")
    L.append(
        f"Fraccion agregado    : {100 * meta['frac_agg']:.2f} %   (codigo matriz = {COD_MATRIZ})"
    )
    L.append("")
    if meta.get("memoria_pico_gb"):
        L.append(f"Memoria pico         : {meta['memoria_pico_gb']:.2f} GB")
        L.append("")
    L.append(f"Tiempo total         : {meta['tiempo_s']:.1f} s")
    L.append(f"  lectura            : {meta['t_lectura']:.1f} s")
    L.append(f"  infocaras          : {meta['t_infocaras']:.1f} s")
    L.append(f"  ensamblaje         : {meta['t_ensamblaje']:.1f} s")
    L.append(f"  solver (6 casos)   : {meta['t_solver']:.1f} s")
    L.append("")
    L.append("-" * 100)
    L.append("MATRIZ C HOMOGENIZADA (Pa) - promedio de esfuerzos por volumen")
    L.append("-" * 100)
    L.append(f"{'':8s}" + "".join(f"{e:>18s}" for e in etiq_c))
    for i in range(6):
        L.append(f"{COMP_SIGMA[i]:<8s}" + "".join(f"{C[i, j]:+18.6e}" for j in range(6)))
    L.append("")
    L.append("-" * 100)
    L.append("MATRIZ C HOMOGENIZADA (Pa) - reacciones en el contorno")
    L.append("-" * 100)
    L.append(f"{'':8s}" + "".join(f"{e:>18s}" for e in etiq_c))
    for i in range(6):
        L.append(f"{COMP_SIGMA[i]:<8s}" + "".join(f"{C_fuer[i, j]:+18.6e}" for j in range(6)))
    L.append("")
    L.append(f"  Discrepancia relativa maxima entre ambas vias: {meta['dif_vias']:.3e}")
    L.append("")
    L.append("-" * 100)
    L.append("CONSTANTES EFECTIVAS")
    L.append("-" * 100)
    L.append("  Ajuste por minimos cuadrados sobre las 36 entradas de C:")
    L.append(f"    E_eq  = {meta['E_eq']:.6e} Pa")
    L.append(f"    nu_eq = {meta['nu_eq']:.6f}")
    L.append("  Extraccion directa C11/C12/C44 (valida solo si C es isotropica):")
    L.append(f"    E_dir  = {meta['E_dir']:.6e} Pa")
    L.append(f"    nu_dir = {meta['nu_dir']:.6f}")
    L.append(f"  Desviacion de C respecto a la mejor C isotropica: {meta['anisotropia']:.3e}")
    L.append("")
    L.append("-" * 100)
    if meta["modo"] == "homogeneo":
        L.append("PATCH TEST (campo lineal impuesto)")
        L.append("-" * 100)
        L.append("  El medio es homogeneo, asi que el campo lineal del contorno")
        L.append(
            "  es la solucion exacta; el error numerico depende del solver y del condicionamiento."
        )
    else:
        L.append("DESVIACION RESPECTO AL CAMPO LINEAL DEL CONTORNO")
        L.append("-" * 100)
        L.append("  NO es un patch test: en un medio heterogeneo el campo interno")
        L.append("  no es lineal, de modo que un valor no nulo es lo esperado.")
    L.append("")
    L.append("  Maximo relativo del desplazamiento, por caso de carga:")
    for nombre, r in resultados.items():
        L.append(f"    {nombre:<8s} {r['err_rel']:.3e}")
    L.append("")

    if meta["modo"] == "homogeneo":
        L.append("-" * 100)
        L.append("VERIFICACION CONTRA LA C TEORICA (material uniforme)")
        L.append("-" * 100)
        Ct = meta["C_teorica"]
        L.append(f"  E  = {E_HOM:.6e} Pa      nu = {NU_HOM}")
        L.append("")
        L.append(f"  {'Entrada':<10s}{'Calculado':>20s}{'Teorico':>20s}{'Err rel':>14s}")
        for et, (i, j) in [
            ("C11", (0, 0)),
            ("C22", (1, 1)),
            ("C33", (2, 2)),
            ("C12", (0, 1)),
            ("C13", (0, 2)),
            ("C23", (1, 2)),
            ("C44", (3, 3)),
            ("C55", (4, 4)),
            ("C66", (5, 5)),
            ("C14", (0, 3)),
            ("C25", (1, 4)),
            ("C36", (2, 5)),
        ]:
            calc, teo = C[i, j], Ct[i, j]
            err = abs(calc - teo) / abs(teo) if abs(teo) > 0 else abs(calc)
            L.append(f"  {et:<10s}{calc:>20.6e}{teo:>20.6e}{err:>14.3e}")
        L.append("")
        L.append(f"  Error relativo maximo : {meta['err_C_max']:.3e}")
        L.append(f"  Tolerancia            : {meta['tol_C']:.1e}")
        estado = "COINCIDE" if meta["verif_ok"] else "NO COINCIDE"
        L.append(f"  Resultado             : {estado}")
        L.append("")

    L.append("-" * 100)
    L.append("ESTADISTICAS DE ESFUERZOS POR CASO")
    L.append("-" * 100)
    for nombre, r in resultados.items():
        L.append(f"\n  Caso: {nombre}")
        L.append(f"  {'Comp':<6s}{'Min':>16s}{'Max':>16s}{'Media':>16s}{'Desv.Std':>16s}")
        L.append("  " + "-" * 70)
        reduced = r.get("stats")
        if reduced is None:
            reduced = estadisticas_esfuerzo(r["sig"])
        for comp in COMP_SIGMA:
            v = reduced[comp]
            L.append(
                f"  {comp:<6s}{v['min']:>16.4e}{v['max']:>16.4e}{v['mean']:>16.4e}{v['std']:>16.4e}"
            )
    L.append("")
    L.append("=" * 100)

    with open(os.path.join(carpeta, "resumen.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))


def escribir_resumen_global(ruta, filas, modo, tiempo_total, tol_C=1e-6):
    """RESUMEN_GLOBAL.txt con una línea por malla."""
    L = []
    L.append("=" * 140)
    L.append(f"RESUMEN GLOBAL - CASO {modo.upper()} - VEM 3D")
    L.append("=" * 140)
    L.append("")

    if modo == "homogeneo":
        L.append(
            f"{'Malla':<24s}{'Estado':<10s}{'Tiempo':>10s}"
            f"{'Nodos':>10s}{'Caras':>10s}{'Elem':>10s}"
            f"{'Err C':>12s}{'Err patch':>12s}"
            f"{'E eq':>15s}{'nu eq':>10s}{'Verif':>13s}"
        )
        L.append("-" * 140)
        for f in filas:
            if not f["ok"]:
                L.append(f"{f['malla']:<24s}{'ERROR':<10s}{f['tiempo_s']:>9.1f}s")
                continue
            m = f["meta"]
            L.append(
                f"{f['malla']:<24s}{'OK':<10s}{m['tiempo_s']:>9.1f}s"
                f"{m['n_nodos']:>10,d}{m['n_caras']:>10,d}"
                f"{m['n_elementos']:>10,d}"
                f"{m['err_C_max']:>12.2e}{f['err_patch']:>12.2e}"
                f"{m['E_eq']:>15.4e}{m['nu_eq']:>10.4f}"
                f"{('COINCIDE' if m['verif_ok'] else 'NO COINCIDE'):>13s}"
            )
    else:
        L.append(
            f"{'Malla':<24s}{'Estado':<10s}{'Tiempo':>10s}"
            f"{'Nodos':>10s}{'Caras':>10s}{'Elem':>10s}"
            f"{'Agg(%)':>10s}{'E eq':>15s}{'nu eq':>10s}"
            f"{'Dif vias':>12s}{'Anisotr':>12s}"
        )
        L.append("-" * 140)
        for f in filas:
            if not f["ok"]:
                L.append(f"{f['malla']:<24s}{'ERROR':<10s}{f['tiempo_s']:>9.1f}s")
                continue
            m = f["meta"]
            L.append(
                f"{f['malla']:<24s}{'OK':<10s}{m['tiempo_s']:>9.1f}s"
                f"{m['n_nodos']:>10,d}{m['n_caras']:>10,d}"
                f"{m['n_elementos']:>10,d}"
                f"{100 * m['frac_agg']:>10.2f}{m['E_eq']:>15.4e}"
                f"{m['nu_eq']:>10.4f}{m['dif_vias']:>12.2e}"
                f"{m['anisotropia']:>12.2e}"
            )

    ok = [f for f in filas if f["ok"]]
    fail = [f for f in filas if not f["ok"]]
    L.append("")
    L.append("-" * 140)
    L.append(f"Mallas procesadas : {len(filas)}    OK: {len(ok)}    ERROR: {len(fail)}")

    if ok:
        L.append(f"Peor err_patch    : {max(f['err_patch'] for f in ok):.3e}")
        L.append(f"Peor dif_vias     : {max(f['meta']['dif_vias'] for f in ok):.3e}")
        L.append(f"Peor dif_volumen  : {max(f['meta']['dif_volumen'] for f in ok):.3e}")

    if modo == "homogeneo" and ok:
        n_bien = sum(1 for f in ok if f["meta"]["verif_ok"])
        peor = max(f["meta"]["err_C_max"] for f in ok)
        L.append(f"Verificacion C    : {n_bien}/{len(ok)} coinciden (tolerancia {tol_C:.1e})")
        L.append(f"Peor error en C   : {peor:.3e}")

    if fail:
        L.append("")
        L.append("Mallas con error:")
        for f in fail:
            L.append(f"  {f['malla']}: {f.get('error', 'desconocido')}")

    L.append("")
    L.append(f"TIEMPO TOTAL: {tiempo_total:.1f}s ({tiempo_total / 60:.1f} min)")
    L.append("=" * 140)

    with open(ruta, "w", encoding="utf-8") as f:
        f.write("\n".join(L))


def escribir_csv_global(ruta, filas, modo):
    """RESUMEN_PROPIEDADES.csv con una fila por malla."""
    campos = [
        "malla",
        "caso",
        "replica",
        "estado",
        "pct_real(%)",
        "n_nodos",
        "n_aristas",
        "n_caras",
        "n_elementos",
        "n_dofs",
        "n_borde",
        "n_gdl_libres",
        "solver",
        "solver_iters_max",
        "solver_res_rel",
        "memoria_pico_gb",
        "vol_total",
        "vol_bbox",
        "dif_volumen",
        "Lx",
        "Ly",
        "Lz",
    ]
    campos += [f"C{i + 1}{j + 1}" for i in range(6) for j in range(6)]
    campos += [f"F{i + 1}{j + 1}" for i in range(6) for j in range(6)]
    campos += [
        "E_eq",
        "nu_eq",
        "E_dir",
        "nu_dir",
        "anisotropia",
        "err_patch",
        "dif_vias",
        "tiempo_s",
        "t_lectura",
        "t_infocaras",
        "t_ensamblaje",
        "t_solver",
    ]
    if modo == "homogeneo":
        campos += ["C11_teo", "C12_teo", "C44_teo", "err_C_max", "verif_ok"]

    for nombre, *_ in CASOS_CARGA:
        for comp in COMP_SIGMA:
            for st in ["min", "max", "mean", "std"]:
                campos.append(f"{nombre}_{comp}_{st}")

    with open(ruta, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=campos)
        w.writeheader()
        for f in filas:
            caso, _, rep = f["malla"].partition("/")
            fila = {c: "" for c in campos}
            fila["malla"] = f["malla"]
            fila["caso"] = caso
            fila["replica"] = rep
            fila["estado"] = "OK" if f["ok"] else "ERROR"
            fila["tiempo_s"] = round(f.get("tiempo_s", 0.0), 2)

            if f["ok"]:
                m, C, Cf = f["meta"], f["C"], f["C_fuerzas"]
                fila["pct_real(%)"] = round(100 * m["frac_agg"], 4)
                for k in [
                    "n_nodos",
                    "n_aristas",
                    "n_caras",
                    "n_elementos",
                    "n_dofs",
                    "n_borde",
                    "n_gdl_libres",
                    "solver",
                    "solver_iters_max",
                ]:
                    fila[k] = m[k]
                fila["solver_res_rel"] = f"{m['solver_res_rel']:.3e}"
                fila["memoria_pico_gb"] = round(m.get("memoria_pico_gb", 0.0), 3)
                for k in ["vol_total", "vol_bbox", "Lx", "Ly", "Lz"]:
                    fila[k] = f"{m[k]:.10e}"
                fila["dif_volumen"] = f"{m['dif_volumen']:.4e}"
                for i in range(6):
                    for j in range(6):
                        fila[f"C{i + 1}{j + 1}"] = f"{C[i, j]:.10e}"
                        fila[f"F{i + 1}{j + 1}"] = f"{Cf[i, j]:.10e}"
                fila["E_eq"] = f"{m['E_eq']:.10e}"
                fila["nu_eq"] = f"{m['nu_eq']:.8f}"
                fila["E_dir"] = f"{m['E_dir']:.10e}"
                fila["nu_dir"] = f"{m['nu_dir']:.8f}"
                fila["anisotropia"] = f"{m['anisotropia']:.4e}"
                fila["err_patch"] = f"{f['err_patch']:.4e}"
                fila["dif_vias"] = f"{m['dif_vias']:.4e}"
                for k in ["t_lectura", "t_infocaras", "t_ensamblaje", "t_solver"]:
                    fila[k] = round(m[k], 2)
                if modo == "homogeneo":
                    Ct = m["C_teorica"]
                    fila["C11_teo"] = f"{Ct[0, 0]:.10e}"
                    fila["C12_teo"] = f"{Ct[0, 1]:.10e}"
                    fila["C44_teo"] = f"{Ct[3, 3]:.10e}"
                    fila["err_C_max"] = f"{m['err_C_max']:.4e}"
                    fila["verif_ok"] = int(m["verif_ok"])
                for k, v in f["stats"].items():
                    fila[k] = f"{v:.10e}"
            w.writerow(fila)
