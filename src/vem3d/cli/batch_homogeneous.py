# -*- coding: utf-8 -*-
"""Comprueba varias mallas con E=1e6 y nu=0.3 en todos los elementos.

Uso:
  vem3d-batch-homogeneous --raiz mis_mallas --listar
  vem3d-batch-homogeneous --raiz mis_mallas --procesos 1 --sin-campos

La salida se guarda en <raiz>/Resultados_Homogeneos/. Cada proceso carga
una malla completa; aumenta --procesos sólo si hay memoria suficiente.
"""

import argparse
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

from .. import compat as core

# ══════════════════════════════════════════════════════════════════════════════
#  VALORES POR DEFECTO  -  configurables con --procesos y --raiz
# ══════════════════════════════════════════════════════════════════════════════

PROCESOS = 1  # cuantas mallas a la vez. Un numero fijo (1, 2, 4, 8...)
# o "auto" para que mida la malla mas grande y lo decida.

RAIZ = "."  # carpeta 3D con los casos. Ejemplo en Windows:
# RAIZ = r"C:\Users\tu_usuario\Downloads\mallas\3D"

# ══════════════════════════════════════════════════════════════════════════════

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


CARPETA_SALIDA = "Resultados_Homogeneos"


def _tarea(args):
    """Ejecuta una malla. Se llama en el proceso hijo."""
    (nombre_malla, carpeta_vem, carpeta_salida, tol_C, guardar_campos, solver, base_tmp) = args
    try:
        return core.correr_malla(
            carpeta_vem,
            carpeta_salida,
            nombre_malla,
            modo="homogeneo",
            tol_C=tol_C,
            guardar_campos=guardar_campos,
            solver=solver,
            base_tmp=base_tmp,
        )
    except Exception as ex:
        return {
            "ok": False,
            "malla": nombre_malla,
            "error": str(ex),
            "traceback": traceback.format_exc(),
            "tiempo_s": 0.0,
        }


def _log(k, total, r):
    if r["ok"]:
        m = r["meta"]
        estado = "COINCIDE" if m["verif_ok"] else "NO COINCIDE"
        print(
            f"  [{k:3d}/{total}] OK  {r['malla']:<22s}"
            f" nod={m['n_nodos']:>7,d} car={m['n_caras']:>7,d}"
            f" ele={m['n_elementos']:>6,d}"
            f"  errC={m['err_C_max']:.2e}"
            f"  patch={r['err_patch']:.2e}"
            f"  {estado:<11s} ({m['tiempo_s']:.1f}s)",
            flush=True,
        )
    else:
        print(f"  [{k:3d}/{total}] ERR {r['malla']:<22s} {r['error']}", flush=True)


def main():
    ap = argparse.ArgumentParser(description="Batch VEM 3D - caso homogeneo (verificacion)")
    ap.add_argument("--raiz", default=RAIZ, help="Carpeta 3D que contiene los casos (default: .)")
    ap.add_argument(
        "--procesos",
        default=PROCESOS,
        help="Numero de procesos (default: 1), o 'auto' "
        "para que se mida la malla mas grande y se "
        "calcule cuantos caben en la RAM libre",
    )
    ap.add_argument(
        "--tmp",
        default=None,
        help="Carpeta para los archivos temporales del "
        "ensamblaje (tripletas y arreglos de K_ll). Por "
        "defecto, el temporal del sistema. Usala para "
        "mandarlos a un disco con espacio: necesita unos "
        "16 bytes por entrada elemental",
    )
    ap.add_argument(
        "--min-gdl", type=int, default=0, help="Correr solo mallas con AL MENOS estos gdl (3*nodos)"
    )
    ap.add_argument(
        "--max-gdl",
        type=int,
        default=0,
        help="Correr solo mallas con COMO MUCHO estos gdl "
        "(0 = sin tope). Sirve para partir la corrida por "
        "tamano y usar mas procesos en las mallas chicas",
    )
    ap.add_argument(
        "--solo", default=None, help="Correr solo los casos cuyo nombre contenga este texto"
    )
    ap.add_argument(
        "--tol",
        type=float,
        default=1e-6,
        help="Tolerancia relativa para la verificacion de C (default: 1e-6)",
    )
    ap.add_argument(
        "--sin-campos",
        action="store_true",
        help="No escribir U_*, esfuerzos_* ni deformaciones_* (ahorra mucho disco)",
    )
    ap.add_argument(
        "--solver",
        choices=["amg", "lu", "auto"],
        default="amg",
        help="amg: multigrid + CG (default). Para las mallas 3D "
        "de este proyecto es ~5x mas rapido que el LU y usa "
        "~1/3 de la memoria. lu: LU disperso exacto, 1 "
        "factorizacion + 6 sustituciones; conviene solo en "
        "mallas chicas. auto: LU por debajo de "
        "core.UMBRAL_GDL_LU gdl libres y AMG por encima",
    )
    ap.add_argument(
        "--listar", action="store_true", help="Solo listar las mallas encontradas y salir"
    )
    args = ap.parse_args()

    if args.solver == "amg" and not core.HAY_PYAMG:
        print("\n  El solver 'amg' requiere pyamg, que no esta instalado.")
        print("      pip install pyamg")
        print("  O corre con --solver lu (exacto, pero mucha mas memoria:")
        print("  ~6.6 GB extrapolados para una malla de 116 000 gdl libres).")
        return 1

    auto = str(args.procesos).strip().lower() == "auto"
    if not auto:
        try:
            args.procesos = int(args.procesos)
        except ValueError:
            print(f"\n  --procesos debe ser un entero o 'auto', no {args.procesos!r}")
            return 1
        if args.procesos < 1:
            print("\n  --procesos debe ser >= 1")
            return 1

    raiz = os.path.abspath(args.raiz)
    salida_raiz = os.path.join(raiz, CARPETA_SALIDA)

    mallas = core.descubrir_mallas(raiz)
    if args.solo:
        mallas = [m for m in mallas if args.solo in m[0]]

    if args.min_gdl or args.max_gdl:
        antes = len(mallas)
        filtradas = []
        for m in mallas:
            gdl = 3 * core.tamano_malla(m[3])[0]
            if gdl < args.min_gdl:
                continue
            if args.max_gdl and gdl > args.max_gdl:
                continue
            filtradas.append(m)
        mallas = filtradas
        tope = f"{args.max_gdl:,d}" if args.max_gdl else "sin tope"
        print(
            f"\n  Filtro de tamano: gdl en [{args.min_gdl:,d}, {tope}]"
            f"  ->  {len(mallas)} de {antes} mallas"
        )

    if not mallas:
        print(f"\n  No se encontro ninguna malla bajo {raiz}.")
        print("  Se espera <RAIZ>/<CASO>/<CASO_R##>/VEM/Input_nodos.txt")
        return 1

    if args.listar:
        print(f"\n  {'Malla':<24s}{'Nodos':>10s}{'Elems':>10s}{'GDL':>12s}   Carpeta")
        print("  " + "-" * 110)
        filas_tam = []
        for caso, rep, nombre, dirvem in mallas:
            nn, ne = core.tamano_malla(dirvem)
            filas_tam.append((nn, nombre))
            gdl = 3 * nn if nn > 0 else -1
            print(f"  {nombre:<24s}{nn:>10,d}{ne:>10,d}{gdl:>12,d}   {dirvem}")
        print("  " + "-" * 110)
        print(f"  Total: {len(mallas)} mallas")
        validas = [n for n, _ in filas_tam if n > 0]
        if validas:
            mayor = max(filas_tam)
            print(f"  La mas grande es {mayor[1]} con {3 * mayor[0]:,d} gdl.")
            print(
                "  Referencia medida: una malla de 124 089 gdl usa ~2.6 GB "
                "por proceso con --solver amg."
            )
            print(
                "  Elige --procesos pensando en la MAS GRANDE, no en el "
                "promedio: si una sola no cabe, el batch muere ahi."
            )
        return 0

    os.makedirs(salida_raiz, exist_ok=True)
    casos = sorted({m[0] for m in mallas})

    print("=" * 78)
    print("  BATCH VEM 3D - CASO HOMOGENEO (verificacion)")
    print("=" * 78)
    print(f"  Raiz      : {raiz}")
    print(f"  Salida    : {salida_raiz}")
    print(f"  Material  : E = {core.E_HOM:.3e} Pa,  nu = {core.NU_HOM}")
    print(f"  Tolerancia: {args.tol:.1e}")
    print(f"  Casos     : {len(casos)}")
    print(f"  Mallas    : {len(mallas)}")
    print(f"  Procesos  : {'auto (se mide la malla mas grande)' if auto else args.procesos}")
    print(f"  Solver    : {args.solver}{'' if core.HAY_PYAMG else '   (pyamg NO instalado -> LU)'}")
    print("=" * 78)

    # De mayor a menor: la mas grande marca el techo de memoria, asi que
    # conviene medirla primero y fallar temprano si no cabe.
    mallas = sorted(mallas, key=lambda m: core.tamano_malla(m[3])[0], reverse=True)

    tareas = []
    for caso, rep, nombre_malla, carpeta_vem in mallas:
        carpeta_salida = os.path.join(salida_raiz, caso, rep)
        tareas.append(
            (
                nombre_malla,
                carpeta_vem,
                carpeta_salida,
                args.tol,
                not args.sin_campos,
                args.solver,
                args.tmp,
            )
        )

    t0 = time.perf_counter()
    filas = []

    # ── cuantos procesos ──────────────────────────────────────────────
    if auto:
        print(f"  Calibrando con la malla mas grande ({tareas[0][0]}) en 1 proceso...", flush=True)
        r = _tarea(tareas[0])
        filas.append(r)
        _log(1, len(tareas), r)
        pico = r["meta"]["memoria_pico_gb"] if r["ok"] else 0.0
        procesos, razon = core.procesos_recomendados(pico, n_tareas=len(tareas) - 1)
        print(f"  Pico medido: {pico:.2f} GB  ->  --procesos {procesos}")
        print(f"    ({razon})", flush=True)
        tareas = tareas[1:]
        if not tareas:
            procesos = 1
    else:
        procesos = args.procesos

    if procesos > 1:
        with ProcessPoolExecutor(max_workers=procesos) as ex:
            futuros = {ex.submit(_tarea, t): t[0] for t in tareas}
            for k, fut in enumerate(as_completed(futuros), len(filas) + 1):
                r = fut.result()
                filas.append(r)
                _log(k, len(tareas), r)
    else:
        for k, t in enumerate(tareas, len(filas) + 1):
            r = _tarea(t)
            filas.append(r)
            _log(k, len(tareas), r)

    tiempo_total = time.perf_counter() - t0
    filas.sort(key=lambda f: f["malla"])

    ruta_resumen = os.path.join(salida_raiz, "RESUMEN_GLOBAL.txt")
    ruta_csv = os.path.join(salida_raiz, "RESUMEN_PROPIEDADES.csv")
    core.escribir_resumen_global(ruta_resumen, filas, "homogeneo", tiempo_total, tol_C=args.tol)
    core.escribir_csv_global(ruta_csv, filas, "homogeneo")

    ok = [f for f in filas if f["ok"]]
    fail = [f for f in filas if not f["ok"]]

    print("=" * 78)
    print(f"  Procesadas : {len(filas)}    OK: {len(ok)}    ERROR: {len(fail)}")
    if ok:
        n_bien = sum(1 for f in ok if f["meta"]["verif_ok"])
        print(f"  Verificacion C   : {n_bien}/{len(ok)} coinciden (tolerancia {args.tol:.1e})")
        print(f"  Peor error en C  : {max(f['meta']['err_C_max'] for f in ok):.3e}")
        print(f"  Peor err_patch   : {max(f['err_patch'] for f in ok):.3e}")
        print(f"  Peor dif_volumen : {max(f['meta']['dif_volumen'] for f in ok):.3e}")
        malas = [f["malla"] for f in ok if not f["meta"]["verif_ok"]]
        if malas:
            print(f"  NO coinciden     : {', '.join(malas)}")
    if fail:
        print(f"  Con error        : {', '.join(f['malla'] for f in fail)}")
    print(f"  Tiempo total     : {tiempo_total:.1f}s ({tiempo_total / 60:.1f} min)")
    print()
    print(f"  {ruta_resumen}")
    print(f"  {ruta_csv}")
    print("=" * 78)

    return 0 if not fail and all(f["meta"]["verif_ok"] for f in ok) else 1


if __name__ == "__main__":
    sys.exit(main())
