# -*- coding: utf-8 -*-
"""
correr_una_malla.py
===================
Caso minimo reproducible: corre la homogenizacion VEM 3D sobre UNA sola
malla y registra todo lo necesario para diagnosticar un fallo.

Esta pensado para enviar junto con su salida: imprime el entorno, el
tamano del problema, la memoria en cada etapa y, si algo falla, el
traceback completo con el estado de la memoria en el momento del error.

USO
---
    python correr_una_malla.py <carpeta_VEM_de_la_malla> [--tmp D:\\temp_vem]

Ejemplo:
    python correr_una_malla.py "D:\\...\\3D\\CA50_A2_G101\\CA50_A2_G101_R03\\VEM"

Deja un archivo `log_<nombre_malla>.txt` junto al script con todo lo que
se imprime en pantalla.

Requiere `vem3d_core.py` en la misma carpeta.
"""

import os
import sys
import time
import argparse
import platform
import traceback

import numpy as np


# ──────────────────────────────────────────────────────────────────────
# Registro: todo lo que se imprime va tambien a un archivo
# ──────────────────────────────────────────────────────────────────────

class Tee:
    def __init__(self, ruta):
        self.archivo = open(ruta, "w", encoding="utf-8")
        self.consola = sys.stdout

    def write(self, texto):
        self.consola.write(texto)
        self.archivo.write(texto)
        self.archivo.flush()

    def flush(self):
        self.consola.flush()
        self.archivo.flush()


def memoria():
    """(comprometida_GB, pico_GB, disponible_GB). 0.0 donde no se pueda."""
    com = pico = disp = 0.0
    try:
        import psutil
        p = psutil.Process()
        mi = p.memory_info()
        com = getattr(mi, "private", getattr(mi, "rss", 0)) / 1024**3
        pico = getattr(mi, "peak_wset", getattr(mi, "rss", 0)) / 1024**3
        disp = psutil.virtual_memory().available / 1024**3
    except Exception:
        try:
            import resource
            pico = (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                    / (1024**3 if sys.platform == "darwin" else 1024**2))
        except Exception:
            pass
    return com, pico, disp


def hito(etiqueta, t0):
    com, pico, disp = memoria()
    print(f"  [{time.perf_counter()-t0:8.1f}s] {etiqueta:<34s}"
          f" en uso {com:6.2f} GB   pico {pico:6.2f} GB"
          f"   libre {disp:6.2f} GB", flush=True)


# ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Corre UNA malla VEM 3D y registra el fallo si lo hay")
    ap.add_argument("carpeta", help="Carpeta VEM de la malla (la que "
                                    "contiene Input_nodos.txt)")
    ap.add_argument("--tmp", default=None,
                    help="Carpeta para los temporales del ensamblaje")
    ap.add_argument("--solver", choices=["amg", "lu", "auto"], default="amg")
    ap.add_argument("--modo", choices=["heterogeneo", "homogeneo"],
                    default="heterogeneo")
    args = ap.parse_args()

    carpeta = os.path.abspath(args.carpeta)
    nombre = os.path.basename(os.path.dirname(carpeta)) or "malla"
    aqui = os.path.dirname(os.path.abspath(__file__))
    sys.stdout = Tee(os.path.join(aqui, f"log_{nombre}.txt"))

    print("=" * 78)
    print("  CASO MINIMO REPRODUCIBLE - VEM 3D")
    print("=" * 78)

    # ── entorno ──────────────────────────────────────────────────────
    print("\nENTORNO")
    print(f"  fecha             : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  sistema           : {platform.platform()}")
    print(f"  procesador        : {platform.processor()}")
    print(f"  python            : {sys.version.split()[0]}  ({sys.executable})")
    print(f"  numpy             : {np.__version__}")
    try:
        import scipy
        print(f"  scipy             : {scipy.__version__}")
    except Exception as ex:
        print(f"  scipy             : NO DISPONIBLE ({ex})")
    try:
        import pyamg
        print(f"  pyamg             : {pyamg.__version__}")
    except Exception as ex:
        print(f"  pyamg             : NO INSTALADO ({ex})")
    try:
        import psutil
        vm = psutil.virtual_memory()
        print(f"  RAM total         : {vm.total/1024**3:.1f} GB")
        print(f"  RAM disponible    : {vm.available/1024**3:.1f} GB")
        print(f"  nucleos fisicos   : {psutil.cpu_count(logical=False)}"
              f"   logicos: {psutil.cpu_count(logical=True)}")
        sw = psutil.swap_memory()
        print(f"  archivo paginacion: {sw.total/1024**3:.1f} GB total, "
              f"{sw.used/1024**3:.1f} GB en uso")
    except Exception as ex:
        print(f"  psutil            : NO INSTALADO ({ex})")

    sys.path.insert(0, aqui)
    import vem3d_core as core
    print(f"  vem3d_core        : {os.path.join(aqui, 'vem3d_core.py')}")
    print(f"  AMG               : smooth={core.AMG_SMOOTH}, "
          f"strength={core.AMG_STRENGTH}")

    print("\nMALLA")
    print(f"  carpeta           : {carpeta}")
    print(f"  modo              : {args.modo}")
    print(f"  solver            : {args.solver}")
    print(f"  temporales        : {args.tmp or '(temporal del sistema)'}")

    t0 = time.perf_counter()
    try:
        # ── lectura ──────────────────────────────────────────────────
        f = lambda *n: core._buscar_archivo(carpeta, *n)
        nodos = core.leer_nodos(f("Input_nodos.txt", "input_nodos.txt"))
        aristas = core.leer_aristas(f("Input_aristas.txt",
                                      "input_aristas.txt"))
        caras = core.leer_caras(f("Input_caras.txt", "input_caras.txt"))
        elementos, codigos = core.leer_elementos(
            f("input_elements.txt", "Input_elements.txt",
              "input_elementos.txt"))

        if args.modo == "homogeneo":
            core.asignar_propiedades_uniformes(elementos, core.E_HOM,
                                               core.NU_HOM)
        else:
            props = core.leer_propiedades(
                f("Propiedades.txt", "propiedades.txt"))
            core.asignar_propiedades(elementos, props, codigos)
            print(f"\n  materiales        : {props}")

        ndof = 3 * len(nodos)
        print(f"\n  nodos             : {len(nodos):,}")
        print(f"  aristas           : {len(aristas):,}")
        print(f"  caras             : {len(caras):,}")
        print(f"  elementos         : {len(elementos):,}")
        print(f"  grados de libertad: {ndof:,}")

        print("\nEJECUCION")
        hito("lectura", t0)

        # ── geometria de caras ───────────────────────────────────────
        infocaras = core.construir_infocaras(nodos, aristas, caras)
        hito("infocaras", t0)

        # ── ensamblaje (impone el borde al ensamblar) ────────────────
        borde = core.detectar_borde(nodos)
        n_borde = int(borde.sum())
        print(f"           nodos en el borde: {n_borde:,}"
              f"   gdl libres: {ndof - 3*n_borde:,}")

        info = {}
        part = core.ensamblar_particionado(nodos, aristas, caras, elementos,
                                           infocaras, info, borde,
                                           base_tmp=args.tmp)
        del infocaras
        print(f"           no-ceros de K_ll : {part['K_ll'].nnz:,}")
        hito("ensamblaje", t0)

        # ── solver ───────────────────────────────────────────────────
        sistema = core.factorizar_particionado(part, solver=args.solver)
        del part
        hito("preparacion del solver", t0)

        x, y, z = nodos[:, 3], nodos[:, 4], nodos[:, 5]
        V_bbox = np.ptp(x) * np.ptp(y) * np.ptp(z)
        C, C_f, res, isol = core.homogenizar(nodos, elementos, info, sistema,
                                             borde, V_bbox)
        core.liberar_sistema(sistema)
        hito("homogenizacion (6 casos)", t0)

        # ── resultados ───────────────────────────────────────────────
        E_eq, nu_eq = core.ajustar_E_nu(C)
        print("\nRESULTADO")
        print(f"  E_eq              : {E_eq:.8e} Pa")
        print(f"  nu_eq             : {nu_eq:.8f}")
        print(f"  solver            : {isol['nombre']}, "
              f"iteraciones max {isol['iters_max']}, "
              f"residuo rel. {isol['res_rel_max']:.3e}")
        print(f"  dif. entre las dos vias de C: "
              f"{np.abs(C-C_f).max()/np.abs(C).max():.3e}")
        print(f"  peor desviacion del campo lineal: "
              f"{max(r['err_rel'] for r in res.values()):.3e}")
        print("\n  MATRIZ C (Pa)")
        for i in range(6):
            print("   " + "".join(f"{C[i, j]:>+15.5e}" for j in range(6)))
        print(f"\n  TERMINO BIEN en {time.perf_counter()-t0:.1f} s")
        return 0

    except BaseException:
        com, pico, disp = memoria()
        print("\n" + "=" * 78)
        print("  FALLO")
        print("=" * 78)
        print(f"  tiempo hasta el fallo : {time.perf_counter()-t0:.1f} s")
        print(f"  memoria en uso        : {com:.2f} GB")
        print(f"  pico de memoria       : {pico:.2f} GB")
        print(f"  memoria libre         : {disp:.2f} GB")
        print("\n  TRACEBACK COMPLETO\n")
        traceback.print_exc(file=sys.stdout)
        print("\n" + "=" * 78)
        return 1


if __name__ == "__main__":
    sys.exit(main())
