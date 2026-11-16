"""VEM 3D: resources."""

import os
import sys


def memoria_pico_gb():
    """
    Pico de memoria residente del proceso actual, en GB. Devuelve 0.0 si no
    se puede medir. En Windows usa psutil (peak_wset) y en Linux/macOS el
    ru_maxrss de resource, cuya unidad cambia entre sistemas.
    """
    try:
        import psutil

        mi = psutil.Process().memory_info()
        if hasattr(mi, "peak_wset"):  # Windows
            return float(mi.peak_wset) / 1024**3
    except Exception:
        pass
    try:
        import resource

        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux reporta KiB; macOS, bytes.
        return float(r) / (1024**3 if sys.platform == "darwin" else 1024**2)
    except Exception:
        return 0.0


def memoria_disponible_gb():
    """RAM disponible ahora mismo, en GB. 0.0 si no se puede averiguar."""
    try:
        import psutil

        return psutil.virtual_memory().available / 1024**3
    except Exception:
        return 0.0


def nucleos_fisicos():
    """Nucleos fisicos (sin hyperthreading); cae a los logicos si no se sabe."""
    try:
        import psutil

        n = psutil.cpu_count(logical=False)
        if n:
            return int(n)
    except Exception:
        pass
    return os.cpu_count() or 1


def procesos_recomendados(pico_gb, n_tareas=None, fraccion=0.75):
    """
    Cuantos procesos caben, dado el pico de memoria medido en UNA malla.

    Se reserva (1 - fraccion) de la RAM disponible para el sistema y para
    que el pico real de alguna malla pueda superar al medido. El limite
    duro son los nucleos fisicos: por encima de eso no hay ganancia en este
    tipo de trabajo, que esta limitado por ancho de banda de memoria.

    Devuelve (n_procesos, explicacion).
    """
    cores = nucleos_fisicos()
    disp = memoria_disponible_gb()
    # Slurm puede asignar sólo una fracción de la RAM/CPU visibles en el nodo.
    allocated_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if allocated_cpus:
        cores = min(cores, max(1, int(allocated_cpus)))
    if hasattr(os, "sched_getaffinity"):
        cores = min(cores, len(os.sched_getaffinity(0)))
    allocated_memory = os.environ.get("SLURM_MEM_PER_NODE")
    if allocated_memory:
        disp = min(disp, float(allocated_memory) / 1024)
    elif os.environ.get("SLURM_MEM_PER_CPU"):
        disp = min(disp, float(os.environ["SLURM_MEM_PER_CPU"]) * cores / 1024)

    if pico_gb <= 0 or disp <= 0:
        return 1, (
            "no se pudo medir la memoria (instala psutil para que "
            "esto funcione); se usa 1 proceso por seguridad"
        )

    caben = int((disp * fraccion) // pico_gb)
    n = max(1, min(cores, caben))
    if n_tareas:
        n = max(1, min(n, int(n_tareas)))

    razon = (
        f"{disp:.1f} GB disponibles x {fraccion:.0%} / {pico_gb:.2f} GB "
        f"por malla = {caben} caben; nucleos fisicos = {cores}"
    )
    if caben == 0:
        razon += "  (ni una sola malla cabe con holgura: cuidado)"
    return n, razon
