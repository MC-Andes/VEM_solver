"""Compara el kernel recibido y el optimizado, dentro de un job en Hypatia."""

import importlib.util
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from examples.cube import prism_mesh
from vem3d.element import elemento_VEM3D
from vem3d.geometry import construir_infocaras

root = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("baseline", root / "reference/original/vem3d_core.py")
baseline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(baseline)
rows = []
for sides in (4, 16, 32):
    nodes, edges, faces, elems = prism_mesh(sides)
    fi = construir_infocaras(nodes, edges, faces)
    old_fi = baseline.construir_infocaras(nodes, edges, faces)
    measurements = {"original": [], "optimized": []}
    # Alternar orden por repetición para reducir el sesgo de calentamiento.
    for repeat in range(22):
        methods = [("original", baseline.elemento_VEM3D, old_fi), ("optimized", elemento_VEM3D, fi)]
        if repeat % 2:
            methods.reverse()
        matrices = {}
        for name, method, face_data in methods:
            info = {}
            start = time.perf_counter()
            matrix, _ = method(0, nodes, edges, faces, elems, face_data, info)
            elapsed = time.perf_counter() - start
            matrices[name] = matrix
            if repeat > 1:
                measurements[name].append(elapsed)
        assert (
            np.max(np.abs(matrices["original"] - matrices["optimized"])) / np.max(np.abs(matrix))
            < 1e-12
        )
    original, optimized = (
        statistics.median(measurements[name]) for name in ("original", "optimized")
    )
    record = info[0]
    full_bytes = sum(v.nbytes for v in record.values() if isinstance(v, np.ndarray))
    strain = record["dpi"] @ record["piast"]
    compact_bytes = full_bytes - record["dpi"].nbytes - record["piast"].nbytes + strain.nbytes
    rows.append(
        {
            "nodes": 2 * sides,
            "original_seconds_median": original,
            "optimized_seconds_median": optimized,
            "speedup": original / optimized,
            "info_arrays_bytes_original": full_bytes,
            "info_arrays_bytes_compact": compact_bytes,
            "samples": 20,
        }
    )
report = {
    "host": platform.node(),
    "python": sys.version,
    "numpy": np.__version__,
    "benchmarks": rows,
}
print(json.dumps(report, indent=2))
if len(sys.argv) > 1:
    Path(sys.argv[1]).write_text(json.dumps(report, indent=2))
