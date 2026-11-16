"""Una malla: resultados científicos y manifiesto reproducible de la ejecución."""

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np

from .. import __version__
from ..workflow import correr_malla


def _json(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("carpeta", type=Path, help="Directorio con archivos VEM")
    parser.add_argument("--salida", type=Path, default=None)
    parser.add_argument("--tmp", default=None)
    parser.add_argument("--solver", choices=["amg", "lu", "auto"], default="amg")
    parser.add_argument("--modo", choices=["homogeneo", "heterogeneo"], default="heterogeneo")
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--sin-campos", action="store_true")
    parser.add_argument(
        "--inicio-cero",
        action="store_true",
        help="Usar cero como inicio de AMG, para comparar con el algoritmo histórico",
    )
    args = parser.parse_args(argv)
    if not np.isfinite(args.tol) or args.tol <= 0:
        parser.error("--tol debe ser positivo y finito")
    mesh = args.carpeta.resolve()
    name = mesh.parent.name if mesh.name.lower() == "vem" else mesh.name
    output = args.salida or Path("results") / f"{name}_{args.modo}_{time.strftime('%Y%m%dT%H%M%S')}"
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "executable": sys.executable,
        "vem3d": __version__,
        "mesh": str(mesh),
        "mode": args.modo,
        "solver": args.solver,
        "tolerance": args.tol,
        "initial_guess": "zero" if args.inicio_cero else "affine",
        "dependencies": {p: version(p) for p in ("numpy", "scipy", "pyamg", "psutil")},
        "inputs": {},
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "source_sha256": {
            str(p.relative_to(Path(__file__).parents[1])): hashlib.sha256(
                p.read_bytes()
            ).hexdigest()
            for p in sorted(Path(__file__).parents[1].rglob("*.py"))
        },
    }
    # Hash only VEM inputs, streaming blocks: do not load giant text files twice in RAM.
    if mesh.is_dir():
        for path in sorted(mesh.iterdir()):
            if path.is_file() and path.suffix.lower() == ".txt":
                digest = hashlib.sha256()
                with path.open("rb") as stream:
                    for block in iter(lambda: stream.read(8 * 1024**2), b""):
                        digest.update(block)
                manifest["inputs"][path.name] = {
                    "bytes": path.stat().st_size,
                    "sha256": digest.hexdigest(),
                }
    print(
        f"Malla: {mesh}\nSalida: {output.resolve()}\nModo: {args.modo}; solver: {args.solver}",
        flush=True,
    )
    manifest_path = output / "run.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    result = correr_malla(
        str(mesh),
        str(output),
        name,
        args.modo,
        tol_C=args.tol,
        guardar_campos=not args.sin_campos,
        solver=args.solver,
        base_tmp=args.tmp,
        inicio_afin=not args.inicio_cero,
    )
    manifest["result"] = result
    manifest["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, default=_json, allow_nan=False), encoding="utf-8"
    )
    if not result["ok"]:
        print(result.get("traceback", result["error"]), file=sys.stderr)
        return 1
    meta = result["meta"]
    print(
        f"E = {meta['E_eq']:.9e}; nu = {meta['nu_eq']:.9f}; "
        f"dif_vias = {meta['dif_vias']:.3e}; residuo = {meta['solver_res_rel']:.3e}"
    )
    if args.modo == "homogeneo":
        print(f"err_C = {meta['err_C_max']:.3e}; patch = {result['err_patch']:.3e}")
        if not meta["verif_ok"]:
            return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
