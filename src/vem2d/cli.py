"""Ejecuta homogenización VEM 2D sobre una malla o un lote."""

import argparse
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path

from vem3d.mesh import descubrir_mallas

from . import __version__
from .mesh import INPUTS
from .workflow import solve


def _options(parser):
    parser.add_argument("--salida", type=Path)
    parser.add_argument("--modo", choices=["homogeneo", "heterogeneo"], default="heterogeneo")
    parser.add_argument(
        "--hipotesis", choices=["tension-plana", "deformacion-plana"], default="tension-plana"
    )
    parser.add_argument("--solver", choices=["auto", "lu", "amg"], default="auto")
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--sin-campos", action="store_true")
    parser.add_argument("--inicio-cero", action="store_true")


def _run(directory, output, args):
    import math
    import platform
    from importlib.metadata import version

    import numpy as np

    if not math.isfinite(args.tol) or args.tol <= 0:
        raise ValueError("--tol debe ser positivo y finito")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"La carpeta de salida no está vacía: {output}")
    output.mkdir(parents=True, exist_ok=True)
    manifest = dict(
        version=__version__,
        dimension=2,
        method="VEM",
        mesh=str(Path(directory).resolve()),
        parameters={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        started_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        python=platform.python_version(),
        dependencies={name: version(name) for name in ("numpy", "scipy", "pyamg")},
    )
    root = Path(__file__).parents[1]
    manifest["source_sha256"] = {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for package in ("vem2d", "vem3d")
        for p in sorted((root / package).rglob("*.py"))
    }
    try:
        files = {p.name.lower(): p for p in Path(directory).iterdir() if p.is_file()}
        manifest["inputs_sha256"] = {}
        for name in INPUTS:
            with files[name.lower()].open("rb") as stream:
                manifest["inputs_sha256"][name] = hashlib.file_digest(stream, "sha256").hexdigest()
        result = solve(
            directory,
            args.modo,
            args.hipotesis,
            args.solver,
            output,
            not args.sin_campos,
            not args.inicio_cero,
            args.tol,
        )
        manifest["result"] = result
        np.savetxt(
            output / "C_equivalente.txt",
            result["C"],
            header="Voigt: xx yy xy; deformaciones cortantes ingenieriles",
        )
        np.savetxt(output / "C_equivalente_fuerzas.txt", result["C_fuerzas"])
        lines = ["VEM 2D", f"Modo: {args.modo}; hipótesis: {args.hipotesis}"]
        lines.extend(
            f"{key}: {value}"
            for key, value in result.items()
            if key not in ("C", "C_fuerzas", "cargas")
        )
        (output / "resumen.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(
            f"{directory}: E={result['E_eq']:.9e}, nu={result['nu_eq']:.9f}, residuo={result['solver_res_rel']:.3e}, dif_vias={result['dif_vias']:.3e}",
            flush=True,
        )
        return 0 if result["ok"] else 2
    except Exception:
        manifest["error"] = traceback.format_exc()
        (output / "ERROR.txt").write_text(manifest["error"], encoding="utf-8")
        print(manifest["error"], file=sys.stderr)
        return 1
    finally:
        manifest["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        (output / "run.json").write_text(
            json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8"
        )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("carpeta", type=Path)
    _options(parser)
    args = parser.parse_args(argv)
    output = (
        args.salida
        or Path("resultados") / f"{args.carpeta.name}_2d_{time.strftime('%Y%m%dT%H%M%S')}"
    )
    try:
        return _run(args.carpeta, output, args)
    except (ValueError, OSError) as error:
        parser.error(str(error))


def batch(argv=None, mode="heterogeneo"):
    parser = argparse.ArgumentParser(
        description="Procesa mallas 2D secuencialmente: raíz/caso/réplica/VEM"
    )
    parser.add_argument("--raiz", type=Path, required=True)
    parser.add_argument("--listar", action="store_true")
    _options(parser)
    parser.set_defaults(modo=mode)
    args = parser.parse_args(argv)
    cases = descubrir_mallas(args.raiz)
    if not cases:
        parser.error("No hay mallas bajo raíz/caso/réplica/VEM")
    if args.listar:
        for _, _, _, directory in cases:
            print(directory)
        return 0
    output = args.salida or Path("resultados") / f"lote_2d_{time.strftime('%Y%m%dT%H%M%S')}"
    status = 0
    for case, replica, _, directory in cases:
        try:
            status = max(status, _run(directory, output / case / replica, args))
        except (ValueError, OSError) as error:
            print(error, file=sys.stderr)
            status = max(status, 1)
    return status


def batch_homogeneous():
    return batch(mode="homogeneo")


def batch_heterogeneous():
    return batch(mode="heterogeneo")


if __name__ == "__main__":
    sys.exit(main())
