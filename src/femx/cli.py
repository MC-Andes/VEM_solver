"""Homogenización FEM P1 con FEniCSx sobre una malla Gmsh 2D o 3D."""

import argparse
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "malla", type=Path, help="Archivo Gmsh .msh con grupos físicos de materiales"
    )
    parser.add_argument("--propiedades", type=Path, required=True)
    parser.add_argument("--dimension", type=int, choices=(2, 3), required=True)
    parser.add_argument("--hipotesis", choices=("tension-plana", "deformacion-plana"), default=None)
    parser.add_argument("--modo", choices=("homogeneo", "heterogeneo"), default="heterogeneo")
    parser.add_argument("--solver", choices=("auto", "lu", "amg"), default="auto")
    parser.add_argument(
        "--mapa-materiales",
        type=Path,
        help='JSON: etiqueta física a código de Propiedades.txt, por ejemplo {"1": 0, "2": 1}',
    )
    parser.add_argument("--salida", type=Path)
    parser.add_argument("--sin-campos", action="store_true")
    parser.add_argument("--tol", type=float, default=1e-6)
    args = parser.parse_args(argv)
    import math

    if not math.isfinite(args.tol) or args.tol <= 0:
        parser.error("--tol debe ser positivo y finito")
    if args.dimension == 3 and args.hipotesis is not None:
        parser.error("--hipotesis se usa sólo en 2D; 3D usa elasticidad tridimensional")
    args.hipotesis = args.hipotesis or ("tension-plana" if args.dimension == 2 else "3d")
    try:
        from mpi4py import MPI

        if MPI.COMM_WORLD.size != 1:
            raise ValueError("fem-run usa un solo proceso MPI; ejecútalo sin mpirun")
        from .solver import solve
    except (ImportError, ValueError) as error:
        print(
            f"No se pudo iniciar FEniCSx: {error}. Consulta la instalación FEM en el README.",
            file=sys.stderr,
        )
        return 1
    output = (
        args.salida
        or Path("resultados") / f"fem_{args.dimension}d_{time.strftime('%Y%m%dT%H%M%S')}"
    )
    if output.exists() and any(output.iterdir()):
        parser.error(f"La salida no está vacía: {output}")
    output.mkdir(parents=True, exist_ok=True)
    manifest = dict(
        parameters={
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        started_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    )
    try:
        import dolfinx
        import gmsh
        import numpy as np
        import ufl
        from petsc4py import PETSc

        manifest["python"] = sys.version
        manifest["dependencies"] = {
            "numpy": np.__version__,
            "dolfinx": dolfinx.__version__,
            "ufl": ufl.__version__,
            "gmsh": gmsh.__version__,
            "petsc": PETSc.Sys.getVersion(),
        }
        root = Path(__file__).parents[1]
        manifest["source_sha256"] = {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for package in ("femx", "vem2d", "vem3d")
            for p in sorted((root / package).rglob("*.py"))
        }
        inputs = [args.malla, args.propiedades] + (
            [args.mapa_materiales] if args.mapa_materiales else []
        )
        manifest["inputs_sha256"] = {}
        for path in inputs:
            with path.open("rb") as stream:
                manifest["inputs_sha256"][str(path.resolve())] = hashlib.file_digest(
                    stream, "sha256"
                ).hexdigest()
        mapping = json.loads(args.mapa_materiales.read_text()) if args.mapa_materiales else None
        result = solve(
            args.malla,
            args.propiedades,
            args.dimension,
            args.modo,
            args.hipotesis,
            args.solver,
            mapping,
            output,
            not args.sin_campos,
            args.tol,
        )
        manifest["result"] = result
        for key, name in (("C", "C_equivalente.txt"), ("C_fuerzas", "C_equivalente_fuerzas.txt")):
            np.savetxt(output / name, result[key])
        (output / "resumen.txt").write_text(
            "FEM / FEniCSx\n"
            + "\n".join(
                f"{k}: {v}" for k, v in result.items() if k not in ("C", "C_fuerzas", "cargas")
            )
            + "\n",
            encoding="utf-8",
        )
        print(
            f"E={result['E_eq']:.9e}, nu={result['nu_eq']:.9f}, dif_vias={result['dif_vias']:.3e}"
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


if __name__ == "__main__":
    sys.exit(main())
