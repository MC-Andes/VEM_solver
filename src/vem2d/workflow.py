"""Tres cargas KUBC y homogenización por esfuerzos y reacciones."""

import time
from pathlib import Path

import numpy as np

from vem3d.solvers import resolver

from .assembly import assemble, prepare_system
from .materials import constitutive, isotropic_fit
from .mesh import read_mesh

LOADS = (("exx", 1.0, 0.0, 0.0), ("eyy", 0.0, 1.0, 0.0), ("gxy", 0.0, 0.0, 1.0))


def affine_field(mesh, exx, eyy, gamma):
    xy = mesh.coordinates - mesh.coordinates.min(axis=0)
    U = np.empty(2 * len(xy))
    U[mesh.dofs[:, 0]] = exx * xy[:, 0] + 0.5 * gamma * xy[:, 1]
    U[mesh.dofs[:, 1]] = eyy * xy[:, 1] + 0.5 * gamma * xy[:, 0]
    return U


def solve(
    directory,
    mode="heterogeneo",
    hypothesis="tension-plana",
    solver="auto",
    output=None,
    save_fields=False,
    affine_start=True,
    tolerance=1e-6,
):
    start = time.perf_counter()
    mesh = read_mesh(directory)
    K, info = assemble(mesh, mode, hypothesis)
    system = prepare_system(K, mesh, solver)
    del K
    C, forces = np.zeros((3, 3)), np.zeros((3, 3))
    areas = np.array([cell[3] for cell in info])
    loads, residuals = {}, []
    xy = mesh.coordinates - mesh.coordinates.min(axis=0)
    for col, (name, exx, eyy, gamma) in enumerate(LOADS):
        exact = affine_field(mesh, exx, eyy, gamma)
        U = resolver(
            system, exact[system["idx_f"]], x0=exact[system["idx_l"]] if affine_start else None
        )
        rhs = -system["K_lf"] @ U[system["idx_f"]]
        residual = system["K_ll"] @ U[system["idx_l"]] - rhs
        relative = (
            float(np.linalg.norm(residual) / np.linalg.norm(rhs))
            if np.any(rhs)
            else float(np.linalg.norm(residual))
        )
        if not np.isfinite(U).all() or not np.isfinite(relative) or relative > 1e-10:
            raise RuntimeError(f"{name}: residuo no aceptable ({relative:.3e})")
        residuals.append(relative)
        strain = np.array([B @ U[dofs] for dofs, B, _, _ in info])
        stress = np.array([material @ e for (_, _, material, _), e in zip(info, strain)])
        C[:, col] = areas @ stress / areas.sum()
        reaction = np.zeros_like(U)
        reaction[system["idx_f"]] = system["K_f"] @ U
        moment = reaction[mesh.dofs].T @ xy / mesh.area
        forces[:, col] = (moment[0, 0], moment[1, 1], 0.5 * (moment[0, 1] + moment[1, 0]))
        loads[name] = {
            "residual": relative,
            "affine_error": float(np.max(np.abs(U - exact)) / np.max(np.abs(exact))),
        }
        if save_fields and output is not None:
            output = Path(output)
            np.savetxt(
                output / f"U_{name}.txt",
                np.column_stack((mesh.coordinates, U[mesh.dofs])),
                header="x y ux uy",
            )
            np.savetxt(
                output / f"deformaciones_{name}.txt",
                strain,
                header="exx eyy gamma_xy; fila = id_elemento",
            )
            np.savetxt(
                output / f"esfuerzos_{name}.txt", stress, header="sxx syy txy; fila = id_elemento"
            )
    E, nu, anisotropy = isotropic_fit(C, hypothesis)
    theoretical = constitutive(1e6, 0.3, hypothesis)
    error = (
        float(np.max(np.abs(C - theoretical)) / np.max(np.abs(theoretical)))
        if mode == "homogeneo"
        else None
    )
    result = dict(
        C=C.tolist(),
        C_fuerzas=forces.tolist(),
        E_eq=E,
        nu_eq=nu,
        anisotropia=anisotropy,
        modo=mode,
        hipotesis=hypothesis,
        n_nodos=len(mesh.coordinates),
        n_elementos=len(mesh.cells),
        n_dofs=2 * len(mesh.coordinates),
        area=float(areas.sum()),
        area_bbox=mesh.area,
        fraccion_agregado=float(areas[mesh.codes != 0].sum() / areas.sum()),
        solver=system["solver"],
        solver_iters_max=max(system["iters"], default=0),
        solver_res_rel=max(residuals),
        dif_vias=float(np.max(np.abs(C - forces)) / np.max(np.abs(C))),
        err_C=error,
        err_patch=max(v["affine_error"] for v in loads.values()),
        cargas=loads,
        tiempo_s=time.perf_counter() - start,
    )
    result["ok"] = error is None or error <= tolerance
    return result
