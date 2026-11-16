"""Elasticidad FEM P1 y KUBC con tres (2D) o seis (3D) cargas.

Ensamblaje PETSc basado en las interfaces públicas de DOLFINx 0.9–0.11.
"""

import time
from pathlib import Path

import numpy as np

from vem2d.materials import constitutive, isotropic_fit

from .mesh import load_mesh, read_materials


def solve(
    mesh_path,
    properties_path,
    dimension,
    mode="heterogeneo",
    hypothesis="tension-plana",
    solver="auto",
    material_map=None,
    output=None,
    save_fields=True,
    tolerance=1e-6,
):
    import dolfinx
    import ufl
    from dolfinx import fem, io
    from dolfinx import mesh as dmesh
    from dolfinx.fem import petsc as fp
    from petsc4py import PETSc

    start = time.perf_counter()
    if dimension not in (2, 3):
        raise ValueError("La dimensión debe ser 2 o 3")
    if dimension == 2 and hypothesis not in ("tension-plana", "deformacion-plana"):
        raise ValueError("Hipótesis 2D desconocida")
    properties = read_materials(properties_path)
    msh, tags, mapping, names = load_mesh(mesh_path, dimension, properties, material_map)
    dim = dimension
    expected_cell = dmesh.CellType.triangle if dim == 2 else dmesh.CellType.tetrahedron
    if msh.topology.cell_type != expected_cell:
        raise ValueError("FEM P1 requiere triángulos en 2D o tetraedros en 3D")
    if np.issubdtype(PETSc.ScalarType, np.complexfloating):
        raise ValueError("fem-run requiere una instalación PETSc de escalares reales")
    if mode not in ("homogeneo", "heterogeneo"):
        raise ValueError("Modo desconocido")
    V = fem.functionspace(msh, ("Lagrange", 1, (dim,)))
    Q = fem.functionspace(msh, ("DG", 0))
    lam, mu = fem.Function(Q), fem.Function(Q)
    for tag, code in mapping.items():
        cells = tags.find(tag)
        E, nu = properties[code] if mode == "heterogeneo" else (1e6, 0.3)
        shear = E / (2 * (1 + nu))
        if dim == 2 and hypothesis == "tension-plana":
            first = E * nu / (1 - nu**2)
        else:
            first = E * nu / ((1 + nu) * (1 - 2 * nu))
        for cell in cells:
            dof = Q.dofmap.cell_dofs(cell)
            lam.x.array[dof], mu.x.array[dof] = first, shear
    lam.x.scatter_forward()
    mu.x.scatter_forward()
    low, high = msh.geometry.x[:, :dim].min(axis=0), msh.geometry.x[:, :dim].max(axis=0)
    lengths = high - low
    if np.any(lengths <= 0):
        raise ValueError("Dominio degenerado")
    msh.topology.create_connectivity(dim - 1, dim)
    facets = dmesh.exterior_facet_indices(msh.topology)
    fixed_blocks = fem.locate_dofs_topological(V, dim - 1, facets)
    coordinates = V.tabulate_dof_coordinates()[:, :dim]
    normalized = (coordinates[fixed_blocks] - low) / lengths
    if not np.all(np.any((np.abs(normalized) < 1e-8) | (np.abs(normalized - 1) < 1e-8), axis=1)):
        raise ValueError("El dominio debe ser una caja/rectángulo sin huecos interiores")
    fixed = (fixed_blocks[:, None] * dim + np.arange(dim)).ravel()
    volume = float(fem.assemble_scalar(fem.form(fem.Constant(msh, PETSc.ScalarType(1)) * ufl.dx)))
    bbox = float(np.prod(lengths))
    if abs(volume - bbox) / bbox > 1e-7:
        raise ValueError("El volumen/área FEM no coincide con la caja exterior")

    def stress(w):
        strain = ufl.sym(ufl.grad(w))
        return 2 * mu * strain + lam * ufl.tr(strain) * ufl.Identity(dim)

    trial, test = ufl.TrialFunction(V), ufl.TestFunction(V)
    energy = ufl.inner(stress(trial), ufl.sym(ufl.grad(test))) * ufl.dx
    a = fem.form(energy)
    L = fem.form(ufl.inner(fem.Constant(msh, np.zeros(dim, dtype=PETSc.ScalarType)), test) * ufl.dx)
    boundary_value, solution = fem.Function(V), fem.Function(V)
    bc = fem.dirichletbc(boundary_value, fixed_blocks)
    A = fp.assemble_matrix(a, bcs=[bc])
    A.assemble()
    ksp = PETSc.KSP().create(msh.comm)
    resources = [A, ksp]
    try:
        ksp.setOperators(A)
        ndof = V.dofmap.index_map.size_global * dim
        if solver == "auto":
            solver = "lu" if ndof < 40000 else "amg"
        if solver == "lu":
            ksp.setType("preonly")
            ksp.getPC().setType("lu")
            if PETSc.Sys.hasExternalPackage("mumps"):
                ksp.getPC().setFactorSolverType("mumps")
        elif solver == "amg":
            ksp.setType("cg")
            ksp.getPC().setType("gamg")
            ksp.setNormType(PETSc.KSP.NormType.UNPRECONDITIONED)
            # Modos rígidos de traslación/rotación para el precondicionador elástico.
            basis = []
            for axis in range(dim):
                v = np.zeros((len(coordinates), dim))
                v[:, axis] = 1
                basis.append(v.ravel())
            for i in range(dim):
                for j in range(i + 1, dim):
                    v = np.zeros((len(coordinates), dim))
                    v[:, i] = -(coordinates[:, j] - low[j])
                    v[:, j] = coordinates[:, i] - low[i]
                    basis.append(v.ravel())
            modes = np.column_stack(basis)
            modes[fixed] = 0
            if len(fixed) < ndof:
                orthogonal, triangular = np.linalg.qr(modes)
                vectors = [
                    PETSc.Vec().createWithArray(orthogonal[:, i].copy(), bsize=dim, comm=msh.comm)
                    for i in range(len(basis))
                    if abs(triangular[i, i]) > 1e-10
                ]
                nullspace = PETSc.NullSpace().create(vectors=vectors, comm=msh.comm)
                A.setNearNullSpace(nullspace)
                resources.extend(vectors + [nullspace])
            ksp.setTolerances(rtol=1e-12, atol=1e-14, max_it=2000)
        else:
            raise ValueError(f"Solver desconocido: {solver}")
        components = (
            [(0, 0), (1, 1), (0, 1)]
            if dim == 2
            else [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]
        )
        load_names = (
            ["exx", "eyy", "gxy"] if dim == 2 else ["exx", "eyy", "ezz", "gxy", "gxz", "gyz"]
        )
        macros = []
        for i, j in components:
            field = fem.Function(V)
            strain = np.zeros((dim, dim))
            strain[i, j] = strain[j, i] = 1 if i == j else 0.5
            field.interpolate(lambda x, e=strain: e @ (x[:dim] - low[:, None]))
            macros.append(field)
        C, forces = np.zeros((len(macros), len(macros))), np.zeros((len(macros), len(macros)))
        results = {}
        for col, (name, macro) in enumerate(zip(load_names, macros)):
            boundary_value.x.array[:] = macro.x.array
            solution.x.array[:] = macro.x.array if solver == "amg" else 0
            boundary_value.x.scatter_forward()
            b = fp.assemble_vector(L)
            fp.apply_lifting(b, [a], bcs=[[bc]])
            b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
            fp.set_bc(b, [bc])
            ksp.setInitialGuessNonzero(solver == "amg")
            ksp.solve(b, solution.x.petsc_vec)
            solution.x.scatter_forward()
            residual = b.duplicate()
            A.mult(solution.x.petsc_vec, residual)
            residual.axpy(-1, b)
            relative = residual.norm() / b.norm() if b.norm() else residual.norm()
            residual.destroy()
            b.destroy()
            if (
                ksp.getConvergedReason() <= 0
                or not np.isfinite(relative)
                or relative > 1e-10
                or not np.isfinite(solution.x.array).all()
            ):
                raise RuntimeError(
                    f"FEM no convergió en {name}: razón={ksp.getConvergedReason()}, residuo={relative:.3e}"
                )
            for row, (i, j) in enumerate(components):
                C[row, col] = (
                    fem.assemble_scalar(fem.form(stress(solution)[i, j] * ufl.dx)) / volume
                )
            reaction = fp.assemble_vector(fem.form(ufl.action(energy, solution)))
            reaction.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
            for row, virtual in enumerate(macros):
                forces[row, col] = np.dot(virtual.x.array[fixed], reaction.array[fixed]) / volume
            reaction.destroy()
            error = float(
                np.max(np.abs(solution.x.array - macro.x.array)) / np.max(np.abs(macro.x.array))
            )
            results[name] = dict(
                residual=float(relative), iterations=ksp.getIterationNumber(), affine_error=error
            )
            print(
                f"FEM {dim}D {name}: iteraciones={ksp.getIterationNumber()}, residuo={relative:.3e}",
                flush=True,
            )
            if save_fields and output is not None:
                solution.name = f"u_{name}"
                with io.XDMFFile(msh.comm, str(Path(output) / f"U_{name}.xdmf"), "w") as xdmf:
                    xdmf.write_mesh(msh)
                    xdmf.write_function(solution)
                    # Componentes escalares con nombre: XDMF no interpreta un Voigt
                    # de seis componentes como un vector espacial de tres componentes.
                    field_space = Q
                    expressions = {}
                    axes = "xyz"
                    for i, j in components:
                        suffix = axes[i] + axes[j]
                        expressions[("s" if i == j else "t") + suffix] = stress(solution)[i, j]
                        strain_name = "e" + suffix if i == j else "gamma_" + suffix
                        expressions[strain_name] = (1 if i == j else 2) * ufl.sym(
                            ufl.grad(solution)
                        )[i, j]
                    for field_name, expression in expressions.items():
                        field = fem.Function(field_space)
                        field.name = field_name
                        points = field_space.element.interpolation_points
                        field.interpolate(
                            fem.Expression(expression, points() if callable(points) else points)
                        )
                        xdmf.write_function(field)
        if dim == 2:
            Eeq, nueq, anisotropy = isotropic_fit(C, hypothesis)
            theory = constitutive(1e6, 0.3, hypothesis)
        else:
            from vem3d.materials import C_isotropica_3D, ajustar_E_nu, anisotropia

            Eeq, nueq = ajustar_E_nu(C)
            anisotropy = anisotropia(C)
            theory = C_isotropica_3D(1e6, 0.3)
        error_C = (
            float(np.max(np.abs(C - theory)) / np.max(np.abs(theory)))
            if mode == "homogeneo"
            else None
        )
        dx_tag = ufl.Measure("dx", domain=msh, subdomain_data=tags)
        fractions = {
            str(code): float(
                sum(
                    fem.assemble_scalar(
                        fem.form(fem.Constant(msh, PETSc.ScalarType(1)) * dx_tag(tag))
                    )
                    for tag, mat in mapping.items()
                    if mat == code
                )
                / volume
            )
            for code in set(mapping.values())
        }
        return dict(
            ok=error_C is None or error_C <= tolerance,
            method="FEM-P1",
            dimension=dim,
            dolfinx=dolfinx.__version__,
            C=C.tolist(),
            C_fuerzas=forces.tolist(),
            E_eq=Eeq,
            nu_eq=nueq,
            anisotropia=anisotropy,
            n_dofs=ndof,
            n_elementos=msh.topology.index_map(dim).size_global,
            measure=volume,
            bbox_measure=bbox,
            material_fractions=fractions,
            physical_materials={
                str(tag): dict(name=names[tag], material=code) for tag, code in mapping.items()
            },
            solver=solver,
            solver_res_rel=max(v["residual"] for v in results.values()),
            err_C=error_C,
            err_patch=max(v["affine_error"] for v in results.values()),
            dif_vias=float(np.max(np.abs(C - forces)) / np.max(np.abs(C))),
            cargas=results,
            tiempo_s=time.perf_counter() - start,
        )
    finally:
        for resource in reversed(resources):
            resource.destroy()
