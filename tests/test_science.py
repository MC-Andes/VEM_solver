import importlib.util
from pathlib import Path

import numpy as np
import pytest

from vem3d import assembly, boundary, geometry, homogenization, materials, solvers


@pytest.mark.parametrize("solver", ["lu", "amg"])
@pytest.mark.parametrize("interleaved", [False, True])
def test_six_affine_patch_cases(cube_mesh, tmp_path, solver, interleaved):
    nodes, edges, faces, elems = cube_mesh(interleaved=interleaved)
    face_info = geometry.construir_infocaras(nodes, edges, faces)
    border = boundary.detectar_borde(nodes)
    info = {}
    part = assembly.ensamblar_particionado(
        nodes, edges, faces, elems, face_info, info, border, base_tmp=tmp_path
    )
    system = solvers.factorizar_particionado(part, solver=solver)
    try:
        C, Cf, results, _ = homogenization.homogenizar(nodes, elems, info, system, border, 1.0)
        exact = materials.C_isotropica_3D(1e6, 0.3)
        np.testing.assert_allclose(C, exact, atol=1e-8, rtol=1e-12)
        np.testing.assert_allclose(Cf, exact, atol=1e-8, rtol=1e-12)
        assert max(r["err_rel"] for r in results.values()) < 1e-12
        assert sum(e["V"] for e in info.values()) == pytest.approx(1.0)
        E, nu = materials.ajustar_E_nu(C)
        assert E == pytest.approx(1e6)
        assert nu == pytest.approx(0.3)
    finally:
        assembly.liberar_sistema(system)
        assembly.liberar_sistema(part)


@pytest.mark.parametrize("heterogeneous", [False, True])
def test_original_formulation_equivalence(cube_mesh, heterogeneous):
    path = Path(__file__).parents[1] / "reference/original/vem3d_core.py"
    spec = importlib.util.spec_from_file_location("original_core", path)
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    nodes, edges, faces, elems = cube_mesh()
    if heterogeneous:
        for i, elem in enumerate(elems):
            elem[1:3] = [1e6 * (1 + i % 2), 0.25 + 0.05 * (i % 2)]
    old_info, new_info = {}, {}
    old_faces = original.construir_infocaras(nodes, edges, faces)
    new_faces = geometry.construir_infocaras(nodes, edges, faces)
    old = original.ensamblar(nodes, edges, faces, elems, old_faces, old_info)
    new = assembly.ensamblar(nodes, edges, faces, elems, new_faces, new_info)
    np.testing.assert_allclose(old.toarray(), new.toarray(), rtol=1e-13, atol=1e-9)
    for e in new_info:
        for key in ("V", "dpi", "piast", "C"):
            np.testing.assert_allclose(new_info[e][key], old_info[e][key], rtol=1e-13, atol=1e-10)


def test_unconverged_amg_raises():
    import scipy.sparse as sp

    class NotConverged:
        def solve(self, b, **kwargs):
            return np.zeros_like(b)

    system = {
        "ndof": 2,
        "idx_f": np.array([0]),
        "idx_l": np.array([1]),
        "K_lf": sp.csr_matrix([[1.0]]),
        "K_ll": sp.eye(1, format="csr"),
        "solver": "amg",
        "op": NotConverged(),
        "iters": [],
        "res_rel": [],
    }
    with pytest.raises(RuntimeError, match="no convergió"):
        solvers.resolver(system, np.ones(1))


@pytest.mark.parametrize("sides", [3, 8, 24])
def test_optimized_prism_matches_original(sides):
    from examples.cube import prism_mesh
    from vem3d.element import elemento_VEM3D

    path = Path(__file__).parents[1] / "reference/original/vem3d_core.py"
    spec = importlib.util.spec_from_file_location("original_prism", path)
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    nodes, edges, faces, elems = prism_mesh(sides)
    # Translation and skew test invariance beyond the axis-aligned cube.
    nodes[:, 3:6] = nodes[:, 3:6] @ np.array([[1, 0.2, 0.1], [0, 0.8, 0.15], [0, 0, 1.3]]) + [
        2,
        -1,
        0.7,
    ]
    old_faces = original.construir_infocaras(nodes, edges, faces)
    new_faces = geometry.construir_infocaras(nodes, edges, faces)
    old, dofs = original.elemento_VEM3D(0, nodes, edges, faces, elems, old_faces, {})
    new, _ = elemento_VEM3D(0, nodes, edges, faces, elems, new_faces, {})
    assert np.max(np.abs(new - old)) / np.max(np.abs(old)) < 1e-12
    np.testing.assert_allclose(new, new.T, rtol=1e-12, atol=1e-8)
    eigenvalues = np.linalg.eigvalsh(new)
    assert np.max(np.abs(eigenvalues[:6])) < eigenvalues[-1] * 1e-12
    assert eigenvalues[6] > eigenvalues[-1] * 1e-6


def test_old_amg_reports_int64_requirement(monkeypatch):
    import scipy.sparse as sp

    matrix = sp.eye(2, format="csr")
    matrix.indices = matrix.indices.astype(np.int64)
    matrix.indptr = matrix.indptr.astype(np.int64)
    monkeypatch.setattr(solvers.pyamg, "__version__", "5.3.0")
    with pytest.raises(RuntimeError, match="pyamg>=5.4.1"):
        solvers._validar_indices_amg(matrix)


@pytest.mark.parametrize("heterogeneous", [False, True])
def test_affine_start_matches_zero_start(cube_mesh, tmp_path, monkeypatch, heterogeneous):
    monkeypatch.setattr(solvers, "AMG_MAX_COARSE", 2)
    nodes, edges, faces, elems = cube_mesh(n=3)
    if heterogeneous:
        for i, elem in enumerate(elems):
            elem[1:3] = [1e6 * (1 + i % 3), 0.2 + 0.05 * (i % 3)]
    face_info = geometry.construir_infocaras(nodes, edges, faces)
    border = boundary.detectar_borde(nodes)
    info = {}
    part = assembly.ensamblar_particionado(
        nodes, edges, faces, elems, face_info, info, border, base_tmp=tmp_path
    )
    outputs = []
    for affine in (False, True):
        system = solvers.factorizar_particionado(part, solver="amg")
        outputs.append(
            homogenization.homogenizar(nodes, elems, info, system, border, 1.0, inicio_afin=affine)
        )
        assert max(system["res_rel"]) <= solvers.TOL_AMG
        if affine and not heterogeneous:
            assert system["iters"] == [0] * 6
    for i in (0, 1):
        np.testing.assert_allclose(outputs[0][i], outputs[1][i], rtol=1e-10, atol=1e-5)
    for load in outputs[0][2]:
        np.testing.assert_allclose(
            outputs[0][2][load]["U"], outputs[1][2][load]["U"], rtol=1e-10, atol=1e-11
        )
    assembly.liberar_sistema(system)
    assembly.liberar_sistema(part)


def test_zero_rhs_ignores_nonzero_initial_guess():
    import scipy.sparse as sp

    system = {
        "ndof": 2,
        "idx_f": np.array([0]),
        "idx_l": np.array([1]),
        "K_lf": sp.csr_matrix((1, 1)),
        "K_ll": sp.eye(1, format="csr"),
        "solver": "amg",
        "op": None,
        "iters": [],
        "res_rel": [],
    }
    np.testing.assert_array_equal(
        solvers.resolver(system, np.array([2.0]), x0=np.ones(1)), [2.0, 0.0]
    )
    assert system["res_rel"] == [0.0]
