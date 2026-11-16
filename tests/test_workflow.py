import json

import numpy as np
import pytest

from vem3d import assembly, boundary, geometry, homogenization, solvers, workflow
from vem3d.cli.single import main


@pytest.mark.parametrize("mode", ["homogeneo", "heterogeneo"])
@pytest.mark.parametrize("fields", [False, True])
def test_workflow_and_results(mesh_files, tmp_path, mode, fields):
    output = tmp_path / f"{mode}-{fields}"
    r = workflow.correr_malla(
        str(mesh_files),
        str(output),
        "CASE/R01",
        mode,
        solver="amg",
        guardar_campos=fields,
        base_tmp=tmp_path / "scratch",
    )
    assert r["ok"], r.get("traceback")
    assert r["meta"]["dif_vias"] < 1e-10
    assert r["meta"]["dif_volumen"] < 1e-12
    assert (output / "C_equivalente.txt").exists()
    assert bool(list(output.glob("U_*.txt"))) == fields
    assert not list((tmp_path / "scratch").iterdir())
    if mode == "homogeneo":
        assert r["meta"]["verif_ok"]
    else:
        assert 1e6 < r["meta"]["E_eq"] < 2e6
        assert np.min(np.linalg.eigvalsh((r["C"] + r["C"].T) / 2)) > 0


def test_cli_manifest_and_failure(mesh_files, tmp_path):
    out = tmp_path / "cli"
    assert (
        main(
            [
                str(mesh_files),
                "--modo",
                "homogeneo",
                "--solver",
                "lu",
                "--sin-campos",
                "--salida",
                str(out),
            ]
        )
        == 0
    )
    result = json.loads((out / "run.json").read_text())
    assert len(result["inputs"]) == 5
    assert result["result"]["meta"]["verif_ok"]
    assert main([str(tmp_path / "missing"), "--salida", str(tmp_path / "fail")]) == 1
    assert (tmp_path / "fail/ERROR.txt").exists()


def test_compact_info_and_reduced_fields_equal_full(cube_mesh, tmp_path):
    nodes, edges, faces, elems = cube_mesh()
    face_info = geometry.construir_infocaras(nodes, edges, faces)
    border = boundary.detectar_borde(nodes)
    outputs, sizes = [], []
    for compact in [False, True]:
        info = {}
        part = assembly.ensamblar_particionado(
            nodes,
            edges,
            faces,
            elems,
            face_info,
            info,
            border,
            base_tmp=tmp_path,
            compact_info=compact,
        )
        system = solvers.factorizar_particionado(part, solver="lu")
        try:
            sizes.append(
                sum(
                    v.nbytes for r in info.values() for v in r.values() if isinstance(v, np.ndarray)
                )
            )
            outputs.append(
                homogenization.homogenizar(
                    nodes, elems, info, system, border, 1.0, guardar_campos=not compact
                )
            )
        finally:
            assembly.liberar_sistema(system)
            assembly.liberar_sistema(part)
    assert sizes[1] < sizes[0]
    for col in (0, 1):
        np.testing.assert_allclose(outputs[0][col], outputs[1][col], atol=1e-9)
    assert homogenization._estadisticas(outputs[0][2]) == homogenization._estadisticas(
        outputs[1][2]
    )
    assert all("U" not in r and "sig" not in r for r in outputs[1][2].values())


def test_cleanup_on_solver_failure(mesh_files, tmp_path, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("injected setup failure")

    monkeypatch.setattr(workflow, "factorizar_particionado", broken)
    result = workflow.correr_malla(
        str(mesh_files),
        str(tmp_path / "out"),
        "CASE/R01",
        "homogeneo",
        base_tmp=tmp_path / "scratch",
    )
    assert not result["ok"]
    assert not list((tmp_path / "scratch").iterdir())


@pytest.mark.parametrize("module_name", ["batch_homogeneous", "batch_heterogeneous"])
def test_batch_end_to_end(mesh_files, tmp_path, monkeypatch, module_name):
    import importlib

    module = importlib.import_module(f"vem3d.cli.{module_name}")
    monkeypatch.setattr(
        "sys.argv",
        [
            "batch",
            "--raiz",
            str(mesh_files.parents[2]),
            "--solver",
            "lu",
            "--procesos",
            "1",
            "--sin-campos",
        ],
    )
    assert module.main() == 0
    out = mesh_files.parents[2] / module.CARPETA_SALIDA
    assert (out / "RESUMEN_PROPIEDADES.csv").exists()
    assert (out / "RESUMEN_GLOBAL.txt").exists()
