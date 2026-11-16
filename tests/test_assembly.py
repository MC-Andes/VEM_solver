import numpy as np
import pytest

from vem3d import assembly, boundary, geometry


@pytest.mark.parametrize("offset", [0, 2**31 - 2, 2**31 - 1, 2**31, 3_000_000_000])
def test_csr_offsets_cross_int32(offset):
    local = np.array([0, 1, 5], dtype=np.int32)
    np.testing.assert_array_equal(
        assembly.csr_offset(local, offset), [offset, offset + 1, offset + 5]
    )
    assert assembly.csr_offset(local, offset).dtype == np.int64


def test_csr_offset_refuses_int64_overflow():
    with pytest.raises(OverflowError):
        assembly.csr_offset(np.array([0, 2], dtype=np.int64), 2**63 - 2)


@pytest.mark.parametrize("disk", [False, True])
@pytest.mark.parametrize("n", [1, 2])
def test_partition_matches_global(cube_mesh, monkeypatch, tmp_path, disk, n):
    nodes, edges, faces, elems = cube_mesh(n)
    face_info = geometry.construir_infocaras(nodes, edges, faces)
    full = assembly.ensamblar(nodes, edges, faces, elems, face_info, {}, max_entradas=300)
    monkeypatch.setattr(assembly, "FILAS_POR_FRANJA", 1)
    monkeypatch.setattr(assembly, "ENTRADAS_BUFFER", 8)
    part = assembly.ensamblar_particionado(
        nodes,
        edges,
        faces,
        elems,
        face_info,
        {},
        boundary.detectar_borde(nodes),
        base_tmp=tmp_path,
        en_disco=disk,
    )
    try:
        ll, ff = part["idx_l"], part["idx_f"]
        np.testing.assert_allclose(part["K_ll"].toarray(), full[ll][:, ll].toarray(), atol=1e-9)
        np.testing.assert_allclose(part["K_lf"].toarray(), full[ll][:, ff].toarray(), atol=1e-9)
        np.testing.assert_allclose(part["K_f"].toarray(), full[ff].toarray(), atol=1e-9)
    finally:
        assembly.liberar_sistema(part)
    assert not list(tmp_path.iterdir())


def test_cleanup_on_element_failure(cube_mesh, monkeypatch, tmp_path):
    nodes, edges, faces, elems = cube_mesh()

    def broken(*args):
        raise RuntimeError("injected failure")

    monkeypatch.setattr(assembly, "elemento_VEM3D", broken)
    with pytest.raises(RuntimeError, match="injected"):
        assembly.ensamblar_particionado(
            nodes, edges, faces, elems, [], {}, boundary.detectar_borde(nodes), base_tmp=tmp_path
        )
    assert not list(tmp_path.iterdir())
