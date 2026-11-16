"""Ensamblaje disperso 2D; tripletas NumPy preasignadas, sin listas de escalares."""

import numpy as np
from scipy.sparse import coo_matrix

from vem3d.solvers import factorizar_particionado

from .element import element_stiffness
from .materials import constitutive


def assemble(mesh, mode, hypothesis):
    if mode not in ("homogeneo", "heterogeneo"):
        raise ValueError(f"Modo desconocido: {mode}")
    ndof = 2 * len(mesh.coordinates)
    count = sum((2 * len(ids)) ** 2 for ids in mesh.cells)
    dtype = np.int64 if max(count, ndof) > np.iinfo(np.int32).max else np.int32
    rows, columns = np.empty(count, dtype=dtype), np.empty(count, dtype=dtype)
    data = np.empty(count)
    materials = {
        code: constitutive(*(props if mode == "heterogeneo" else (1e6, 0.3)), hypothesis)
        for code, props in mesh.properties.items()
    }
    info, offset = [], 0
    for ids, code in zip(mesh.cells, mesh.codes):
        C = materials[code]
        K, B, area = element_stiffness(mesh.coordinates[ids], C)
        dofs = mesh.dofs[ids].T.ravel()
        size = K.size
        rows[offset : offset + size] = np.repeat(dofs, len(dofs))
        columns[offset : offset + size] = np.tile(dofs, len(dofs))
        data[offset : offset + size] = K.ravel()
        offset += size
        info.append((dofs, B, C, area))
    K = coo_matrix((data, (rows, columns)), shape=(ndof, ndof)).tocsr()
    K.eliminate_zeros()
    return K, info


def prepare_system(K, mesh, solver):
    fixed = np.zeros(K.shape[0], dtype=bool)
    fixed[mesh.dofs[mesh.boundary].ravel()] = True
    prescribed, free = np.flatnonzero(fixed), np.flatnonzero(~fixed)
    part = dict(
        K_ll=K[free][:, free].tocsr(),
        K_lf=K[free][:, prescribed].tocsr(),
        K_f=K[prescribed].tocsr(),
        idx_l=free,
        idx_f=prescribed,
        ndof=K.shape[0],
    )
    system = factorizar_particionado(part, solver)
    # También comprobar el residuo de LU; conservar sólo el bloque libre necesario.
    system["K_ll"] = part["K_ll"]
    return system
