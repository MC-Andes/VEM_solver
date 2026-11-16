"""Comprueba el backend AMG con índices int64 sin reservar una matriz gigante."""

import json

import numpy as np
import pyamg
import scipy
import scipy.sparse as sp

from vem3d.assembly import csr_offset

A = sp.diags([-np.ones(31), 2 * np.ones(32), -np.ones(31)], [-1, 0, 1], format="csr")
A.indices = A.indices.astype(np.int64)
A.indptr = A.indptr.astype(np.int64)
report = {
    "numpy": np.__version__,
    "scipy": scipy.__version__,
    "pyamg": pyamg.__version__,
    "input_indices": str(A.indices.dtype),
    "input_indptr": str(A.indptr.dtype),
}
try:
    hierarchy = pyamg.smoothed_aggregation_solver(
        A, max_coarse=2, smooth="jacobi", strength="symmetric"
    )
    u = hierarchy.solve(np.ones(32), accel="cg", tol=1e-12)
    report.update(
        status="accepted",
        relative_residual=float(np.linalg.norm(A @ u - 1) / np.sqrt(32)),
        hierarchy_dtypes=[str(level.A.indptr.dtype) for level in hierarchy.levels],
    )
except Exception as error:
    report.update(status="rejected", error=f"{type(error).__name__}: {error}")
local = np.array([0, 1, 5], dtype=np.int32)
report["legacy_crossing_2pow31"] = (local + (2**31 - 2)).tolist()
try:
    report["legacy_offset_2pow31"] = (local + 2**31).tolist()
except OverflowError as error:
    report["legacy_offset_2pow31"] = f"{type(error).__name__}: {error}"
report["corrected_offset_2pow31"] = csr_offset(local, 2**31).tolist()
print(json.dumps(report, indent=2))
