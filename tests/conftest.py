"""Mallas analíticas reproducibles."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
from examples.cube import build_mesh, write_mesh


@pytest.fixture
def cube_mesh():
    return build_mesh


@pytest.fixture
def mesh_files(tmp_path):
    return write_mesh(tmp_path / "CASE/CASE_R01/VEM", heterogeneous=True)
