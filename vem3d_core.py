"""Compatibilidad con import vem3d_core; implementación en src/vem3d."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from vem3d import compat

sys.modules[__name__] = compat
