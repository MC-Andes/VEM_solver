"""Alias instalado para scripts que importan vem3d_core."""

import sys

from vem3d import compat

sys.modules[__name__] = compat
