"""Entrada compatible; preferir el comando instalado vem3d-batch-homogeneous."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from vem3d.cli.batch_homogeneous import main

if __name__ == "__main__":
    sys.exit(main())
