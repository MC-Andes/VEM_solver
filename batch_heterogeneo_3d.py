"""Entrada compatible; preferir el comando instalado vem3d-batch-heterogeneous."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from vem3d.cli.batch_heterogeneous import main

if __name__ == "__main__":
    sys.exit(main())
