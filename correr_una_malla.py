"""Entrada histórica para el comando vem3d-run."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from vem3d.cli.single import main

if __name__ == "__main__":
    sys.exit(main())
