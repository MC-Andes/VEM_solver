"""Genera una malla 2D pequeña para comprobar la instalación."""

import argparse
from pathlib import Path

import numpy as np


def write_mesh(directory, n=2, heterogeneous=False):
    if n < 1:
        raise ValueError("n debe ser al menos 1")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    xy = np.array([(i / n, j / n) for j in range(n + 1) for i in range(n + 1)])
    nn = len(xy)
    np.savetxt(
        directory / "Input_nodos.txt",
        np.column_stack((np.arange(nn), np.arange(nn, 2 * nn), xy)),
        fmt="%.17g",
    )
    records = []
    for j in range(n):
        for i in range(n):
            a = j * (n + 1) + i
            vertices = (a, a + 1, a + n + 2, a + n + 1)
            material = int(heterogeneous and i >= n / 2)
            records.append(f"{len(records)} {material} 0 " + ",".join(map(str, vertices)))
    (directory / "input_elements.txt").write_text("\n".join(records) + "\n")
    (directory / "Propiedades.txt").write_text("0 1000000 0.3\n1 2000000 0.25\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--n", type=int, default=2)
    parser.add_argument("--heterogeneous", action="store_true")
    args = parser.parse_args()
    if args.n < 1:
        parser.error("--n debe ser al menos 1")
    write_mesh(args.output, args.n, args.heterogeneous)
