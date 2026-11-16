"""Genera una malla de cubos para el patch test; ejecutar en Hypatia."""

import argparse
import itertools
from pathlib import Path

import numpy as np


def build_mesh(n=2, interleaved=False):
    coords = list(itertools.product(range(n + 1), repeat=3))
    lookup = {c: i for i, c in enumerate(coords)}
    nn = len(coords)
    dofs = np.arange(3 * nn).reshape(nn, 3) if interleaved else np.arange(3 * nn).reshape(3, nn).T
    nodes = np.column_stack([dofs, np.array(coords) / n, np.zeros((nn, 6))])
    edges, faces, elements = [], [], []
    edge_ids, face_ids = {}, {}
    for i, j, k in itertools.product(range(n), repeat=3):
        vertices = [
            (i, j, k),
            (i + 1, j, k),
            (i + 1, j + 1, k),
            (i, j + 1, k),
            (i, j, k + 1),
            (i + 1, j, k + 1),
            (i + 1, j + 1, k + 1),
            (i, j + 1, k + 1),
        ]
        ids = [lookup[v] for v in vertices]
        local_faces = [
            (0, 3, 2, 1),
            (4, 5, 6, 7),
            (0, 1, 5, 4),
            (1, 2, 6, 5),
            (2, 3, 7, 6),
            (3, 0, 4, 7),
        ]
        face_numbers = []
        for f in local_faces:
            v = [ids[q] for q in f]
            key = tuple(sorted(v))
            if key not in face_ids:
                ee = []
                for a, b in zip(v, v[1:] + v[:1]):
                    e = tuple(sorted((a, b)))
                    if e not in edge_ids:
                        edge_ids[e] = len(edges)
                        edges.append([len(edges), a, b])
                    ee.append(edge_ids[e])
                face_ids[key] = len(faces)
                faces.append([len(faces)] + ee)
            face_numbers.append(face_ids[key])
        elements.append([len(elements), 1e6, 0.3] + face_numbers)
    return nodes, np.array(edges), faces, elements


def write_mesh(path, n=2, heterogeneous=False, interleaved=False):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    nodes, edges, faces, elements = build_mesh(n=n, interleaved=interleaved)
    np.savetxt(path / "Input_nodos.txt", nodes, fmt="%.17g")
    np.savetxt(path / "Input_aristas.txt", edges, fmt="%d")
    (path / "Input_caras.txt").write_text("\n".join(" ".join(map(str, f)) for f in faces) + "\n")
    records = []
    for i, e in enumerate(elements):
        code = int(heterogeneous and i % 2 == 1)
        records.append(f"{i}\t{code}\t0\t" + ",".join(map(str, e[3:])))
    (path / "input_elements.txt").write_text("\n".join(records) + "\n")
    (path / "Propiedades.txt").write_text("0 1000000 0.3\n1 2000000 0.25\n")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--n", type=int, default=2)
    parser.add_argument("--heterogeneous", action="store_true")
    args = parser.parse_args()
    if args.n < 1:
        parser.error("--n debe ser al menos 1")
    write_mesh(args.output, args.n, args.heterogeneous)


def prism_mesh(sides=16):
    """Un prisma regular con 2*sides vértices: ejercita elementos de muchas caras."""
    theta = np.linspace(0, 2 * np.pi, sides, endpoint=False)
    xy = np.column_stack((np.cos(theta), np.sin(theta)))
    xyz = np.vstack((np.column_stack((xy, np.zeros(sides))), np.column_stack((xy, np.ones(sides)))))
    nn = len(xyz)
    nodes = np.column_stack((np.arange(3 * nn).reshape(3, nn).T, xyz, np.zeros((nn, 6))))
    polygons = [list(range(sides - 1, -1, -1)), list(range(sides, 2 * sides))]
    polygons += [[i, (i + 1) % sides, (i + 1) % sides + sides, i + sides] for i in range(sides)]
    edges, faces, lookup = [], [], {}
    for polygon in polygons:
        ids = []
        for a, b in zip(polygon, polygon[1:] + polygon[:1]):
            key = tuple(sorted((a, b)))
            if key not in lookup:
                lookup[key] = len(edges)
                edges.append([len(edges), a, b])
            ids.append(lookup[key])
        faces.append([len(faces)] + ids)
    return nodes, np.asarray(edges), faces, [[0, 1e6, 0.3] + list(range(len(faces)))]
