"""Lectura del formato 2D de MC-Andes: nodos, polígonos y materiales."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .geometry import polygon_geometry
from .materials import constitutive

INPUTS = ("Input_nodos.txt", "input_elements.txt", "Propiedades.txt")


@dataclass
class Mesh:
    coordinates: np.ndarray
    dofs: np.ndarray
    cells: list
    codes: np.ndarray
    properties: dict
    boundary: np.ndarray
    area: float
    input_paths: tuple


def _rows(path):
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            line = line.split("#", 1)[0].strip()
            if line:
                yield number, line.replace(",", " ").split()


def read_mesh(directory):
    directory = Path(directory)
    names = {p.name.lower(): p for p in directory.iterdir() if p.is_file()}
    try:
        paths = tuple(names[name.lower()] for name in INPUTS)
    except KeyError as error:
        raise ValueError(f"Falta un archivo de entrada 2D: {error.args[0]}") from error
    rows = [values for _, values in _rows(paths[0])]
    if not rows or any(len(row) < 4 for row in rows):
        raise ValueError("Input_nodos.txt requiere gdl_x gdl_y x y por fila")
    nodes = np.array([row[:4] for row in rows], dtype=float)
    if not np.isfinite(nodes).all():
        raise ValueError("Nodos con valores no finitos")
    if not np.array_equal(np.sort(nodes[:, :2].ravel()), np.arange(2 * len(nodes))):
        raise ValueError("Los GDL deben ser una permutación de 0 a 2*N-1")
    coordinates, dofs = nodes[:, 2:4], nodes[:, :2].astype(np.int64)
    if len(np.unique(coordinates, axis=0)) != len(coordinates):
        raise ValueError("Hay nodos con coordenadas duplicadas")
    properties = {}
    for number, row in _rows(paths[2]):
        if len(row) != 3:
            raise ValueError(f"Propiedades.txt:{number}: se requiere material E nu")
        code = int(row[0])
        if code in properties or code < 0:
            raise ValueError("Código de material negativo o duplicado")
        E, nu = map(float, row[1:])
        constitutive(E, nu)
        properties[code] = (E, nu)
    cells, codes, edges = [], [], {}
    used = np.zeros(len(nodes), dtype=bool)
    total_area = 0.0
    for number, row in _rows(paths[1]):
        values = list(map(int, row))
        if len(values) < 6 or values[0] != len(cells):
            raise ValueError(
                f"input_elements.txt:{number}: id consecutivo y al menos tres nodos requeridos"
            )
        ids = np.array(values[3:], dtype=np.int64)
        if len(set(ids)) != len(ids) or ids.min() < 0 or ids.max() >= len(nodes):
            raise ValueError(f"Elemento {values[0]}: nodos repetidos o fuera de rango")
        if values[1] not in properties:
            raise ValueError(f"Elemento {values[0]}: material {values[1]} no definido")
        try:
            area, _, ccw = polygon_geometry(coordinates[ids])
        except ValueError as error:
            raise ValueError(f"Elemento {values[0]}: {error}") from error
        if not ccw:
            ids = ids[::-1]
        for a, b in zip(ids, np.roll(ids, -1)):
            key = (min(a, b), max(a, b))
            direction = 1 if a < b else -1
            count, balance = edges.get(key, (0, 0))
            edges[key] = (count + 1, balance + direction)
        cells.append(ids)
        codes.append(values[1])
        total_area += area
        used[ids] = True
    if not cells or not used.all():
        raise ValueError("Malla vacía o con nodos sin elementos")
    low, high = coordinates.min(axis=0), coordinates.max(axis=0)
    lengths = high - low
    if np.any(lengths <= 0):
        raise ValueError("El dominio debe tener área positiva")
    bbox = float(np.prod(lengths))
    if abs(total_area - bbox) / bbox > 1e-8:
        raise ValueError("El área de los elementos no coincide con el rectángulo exterior")
    boundary = np.zeros(len(nodes), dtype=bool)
    normalized = (coordinates - low) / lengths
    for (a, b), (count, balance) in edges.items():
        if count == 2 and balance == 0:
            continue
        if count != 1:
            raise ValueError("Arista no conforme, solapada o compartida por más de dos elementos")
        same_side = np.any(np.all(np.abs(normalized[[a, b]]) < 1e-9, axis=0)) or np.any(
            np.all(np.abs(normalized[[a, b]] - 1) < 1e-9, axis=0)
        )
        if not same_side:
            raise ValueError(
                "Borde interior o malla no conforme: se requiere un rectángulo sin huecos"
            )
        boundary[[a, b]] = True
    return Mesh(coordinates, dofs, cells, np.array(codes), properties, boundary, bbox, paths)
