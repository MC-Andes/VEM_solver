"""Mallas Gmsh físicas; asignación explícita entre etiquetas y materiales."""

import re
from pathlib import Path

from vem2d.materials import constitutive
from vem2d.mesh import _rows


def read_materials(path):
    properties = {}
    for _, row in _rows(Path(path)):
        if len(row) != 3:
            raise ValueError("Propiedades.txt requiere material E nu")
        code = int(row[0])
        if code < 0 or code in properties:
            raise ValueError("Código de material negativo o repetido")
        E, nu = map(float, row[1:])
        constitutive(E, nu)
        properties[code] = (E, nu)
    if not properties:
        raise ValueError("Archivo de propiedades vacío")
    return properties


def load_mesh(path, dimension, properties, explicit_map=None):
    import gmsh
    from mpi4py import MPI

    try:
        from dolfinx.io import gmsh as gmshio
    except ImportError:
        from dolfinx.io import gmshio

    if MPI.COMM_WORLD.size != 1:
        raise ValueError("Esta versión de fem-run usa un solo proceso MPI")
    if explicit_map is not None and (
        not isinstance(explicit_map, dict)
        or any(type(code) is not int or code < 0 for code in explicit_map.values())
    ):
        raise ValueError(
            "El mapa debe ser un objeto JSON con códigos de material enteros no negativos"
        )
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.open(str(path))
        if gmsh.model.getDimension() != dimension:
            raise ValueError("La dimensión solicitada no coincide con la malla Gmsh")
        materials, names, covered = {}, {}, []
        for _, tag in gmsh.model.getPhysicalGroups(dimension):
            name = gmsh.model.getPhysicalName(dimension, tag)
            match = re.fullmatch(r"(?:matriz|agregado)_tipo_(\d+)", name)
            if explicit_map is not None:
                if str(tag) not in explicit_map:
                    raise ValueError(f"Falta la etiqueta física {tag} en el mapa de materiales")
                code = int(explicit_map[str(tag)])
            elif match:
                code = int(match.group(1))
            else:
                raise ValueError(
                    f"Etiqueta {tag} ({name!r}) sin material reconocido: usa --mapa-materiales"
                )
            if code not in properties:
                raise ValueError(
                    f"El material {code} de la etiqueta {tag} no existe en Propiedades.txt"
                )
            materials[tag], names[tag] = code, name
            covered.extend(gmsh.model.getEntitiesForPhysicalGroup(dimension, tag).tolist())
        entities = [tag for _, tag in gmsh.model.getEntities(dimension)]
        if sorted(covered) != sorted(entities):
            raise ValueError(
                "Cada entidad del volumen/área debe tener exactamente un grupo físico material"
            )
        if not entities:
            raise ValueError("Malla vacía")
        mesh_data = gmshio.model_to_mesh(gmsh.model, MPI.COMM_SELF, 0, gdim=dimension)
        if hasattr(mesh_data, "mesh"):
            mesh, tags = mesh_data.mesh, mesh_data.cell_tags
        else:
            mesh, tags, _ = mesh_data
        if tags is None or len(tags.indices) != mesh.topology.index_map(dimension).size_local:
            raise ValueError("Hay celdas FEM sin etiqueta material")
        return mesh, tags, materials, names
    finally:
        gmsh.finalize()
