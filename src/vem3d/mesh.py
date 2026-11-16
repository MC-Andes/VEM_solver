"""VEM 3D: mesh."""

import os

import numpy as np


def _buscar_archivo(carpeta, *patrones):
    """Busca un archivo por nombre sin distinguir mayúsculas."""
    existentes = os.listdir(carpeta)
    bajos = {n.lower(): n for n in existentes}
    for pat in patrones:
        if pat.lower() in bajos:
            return os.path.join(carpeta, bajos[pat.lower()])
    raise FileNotFoundError(f"No se encontró ninguno de {patrones} en {carpeta}")


def leer_nodos(archivo):
    """
    Columnas: gdlx gdly gdlz  x y z  [ux ux_fix uy uy_fix uz uz_fix]

    Solo se usan las 6 primeras; las de condiciones de borde se regeneran
    por caso de carga, así que se completan con ceros si faltan.
    """
    nodos = np.loadtxt(archivo)
    if nodos.ndim == 1:
        nodos = nodos.reshape(1, -1)
    if nodos.shape[1] < 6:
        raise ValueError(
            f"{archivo}: se esperaban al menos 6 columnas "
            f"(gdlx gdly gdlz x y z), se leyeron {nodos.shape[1]}"
        )
    if nodos.shape[1] < 12:
        faltan = 12 - nodos.shape[1]
        nodos = np.hstack([nodos, np.zeros((len(nodos), faltan))])
    return nodos


def leer_aristas(archivo):
    """Columnas: id  nodo_i  nodo_j"""
    aristas = np.loadtxt(archivo).astype(int)
    if aristas.ndim == 1:
        aristas = aristas.reshape(1, -1)
    if aristas.shape[1] < 3:
        raise ValueError(f"{archivo}: se esperaban 3 columnas (id nodo_i nodo_j)")
    return aristas


def leer_caras(archivo):
    """Cada línea: id  arista_1 arista_2 ...  -> lista de listas de int."""
    caras = []
    with open(archivo) as f:
        for linea in f:
            linea = linea.strip()
            if not linea:
                continue
            caras.append([int(v) for v in linea.replace(",", " ").split()])
    return caras


def leer_elementos(archivo):
    """
    Acepta los dos formatos del generador:
      id \t mat \t mat2 \t c0,c1,c2,...      (tabs + comas)
      id   mat   mat2   c0 c1 c2 ...         (espacios)

    Devuelve (elementos, codigos) donde `elementos` es una lista de listas
    [id, mat, mat2, cara_0, cara_1, ...] con mat/mat2 aún sin convertir a
    (E, nu), y `codigos` guarda el código de material original porque la
    columna 1 queda sobreescrita con E.

    La tercera columna (mat2) no se usa en la formulación; se conserva solo
    para no alterar el desplazamiento de índices (elemento[3:] = caras).
    """
    elementos, codigos = [], []
    with open(archivo) as f:
        for linea in f:
            linea = linea.strip()
            if not linea:
                continue
            if "," in linea:
                partes = linea.replace("\t", " ").split()
                elem_id = int(partes[0])
                mat_id = int(partes[1])
                mat_id2 = int(partes[2])
                caras_e = [int(c) for c in partes[3].split(",") if c != ""]
            else:
                v = [int(t) for t in linea.split()]
                elem_id, mat_id, mat_id2 = v[0], v[1], v[2]
                caras_e = v[3:]
            if not caras_e:
                raise ValueError(f"{archivo}: el elemento {elem_id} no tiene caras")
            elementos.append([elem_id, mat_id, mat_id2] + caras_e)
            codigos.append(mat_id)
    return elementos, np.array(codigos, dtype=int)


def leer_propiedades(archivo):
    """Formato por línea: material_id  E  nu   ->   {id: (E, nu)}"""
    datos = np.loadtxt(archivo)
    if datos.ndim == 1:
        datos = datos.reshape(1, -1)
    if datos.shape[1] < 3:
        raise ValueError(f"{archivo}: se esperaban 3 columnas (material_id E nu)")
    return {int(f[0]): (float(f[1]), float(f[2])) for f in datos}


def asignar_propiedades(elementos, propiedades, codigos):
    """Asigna (E, nu) a cada elemento según su código de material."""
    for i in range(len(elementos)):
        mat = int(codigos[i])
        if mat not in propiedades:
            raise ValueError(
                f"Elemento {i} (id {elementos[i][0]}): el material {mat} "
                f"no está en Propiedades.txt "
                f"(disponibles: {sorted(propiedades)})"
            )
        E, nu = propiedades[mat]
        elementos[i][1] = E
        elementos[i][2] = nu


def asignar_propiedades_uniformes(elementos, E, nu):
    """Caso homogéneo: mismas propiedades para todos los elementos."""
    for i in range(len(elementos)):
        elementos[i][1] = E
        elementos[i][2] = nu


def _tiene_nodos(carpeta):
    try:
        return any(n.lower().startswith("input_nodos") for n in os.listdir(carpeta))
    except OSError:
        return False


def tamano_malla(carpeta_vem):
    """
    Numero de nodos y elementos de una malla SIN leerla entera: basta con
    contar las lineas no vacias de los .txt. Sirve para dimensionar la
    corrida (gdl = 3*nodos) y elegir --procesos antes de empezar.
    Devuelve (n_nodos, n_elementos), con -1 donde no se pudo contar.
    """

    def _contar(*nombres):
        for n in nombres:
            ruta = os.path.join(carpeta_vem, n)
            if not os.path.isfile(ruta):
                continue
            try:
                with open(ruta, "rb") as f:
                    return sum(1 for linea in f if linea.strip())
            except OSError:
                return -1
        return -1

    return (
        _contar("Input_nodos.txt", "input_nodos.txt"),
        _contar("input_elements.txt", "Input_elements.txt", "input_elementos.txt"),
    )


def descubrir_mallas(raiz_3d):
    """
    Recorre  <raiz>/<CASO>/<CASO_R##>/VEM/  y devuelve una lista de
    (caso, replica, nombre_malla, carpeta_vem) ordenada.

    Si una réplica no tiene subcarpeta VEM pero sí los archivos de entrada
    directamente, se acepta esa carpeta.
    """
    mallas = []
    for caso in sorted(os.listdir(raiz_3d)):
        if caso.lower().startswith("resultados"):
            continue
        dir_caso = os.path.join(raiz_3d, caso)
        if not os.path.isdir(dir_caso):
            continue
        for rep in sorted(os.listdir(dir_caso)):
            dir_rep = os.path.join(dir_caso, rep)
            if not os.path.isdir(dir_rep):
                continue
            dir_vem = os.path.join(dir_rep, "VEM")
            if not os.path.isdir(dir_vem):
                if _tiene_nodos(dir_rep):
                    dir_vem = dir_rep
                else:
                    continue
            elif not _tiene_nodos(dir_vem):
                continue
            # "R01" a partir de "CA1_A2_G1_R01"
            etiqueta = rep.split("_")[-1] if "_" in rep else rep
            mallas.append((caso, etiqueta, f"{caso}/{etiqueta}", dir_vem))
    return mallas
