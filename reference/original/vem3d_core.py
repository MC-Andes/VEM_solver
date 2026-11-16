# -*- coding: utf-8 -*-
"""
vem3d_core.py
=============
Núcleo compartido para los batch de homogenización VEM 3D. Contiene la
lectura de mallas poliédricas, el solver VEM 3D, el procedimiento de
homogenización y la escritura de resultados.

Lo usan `batch_homogeneo_3d.py` y `batch_heterogeneo_3d.py`; no se ejecuta
solo.

Estructura de carpetas esperada
-------------------------------
    <RAIZ_3D>/
      CA1_A2_G1/
        CA1_A2_G1_R01/VEM/  Input_nodos.txt   Input_aristas.txt
                            Input_caras.txt   input_elements.txt
                            Propiedades.txt
        CA1_A2_G1_R02/VEM/  ...
      CA1_A2_G2/
        ...

Convenciones
------------
* Voigt 3D:  sigma = [sxx, syy, szz, txy, txz, tyz]
             eps   = [exx, eyy, ezz, gxy, gxz, gyz]
* Casos de carga (columnas de C), impuestos como KUBC:
    col 0 -> exx = 1      col 3 -> gxy = 1   (eps_xy tensorial = 1/2)
    col 1 -> eyy = 1      col 4 -> gxz = 1
    col 2 -> ezz = 1      col 5 -> gyz = 1
* El código de material COD_MATRIZ (= 0) es la matriz; cualquier otro
  código se contabiliza como agregado.

Formato de los archivos de entrada
----------------------------------
* Input_nodos.txt   : gdlx gdly gdlz  x y z  [columnas de BC]
                      Las columnas de BC se regeneran por caso de carga.
* Input_aristas.txt : id  nodo_i  nodo_j
* Input_caras.txt   : id  arista_1 arista_2 ...
* input_elements.txt: id \t mat \t mat2 \t cara_1,cara_2,...
                      (también se acepta el formato antiguo separado por
                      espacios). La tercera columna no se usa, igual que
                      en el código 3D verificado.
* Propiedades.txt   : material_id  E  nu

Diferencias respecto al script 3D monolítico anterior
-----------------------------------------------------
Se conserva intacta la formulación VEM (infocara, orientar_cara,
divergencia, monomios3D, elemento_VEM3D). Lo que cambia es:

1. Condiciones de borde impuestas por partición de grados de libertad y
   una única factorización LU reutilizada por los seis casos, en lugar de
   penalización con P = 1e10 * max(diag(K)). Es la misma estrategia del
   núcleo 2D. La imposición por partición es exacta; la penalización
   introduce un error de orden 1/P y degrada el condicionamiento.
   VERIFICACIÓN: en el caso homogéneo el error del campo lineal debe
   bajar a ~1e-14; con penalización quedaba acotado por 1/P.
2. Detección del contorno con tolerancia relativa al tamaño del dominio
   en lugar de igualdad exacta de flotantes.
3. `informacion` guarda solo lo que usa el post-proceso (dpi, piast, gdl,
   C, V, centroide), no las matrices densas por elemento. En 3D esto
   reduce la memoria en un factor grande.
4. Se calcula C por dos vías independientes (promedio volumétrico de
   esfuerzos y reacciones en el contorno) para poder contrastarlas.
5. (E, nu) efectivos se ajustan por mínimos cuadrados sobre las 36
   entradas de C, igual que en 2D. Se reporta además la extracción
   directa C11/C12/C44 que usaba el script anterior, para comparar.
"""

import os
import sys
import csv
import shutil
import tempfile
import time
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu
from scipy.optimize import least_squares

try:
    import pyamg
    HAY_PYAMG = True
except ImportError:
    HAY_PYAMG = False

# Por encima de este número de gdl libres el LU disperso de K_ll deja de
# ser viable en memoria (una malla de ~116 000 gdl libres agotó más de 6 GB)
# y se pasa a AMG, que necesitó 2.8 GB para la misma malla.
UMBRAL_GDL_LU = 40_000

# Tolerancia relativa del residuo para el solver iterativo.
TOL_AMG = 1e-12

# Parametros de la jerarquia AMG. Medido sobre la misma K_ll (34 M no-ceros),
# con identico residuo final (~6e-13) en las cuatro combinaciones:
#
#   smooth / strength        setup    iters   memoria AMG   bytes/nnz
#   energy / evolution       51.4 s      32       0.66 GB        22.3
#   energy / symmetric        5.3 s      47       0.31 GB        10.4
#   jacobi / symmetric        3.1 s      47       0.31 GB        10.5
#   None   / symmetric        1.1 s      47       0.17 GB         5.7
#
# 'evolution' cuesta 22 de los ~80 bytes por no-cero de toda la corrida y
# hace el setup 16 veces mas lento, a cambio de 15 iteraciones menos. Se usa
# 'jacobi'+'symmetric': conserva la agregacion suavizada (que es la que
# escala bien al crecer la malla) con una fraccion de la memoria.
AMG_SMOOTH = "jacobi"
AMG_STRENGTH = "symmetric"
AMG_MAX_COARSE = 500

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTES
# ══════════════════════════════════════════════════════════════════════════════

# Código de material de la matriz. Todo código distinto se cuenta como agregado.
COD_MATRIZ = 0

# Propiedades del caso homogéneo de verificación
E_HOM  = 1.0e6
NU_HOM = 0.3

R_mat = np.array([[0, 1], [-1, 0]])

# (nombre, exx, eyy, ezz, exy, exz, eyz)  -- componentes TENSORIALES
CASOS_CARGA = [
    ("exx=1", 1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    ("eyy=1", 0.0, 1.0, 0.0, 0.0, 0.0, 0.0),
    ("ezz=1", 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
    ("gxy=1", 0.0, 0.0, 0.0, 0.5, 0.0, 0.0),
    ("gxz=1", 0.0, 0.0, 0.0, 0.0, 0.5, 0.0),
    ("gyz=1", 0.0, 0.0, 0.0, 0.0, 0.0, 0.5),
]

COMP_SIGMA = ["sxx", "syy", "szz", "txy", "txz", "tyz"]
COMP_EPS   = ["exx", "eyy", "ezz", "gxy", "gxz", "gyz"]


# ══════════════════════════════════════════════════════════════════════════════
# 1. LECTURA DE ARCHIVOS
# ══════════════════════════════════════════════════════════════════════════════

def _buscar_archivo(carpeta, *patrones):
    """Busca un archivo por nombre sin distinguir mayúsculas."""
    existentes = os.listdir(carpeta)
    bajos = {n.lower(): n for n in existentes}
    for pat in patrones:
        if pat.lower() in bajos:
            return os.path.join(carpeta, bajos[pat.lower()])
    raise FileNotFoundError(
        f"No se encontró ninguno de {patrones} en {carpeta}")


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
            f"(gdlx gdly gdlz x y z), se leyeron {nodos.shape[1]}")
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
        raise ValueError(
            f"{archivo}: se esperaban 3 columnas (id nodo_i nodo_j)")
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
                mat_id  = int(partes[1])
                mat_id2 = int(partes[2])
                caras_e = [int(c) for c in partes[3].split(",") if c != ""]
            else:
                v = [int(t) for t in linea.split()]
                elem_id, mat_id, mat_id2 = v[0], v[1], v[2]
                caras_e = v[3:]
            if not caras_e:
                raise ValueError(
                    f"{archivo}: el elemento {elem_id} no tiene caras")
            elementos.append([elem_id, mat_id, mat_id2] + caras_e)
            codigos.append(mat_id)
    return elementos, np.array(codigos, dtype=int)


def leer_propiedades(archivo):
    """Formato por línea: material_id  E  nu   ->   {id: (E, nu)}"""
    datos = np.loadtxt(archivo)
    if datos.ndim == 1:
        datos = datos.reshape(1, -1)
    if datos.shape[1] < 3:
        raise ValueError(
            f"{archivo}: se esperaban 3 columnas (material_id E nu)")
    return {int(f[0]): (float(f[1]), float(f[2])) for f in datos}


def asignar_propiedades(elementos, propiedades, codigos):
    """Asigna (E, nu) a cada elemento según su código de material."""
    for i in range(len(elementos)):
        mat = int(codigos[i])
        if mat not in propiedades:
            raise ValueError(
                f"Elemento {i} (id {elementos[i][0]}): el material {mat} "
                f"no está en Propiedades.txt "
                f"(disponibles: {sorted(propiedades)})")
        E, nu = propiedades[mat]
        elementos[i][1] = E
        elementos[i][2] = nu


def asignar_propiedades_uniformes(elementos, E, nu):
    """Caso homogéneo: mismas propiedades para todos los elementos."""
    for i in range(len(elementos)):
        elementos[i][1] = E
        elementos[i][2] = nu


# ══════════════════════════════════════════════════════════════════════════════
# 2. CONDICIONES DE BORDE
# ══════════════════════════════════════════════════════════════════════════════

def detectar_borde(nodos):
    """
    Nodos sobre el contorno del dominio prismático.
    La tolerancia es relativa al tamaño del dominio para no depender de que
    el generador escriba exactamente el valor de la cara.
    """
    x, y, z = nodos[:, 3], nodos[:, 4], nodos[:, 5]
    Lx = x.max() - x.min()
    Ly = y.max() - y.min()
    Lz = z.max() - z.min()
    tol = 1e-9 * max(Lx, Ly, Lz)
    return ((x <= x.min() + tol) | (x >= x.max() - tol) |
            (y <= y.min() + tol) | (y >= y.max() - tol) |
            (z <= z.min() + tol) | (z >= z.max() - tol))


def indices_gdl(nodos):
    """Índices globales de los gdl x, y, z de cada nodo (columnas 0,1,2)."""
    return (nodos[:, 0].astype(int),
            nodos[:, 1].astype(int),
            nodos[:, 2].astype(int))


def campo_lineal(nodos, exx, eyy, ezz, exy, exz, eyz):
    """
    Campo de desplazamiento lineal asociado a una deformación uniforme,
    evaluado en TODOS los nodos y devuelto en el orden global de gdl:

        u = exx*x + exy*y + exz*z
        v = exy*x + eyy*y + eyz*z
        w = exz*x + eyz*y + ezz*z
    """
    x, y, z = nodos[:, 3], nodos[:, 4], nodos[:, 5]
    ix, iy, iz = indices_gdl(nodos)

    U = np.zeros(3 * len(nodos))
    U[ix] = exx * x + exy * y + exz * z
    U[iy] = exy * x + eyy * y + eyz * z
    U[iz] = exz * x + eyz * y + ezz * z
    return U


# ══════════════════════════════════════════════════════════════════════════════
# 3. SOLVER VEM 3D
#    (formulación trasladada sin cambios del código 3D verificado)
# ══════════════════════════════════════════════════════════════════════════════

def nodos_e(n, nodos, aristas, caras, elementos):
    """Nodos del elemento n, recorridos cara -> arista -> nodo."""
    elemento = elementos[n]
    caras_e = elemento[3:]
    idnodos = []
    for i in caras_e:
        cara = caras[i]
        for j in cara[1:]:
            arista = aristas[j]
            for k in range(1, len(arista)):
                if arista[k] not in idnodos:
                    idnodos.append(arista[k])
    return np.array((nodos[idnodos])[:, 0:6])


def monomios2D(x, y, xe, ye, he):
    return np.array([np.ones_like(x), (x - xe) / he, (y - ye) / he])


def infocara(nf, nodos, aristas, caras):
    """
    Geometría local de la cara nf y su proyector L2 (piast) en el plano
    de la cara. Devuelve los gdl globales, el sistema local (u, v, n), el
    área con signo, el centroide global y el proyector.
    """
    idaristas = caras[nf][1:]
    idnodos_c = []
    for i in idaristas:
        if aristas[i, 1] not in idnodos_c:
            idnodos_c.append(aristas[i, 1])
        if aristas[i, 2] not in idnodos_c:
            idnodos_c.append(aristas[i, 2])
        if i == idaristas[-1]:
            if aristas[i, 1] != idnodos_c[0] and aristas[i, 2] != idnodos_c[0]:
                idnodos_c = [idnodos_c[1]] + [idnodos_c[0]] + idnodos_c[2:]

    nodos_c = (nodos[idnodos_c])[:, 0:6]
    gdlx = nodos_c[:, 0]
    gdly = nodos_c[:, 1]
    gdlz = nodos_c[:, 2]

    p = nodos_c[:, 3:]

    # ── Normal robusta ───────────────────────────────────────────────────
    # Tomar siempre p[0], p[1], p[2] falla cuando esos tres vértices son
    # colineales (cara válida, pero a x b = 0 -> normal NaN).
    # Intento 1: buscar el primer par (p[0]->p[i], p[0]->p[j]) cuyo
    #            producto cruz no sea degenerado. Si p[0], p[1], p[2] no
    #            son colineales se obtiene i=1, j=2, es decir el mismo
    #            sistema local de antes.
    n_found = None
    u_found = None
    tol_cross = 1e-12

    for i in range(1, len(p)):
        a = p[i] - p[0]
        norm_a = np.linalg.norm(a)
        if norm_a < tol_cross:
            continue
        u_candidate = a / norm_a
        for j in range(i + 1, len(p)):
            b = p[j] - p[0]
            cruz = np.cross(a, b)
            norm_cruz = np.linalg.norm(cruz)
            if norm_cruz > tol_cross:
                n_found = cruz / norm_cruz
                u_found = u_candidate
                break
        if n_found is not None:
            break

    # Intento 2: SVD. El vector singular asociado al menor valor singular
    # es la normal del plano de mejor ajuste; funciona siempre que la cara
    # tenga área > 0.
    if n_found is None:
        centro = p.mean(axis=0)
        M = p - centro
        _, S, Vt = np.linalg.svd(M, full_matrices=False)
        n_found = Vt[-1]
        n_found = n_found / np.linalg.norm(n_found)
        # u en la dirección de mayor varianza de la cara
        u_found = Vt[0]
        u_found = u_found / np.linalg.norm(u_found)

    n = n_found
    u = u_found
    v = np.cross(n, u)
    # ─────────────────────────────────────────────────────────────────────

    p0 = np.array([np.mean(p[:, 0]), np.mean(p[:, 1]), np.mean(p[:, 2])])
    w = p - p0
    x = np.dot(w, u)
    y = np.dot(w, v)

    ngdl = len(x)
    E = 0.0
    xe = 0.0
    ye = 0.0
    for i in range(ngdl):
        E  += x[i-1] * y[i] - y[i-1] * x[i]
        xe += (x[i-1] + x[i]) * (x[i-1] * y[i] - x[i] * y[i-1])
        ye += (y[i-1] + y[i]) * (x[i-1] * y[i] - x[i] * y[i-1])
    E /= 2.0
    ye /= 6 * E
    xe /= 6 * E

    he = max((((x[i] - x[j])**2 + (y[i] - y[j])**2)**0.5
              for i in range(ngdl) for j in range(ngdl)), default=0.0)
    centroideglobal = p0 + xe * u + ye * v

    D = monomios2D(x, y, xe, ye, he).T
    G = np.zeros((3, 3))
    for i in range(1, 3):
        G[i, i] = E / he**2
    G[0] = np.mean(D, axis=0)

    dm = np.array([[0, 0], [1/he, 0], [0, 1/he]])
    B = np.zeros((3, ngdl))
    for j in range(ngdl):
        x0, y0, x1, y1 = x[j-1], y[j-1], x[j], y[j]
        x2, y2 = (x[0], y[0]) if j + 1 == ngdl else (x[j+1], y[j+1])

        nant = R_mat @ np.array([x1 - x0, y1 - y0])
        nant = nant / np.linalg.norm(nant)
        Lant = np.hypot(x1 - x0, y1 - y0)

        nsig = R_mat @ np.array([x2 - x1, y2 - y1])
        nsig = nsig / np.linalg.norm(nsig)
        Lsig = np.hypot(x2 - x1, y2 - y1)

        for i in range(3):
            B[i, j] += (np.dot(nant, dm[i]) * Lant / 2
                        + np.dot(nsig, dm[i]) * Lsig / 2)
    B[0] = 1 / ngdl
    piast = np.linalg.solve(G, B)

    return {"gdlx": gdlx, "gdly": gdly, "gdlz": gdlz, "x": x, "y": y,
            "E": E, "xe": xe, "ye": ye, "he": he, "n": n,
            "cglobal": centroideglobal, "aristas": idaristas, "piast": piast}


def construir_infocaras(nodos, aristas, caras):
    return [infocara(i, nodos, aristas, caras) for i in range(len(caras))]


def _copiar_cara(c):
    """Copia superficial con los campos que `orientar_cara` puede invertir."""
    return {"gdlx": c["gdlx"].copy(),
            "gdly": c["gdly"].copy(),
            "gdlz": c["gdlz"].copy(),
            "E": c["E"],
            "he": c["he"],
            "n": c["n"].copy(),
            "cglobal": c["cglobal"],
            "aristas": c["aristas"],
            "piast": c["piast"].copy()}


def orientar_cara(infocaras_e, aristas, max_iter=1000):
    """Orienta coherentemente las caras de un elemento (normales salientes
    o entrantes todas a la vez; el signo global se corrige en divergencia)."""
    aristascompletadas = []
    carascompletadas = [0]
    it = 1
    while len(carascompletadas) != len(infocaras_e):
        if it > max_iter:
            raise RuntimeError("orientar_cara no converge")
        it += 1
        for k in range(len(infocaras_e)):
            aristacara = infocaras_e[k]["aristas"]
            cond = True
            if k > 0 and k not in carascompletadas:
                cond = False
            if cond:
                for a in aristacara:
                    if a not in aristascompletadas:
                        orientacion1 = 0
                        aristascompletadas.append(a)
                        nodoa = aristas[a, 1]
                        nodob = aristas[a, 2]
                        for m in range(len(infocaras_e[k]["gdlx"])):
                            if infocaras_e[k]["gdlx"][m] == nodoa:
                                if infocaras_e[k]["gdlx"][m-1] == nodob:
                                    orientacion1 = 1
                        for j in range(len(infocaras_e)):
                            if a in infocaras_e[j]["aristas"]:
                                if j not in carascompletadas:
                                    carascompletadas.append(j)
                                    for i in range(len(infocaras_e[j]["gdlx"])):
                                        if infocaras_e[j]["gdlx"][i] == nodoa:
                                            if infocaras_e[j]["gdlx"][i-1] == nodob:
                                                orientacion1 += 1
                                            if orientacion1 == 0 or orientacion1 == 2:
                                                infocaras_e[j]["gdlx"] = infocaras_e[j]["gdlx"][::-1]
                                                infocaras_e[j]["gdly"] = infocaras_e[j]["gdly"][::-1]
                                                infocaras_e[j]["gdlz"] = infocaras_e[j]["gdlz"][::-1]
                                                infocaras_e[j]["piast"] = infocaras_e[j]["piast"][:, ::-1]
                                                infocaras_e[j]["n"] *= -1
                                                break
    return infocaras_e


def divergencia(n, aristas, caras, elementos, infocaras):
    """
    Volumen y centroide del elemento por el teorema de la divergencia,
    con las caras ya orientadas hacia afuera.
    """
    caras_e = np.array(elementos[n][3:])
    infocaras_e = [_copiar_cara(infocaras[i]) for i in caras_e]
    infocaras_e = orientar_cara(infocaras_e, aristas)

    V = 0.0
    C = np.zeros(3)
    for cara in infocaras_e:
        flujo = cara["E"] * np.dot(cara["cglobal"], cara["n"])
        V += flujo
        C += flujo * cara["cglobal"]
    V /= 3.0
    C /= 4 * V

    if V < 0:
        V = -V
        for cara in infocaras_e:
            cara["gdlx"] = cara["gdlx"][::-1]
            cara["gdly"] = cara["gdly"][::-1]
            cara["gdlz"] = cara["gdlz"][::-1]
            cara["piast"] = cara["piast"][:, ::-1]
            cara["n"] *= -1

    return V, C, infocaras_e


def monomios3D(x, y, z, centroide, he):
    xi    = (x - centroide[0]) / he
    eta   = (y - centroide[1]) / he
    citha = (z - centroide[2]) / he
    monomio = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1],
                        [-eta, xi, 0], [citha, 0, -xi], [0, -citha, eta],
                        [xi, 0, 0], [0, eta, 0], [0, 0, citha],
                        [eta, xi, 0], [citha, 0, xi], [0, citha, eta]])
    return monomio.T


def elemento_VEM3D(ne, nodos, aristas, caras, elementos, infocaras, info):
    """
    Matriz de rigidez del elemento poliédrico ne (VEM de orden 1).
    Guarda en `info` solo lo que usa el post-proceso.
    """
    Ematerial = elementos[ne][1]
    nu = elementos[ne][2]
    C1 = Ematerial * (1 - nu) / ((1 + nu) * (1 - 2 * nu))
    C2 = C1 * nu / (1 - nu)
    C3 = Ematerial / (2 * (1 + nu))
    C = np.array([[C1, C2, C2, 0, 0, 0],
                  [C2, C1, C2, 0, 0, 0],
                  [C2, C2, C1, 0, 0, 0],
                  [0, 0, 0, C3, 0, 0],
                  [0, 0, 0, 0, C3, 0],
                  [0, 0, 0, 0, 0, C3]])

    nodo_e = nodos_e(ne, nodos, aristas, caras, elementos)
    gdlx = nodo_e[:, 0].astype(int)
    gdly = nodo_e[:, 1].astype(int)
    gdlz = nodo_e[:, 2].astype(int)
    gdl = np.concatenate([gdlx, gdly, gdlz])

    x = nodo_e[:, 3]
    y = nodo_e[:, 4]
    z = nodo_e[:, 5]

    V, centroide, infocaras_e = divergencia(ne, aristas, caras, elementos,
                                            infocaras)
    ngdl = len(x)

    coords = np.stack([x, y, z], axis=1)
    diffs = coords[:, None, :] - coords[None, :, :]
    he = np.sqrt((diffs**2).sum(axis=2)).max()

    D = np.zeros((3 * ngdl, 12))
    for i in range(ngdl):
        mnodo = monomios3D(x[i], y[i], z[i], centroide, he)
        for j in range(12):
            D[i, j]            = mnodo[0, j]
            D[ngdl + i, j]     = mnodo[1, j]
            D[2 * ngdl + i, j] = mnodo[2, j]

    e11 = 1/he * np.array([0, 0, 0,  0, 0,  0, 1, 0, 0, 0, 0, 0])
    e12 = 1/he * np.array([0, 0, 0, -1, 0,  0, 0, 0, 0, 1, 0, 0])
    e13 = 1/he * np.array([0, 0, 0,  0, 1,  0, 0, 0, 0, 0, 1, 0])
    e21 = 1/he * np.array([0, 0, 0,  1, 0,  0, 0, 0, 0, 1, 0, 0])
    e22 = 1/he * np.array([0, 0, 0,  0, 0,  0, 0, 1, 0, 0, 0, 0])
    e23 = 1/he * np.array([0, 0, 0,  0, 0, -1, 0, 0, 0, 0, 0, 1])
    e31 = 1/he * np.array([0, 0, 0,  0, -1, 0, 0, 0, 0, 0, 1, 0])
    e32 = 1/he * np.array([0, 0, 0,  0, 0,  1, 0, 0, 0, 0, 0, 1])
    e33 = 1/he * np.array([0, 0, 0,  0, 0,  0, 0, 0, 1, 0, 0, 0])

    G = np.zeros((12, 12))
    for i in range(12):
        for j in range(12):
            G[i, j] = V * (
                (C1*e11[i] + C2*(e22[i] + e33[i])) * e11[j]
                + (C1*e22[i] + C2*(e11[i] + e33[i])) * e22[j]
                + (C1*e33[i] + C2*(e22[i] + e11[i])) * e33[j]
                + C3 * ((e12[i] + e21[i]) * (e12[j] + e21[j])
                        + (e31[i] + e13[i]) * (e31[j] + e13[j])
                        + (e32[i] + e23[i]) * (e32[j] + e23[j])))

    dpi = np.array([e11, e22, e33, e12 + e21, e13 + e31, e23 + e32])

    B  = np.zeros((12, ngdl * 3))
    Bs = np.zeros((12, ngdl * 3))
    for k in range(len(infocaras_e)):
        gdlglob = infocaras_e[k]["gdlx"]
        piastcara = infocaras_e[k]["piast"]
        intvh = infocaras_e[k]["E"] * np.array([1, 0, 0]) @ piastcara
        gdlcara = np.array([np.where(gdlx == g)[0][0] for g in gdlglob],
                           dtype=int)
        n = infocaras_e[k]["n"]
        N = np.array([[n[0], 0, 0], [0, n[1], 0], [0, 0, n[2]],
                      [n[1], n[0], 0], [n[2], 0, n[0]], [0, n[2], n[1]]])

        Bs[0, gdlcara]            += intvh
        Bs[1, gdlcara + ngdl]     += intvh
        Bs[2, gdlcara + ngdl * 2] += intvh
        Bs[3, gdlcara + ngdl]     += n[0] * intvh
        Bs[3, gdlcara]            += -n[1] * intvh
        Bs[4, gdlcara]            += n[2] * intvh
        Bs[4, gdlcara + ngdl * 2] += -n[0] * intvh
        Bs[5, gdlcara + ngdl * 2] += n[1] * intvh
        Bs[5, gdlcara + ngdl]     += -n[2] * intvh

        for i in range(12):
            em = dpi[:, i]
            t = ((em @ C) @ N)
            B[i, gdlcara]            += t[0] * intvh
            B[i, gdlcara + ngdl]     += t[1] * intvh
            B[i, gdlcara + 2 * ngdl] += t[2] * intvh

    Bs[6:, :] = B[6:, :]

    Gs = Bs @ D
    piast = np.linalg.solve(Gs, Bs)
    pi = D @ piast

    I = np.identity(3 * ngdl)
    Kc = piast.T @ (G @ piast)
    alpha = np.trace(Kc) / ngdl
    Ks = alpha * ((I - pi).T) @ (I - pi)
    K = Kc + Ks

    info[ne] = {"dpi": dpi, "piast": piast, "C": C, "gdl": gdl,
                "V": V, "he": he, "centroide": centroide}
    return K, gdl


def _acumular(K, filas, cols, vals, ndof):
    """Pasa las tripletas acumuladas a disperso y las suma a K."""
    if not vals:
        return K
    bloque = sp.coo_matrix((np.concatenate(vals),
                            (np.concatenate(filas), np.concatenate(cols))),
                           shape=(ndof, ndof)).tocsr()
    return bloque if K is None else (K + bloque)


#: Entradas de tripletas acumuladas antes de volcarlas a disperso. Cada
#: entrada cuesta 16 bytes (un float64 y dos int32), y `np.concatenate`
#: necesita ademas una copia, asi que el tope de abajo son ~380 MB por
#: volcado, independientemente del tamano de la malla.
MAX_ENTRADAS_BLOQUE = 12_000_000


def ensamblar(nodos, aristas, caras, elementos, infocaras, info,
              max_entradas=MAX_ENTRADAS_BLOQUE):
    """
    Ensambla K en formato disperso volcando por bloques de tamaño acotado.

    Guardar TODAS las tripletas y concatenarlas al final es lo que más
    memoria pide de toda la corrida: antes de que COO sume los repetidos
    hay sum_e (3*n_e)^2 entradas, varias veces el número de no-ceros de la
    K final, y `np.concatenate` necesita además una copia completa. En una
    malla de elementos poliédricos grandes eso llega a decenas de GB y
    basta para agotar la RAM.

    El volcado se dispara por NÚMERO DE ENTRADAS ACUMULADAS, no por número
    de elementos: un elemento VEM con 60 nodos aporta (3*60)^2 = 32 400
    entradas y uno con 10 aporta 900, así que contar elementos no acota
    nada. Contando entradas, el pico es el mismo en cualquier malla.
    """
    ndof = 3 * len(nodos)
    nelem = len(elementos)

    K = None
    filas, cols, vals = [], [], []
    acumuladas = 0
    for n in range(nelem):
        Ke, gdl = elemento_VEM3D(n, nodos, aristas, caras, elementos,
                                 infocaras, info)
        ii, jj = np.meshgrid(gdl, gdl, indexing="ij")
        filas.append(ii.ravel().astype(np.int32))
        cols.append(jj.ravel().astype(np.int32))
        vals.append(Ke.ravel())
        acumuladas += Ke.size
        if acumuladas >= max_entradas:
            K = _acumular(K, filas, cols, vals, ndof)
            filas, cols, vals = [], [], []
            acumuladas = 0

    K = _acumular(K, filas, cols, vals, ndof)
    if K is None:
        K = sp.csr_matrix((ndof, ndof))
    return K.tocsr()


# ══════════════════════════════════════════════════════════════════════════════
# 3b. ENSAMBLAJE PARTICIONADO (impone el borde durante el ensamblaje)
# ══════════════════════════════════════════════════════════════════════════════
#
# `ensamblar` construye la K completa y despues `factorizar` extrae K_ll. Eso
# obliga a tener K y K_ll (que es ~95 % de K) vivas a la vez, y ademas la
# acumulacion por bloques duplica la memoria en el instante de cada suma.
# Medido en dos mallas, el total de una corrida sale ~80 bytes por no-cero
# de K, y una malla de 2 600 M de no-ceros no cabe en 64 GB.
#
# `ensamblar_particionado` evita las dos cosas:
#
#   * Separa cada entrada elemental segun sus gdl YA en el bucle de
#     elementos, de modo que K nunca existe: salen directamente K_ll (libre-
#     libre), K_lf (libre-prescrito) y K_f (filas prescritas, que es lo unico
#     que hace falta para las reacciones del contorno).
#
#   * Reparte las tripletas de K_ll por FRANJAS DE FILAS y las vuelca a
#     archivos temporales. Como las franjas no comparten filas, al final se
#     concatenan en vez de sumarse: no hay ningun momento en que existan dos
#     copias de la matriz. Los archivos se leen con np.memmap, asi que las
#     tripletas no cuentan contra la memoria comprometida del proceso.
#
# El resultado es identico al de `ensamblar` + `factorizar`; lo unico que
# cambia es el orden en que se suman los numeros en punto flotante.

#: Entradas acumuladas en RAM por franja antes de volcarlas a disco.
ENTRADAS_BUFFER = 4_000_000

#: Filas de K_ll por franja. Con menos filas por franja hay mas archivos y
#: menos memoria en el armado final de cada bloque.
FILAS_POR_FRANJA = 1_500_000


class _Franja:
    """Acumula tripletas de un rango de filas y las vuelca a disco."""

    def __init__(self, carpeta, k, fila0, nfilas):
        self.fila0 = fila0
        self.nfilas = nfilas
        self.n = 0
        self._buf_i, self._buf_j, self._buf_v = [], [], []
        self._n_buf = 0
        self._f_i = open(os.path.join(carpeta, f"f{k}_i.bin"), "wb")
        self._f_j = open(os.path.join(carpeta, f"f{k}_j.bin"), "wb")
        self._f_v = open(os.path.join(carpeta, f"f{k}_v.bin"), "wb")
        self.rutas = (self._f_i.name, self._f_j.name, self._f_v.name)

    def agregar(self, i, j, v):
        self._buf_i.append(i)
        self._buf_j.append(j)
        self._buf_v.append(v)
        self._n_buf += i.size
        self.n += i.size
        if self._n_buf >= ENTRADAS_BUFFER:
            self.volcar()

    def volcar(self):
        if not self._buf_i:
            return
        np.concatenate(self._buf_i).astype(np.int32).tofile(self._f_i)
        np.concatenate(self._buf_j).astype(np.int32).tofile(self._f_j)
        np.concatenate(self._buf_v).astype(np.float64).tofile(self._f_v)
        self._buf_i, self._buf_j, self._buf_v = [], [], []
        self._n_buf = 0

    def cerrar(self):
        self.volcar()
        self._f_i.close(); self._f_j.close(); self._f_v.close()

    def borrar(self):
        """Elimina los archivos de la franja una vez consumida."""
        for r in self.rutas:
            try:
                os.remove(r)
            except OSError:
                pass

    def a_csr(self, ncols):
        """CSR de esta franja, leyendo las tripletas por memmap."""
        if self.n == 0:
            return sp.csr_matrix((self.nfilas, ncols))
        ri, rj, rv = self.rutas
        mi = np.memmap(ri, dtype=np.int32, mode="r", shape=(self.n,))
        mj = np.memmap(rj, dtype=np.int32, mode="r", shape=(self.n,))
        mv = np.memmap(rv, dtype=np.float64, mode="r", shape=(self.n,))
        M = sp.coo_matrix((mv, (mi - self.fila0, mj)),
                          shape=(self.nfilas, ncols)).tocsr()
        del mi, mj, mv
        M.sum_duplicates()
        return M


def ensamblar_particionado(nodos, aristas, caras, elementos, infocaras, info,
                           borde_mask, base_tmp=None, en_disco=True):
    """
    Ensambla directamente el sistema particionado por las condiciones KUBC.

    Devuelve el mismo diccionario que `factorizar` espera recibir ya armado:
        {"K_ll", "K_lf", "K_f", "idx_l", "idx_f", "ndof"}

    `base_tmp` es la carpeta DENTRO de la cual se crea el directorio de
    trabajo (por defecto, el temporal del sistema). Necesita espacio en
    disco del orden de 16 bytes por entrada elemental, sum_e (3*n_e)^2, mas
    12 bytes por no-cero de K_ll para los arreglos definitivos.
    """
    ndof = 3 * len(nodos)
    ix, iy, iz = indices_gdl(nodos)

    fijo = np.zeros(ndof, dtype=bool)
    fijo[ix[borde_mask]] = True
    fijo[iy[borde_mask]] = True
    fijo[iz[borde_mask]] = True

    idx_l = np.where(~fijo)[0]
    idx_f = np.where(fijo)[0]
    n_l, n_f = len(idx_l), len(idx_f)

    # gdl global -> indice local dentro de cada bloque (-1 si no pertenece)
    a_libre = np.full(ndof, -1, dtype=np.int64)
    a_libre[idx_l] = np.arange(n_l)
    a_fijo = np.full(ndof, -1, dtype=np.int64)
    a_fijo[idx_f] = np.arange(n_f)

    ok = False
    propia = True
    if base_tmp:
        os.makedirs(base_tmp, exist_ok=True)
    carpeta_tmp = tempfile.mkdtemp(prefix="vem3d_", dir=base_tmp or None)

    try:
        n_franjas = max(1, -(-n_l // FILAS_POR_FRANJA))
        filas_franja = -(-n_l // n_franjas) if n_franjas else n_l
        franjas = []
        for k in range(n_franjas):
            f0 = k * filas_franja
            franjas.append(_Franja(carpeta_tmp, k, f0,
                                   min(filas_franja, n_l - f0)))

        lf_i, lf_j, lf_v = [], [], []
        ff_i, ff_j, ff_v = [], [], []

        for n in range(len(elementos)):
            Ke, gdl = elemento_VEM3D(n, nodos, aristas, caras, elementos,
                                     infocaras, info)
            gdl = np.asarray(gdl, dtype=np.int64)
            loc_l = a_libre[gdl]
            es_libre = loc_l >= 0

            # ── filas prescritas: se guardan enteras (para las reacciones)
            if (~es_libre).any():
                fil = a_fijo[gdl[~es_libre]]
                sub = Ke[~es_libre, :]
                ff_i.append(np.repeat(fil, gdl.size))
                ff_j.append(np.tile(gdl, fil.size))
                ff_v.append(sub.ravel())

            if not es_libre.any():
                continue

            rl = loc_l[es_libre]

            # ── bloque libre-libre, repartido por franjas de filas
            sub_ll = Ke[np.ix_(es_libre, es_libre)]
            ii = np.repeat(rl, rl.size)
            jj = np.tile(rl, rl.size)
            vv = sub_ll.ravel()
            if n_franjas == 1:
                franjas[0].agregar(ii, jj, vv)
            else:
                kk = ii // filas_franja
                for k in np.unique(kk):
                    m = kk == k
                    franjas[int(k)].agregar(ii[m], jj[m], vv[m])

            # ── bloque libre-prescrito
            if (~es_libre).any():
                col = a_fijo[gdl[~es_libre]]
                sub_lf = Ke[np.ix_(es_libre, ~es_libre)]
                lf_i.append(np.repeat(rl, col.size))
                lf_j.append(np.tile(col, rl.size))
                lf_v.append(sub_lf.ravel())

        for fr in franjas:
            fr.cerrar()

        # ── armado final de K_ll ────────────────────────────────────
        # Las franjas no comparten filas, de modo que los arreglos CSR
        # finales son la simple concatenacion de los suyos: no hace falta
        # sumar matrices ni tener dos copias vivas. Cada franja se arma, se
        # escribe a los archivos de salida y se libera antes de armar la
        # siguiente, asi que el pico es UNA franja, no la matriz entera.
        #
        # Si `en_disco`, los arreglos definitivos quedan mapeados a archivo.
        # En Windows esas paginas no cuentan contra la memoria comprometida
        # (que es la que produce el "Unable to allocate"), sino que el
        # sistema las cachea segun la RAM disponible. Es lo que permite
        # resolver mallas cuya K_ll no cabe holgadamente en memoria.
        ruta_d = os.path.join(carpeta_tmp, "kll_datos.bin")
        ruta_i = os.path.join(carpeta_tmp, "kll_indices.bin")
        indptr = np.zeros(n_l + 1, dtype=np.int64)

        total_nnz = 0
        fila = 0
        with open(ruta_d, "wb") as fd, open(ruta_i, "wb") as fi:
            for fr in franjas:
                b = fr.a_csr(n_l)
                b.sort_indices()
                b.data.astype(np.float64).tofile(fd)
                b.indices.astype(np.int32).tofile(fi)
                indptr[fila + 1:fila + 1 + b.shape[0]] = b.indptr[1:] + total_nnz
                total_nnz += b.nnz
                fila += b.shape[0]
                fr.borrar()
                del b
        franjas.clear()

        if en_disco:
            datos = np.memmap(ruta_d, dtype=np.float64, mode="r",
                              shape=(total_nnz,))
            indices = np.memmap(ruta_i, dtype=np.int32, mode="r",
                                shape=(total_nnz,))
        else:
            datos = np.fromfile(ruta_d, dtype=np.float64)
            indices = np.fromfile(ruta_i, dtype=np.int32)
            os.remove(ruta_d)
            os.remove(ruta_i)

        K_ll = sp.csr_matrix((datos, indices, indptr), shape=(n_l, n_l))

        K_lf = _coo_de_listas(lf_i, lf_j, lf_v, (n_l, n_f))
        K_f = _coo_de_listas(ff_i, ff_j, ff_v, (n_f, ndof))

        ok = True
        return {"K_ll": K_ll, "K_lf": K_lf, "K_f": K_f,
                "idx_l": idx_l, "idx_f": idx_f, "ndof": ndof,
                "carpeta_tmp": carpeta_tmp if en_disco else None}

    finally:
        # La carpeta solo se borra aqui si algo fallo o si no quedo nada
        # mapeado a ella; si K_ll vive en disco, la borra el llamador
        # cuando suelta el sistema (`liberar_sistema`).
        if propia and (not ok or not en_disco):
            shutil.rmtree(carpeta_tmp, ignore_errors=True)


def _coo_de_listas(li, lj, lv, forma):
    """CSR a partir de listas de tripletas (bloques chicos: K_lf y K_f)."""
    if not li:
        return sp.csr_matrix(forma)
    M = sp.coo_matrix((np.concatenate(lv),
                       (np.concatenate(li).astype(np.int32),
                        np.concatenate(lj).astype(np.int32))),
                      shape=forma).tocsr()
    li.clear(); lj.clear(); lv.clear()
    return M


def liberar_sistema(sistema):
    """
    Suelta el sistema y borra la carpeta temporal donde viven los arreglos
    de K_ll cuando se ensamblo en disco. Hay que llamarla DESPUES de la
    ultima operacion que use K_ll.
    """
    if not sistema:
        return
    carpeta = sistema.pop("carpeta_tmp", None)
    sistema.pop("op", None)
    sistema.pop("K_ll", None)
    sistema.pop("K_lf", None)
    sistema.pop("K_f", None)
    if carpeta:
        shutil.rmtree(carpeta, ignore_errors=True)


def factorizar_particionado(part, solver="auto"):
    """
    Prepara el operador del solver sobre un sistema ya particionado por
    `ensamblar_particionado`. Es la mitad de `factorizar` que no depende de
    como se haya construido K_ll, de modo que ambos caminos comparten solver
    y los tiempos siguen siendo comparables.
    """
    K_ll = part["K_ll"]
    n_l = K_ll.shape[0]

    if solver == "auto":
        solver = "lu" if (n_l < UMBRAL_GDL_LU or not HAY_PYAMG) else "amg"
    if solver == "amg" and not HAY_PYAMG:
        raise RuntimeError("solver='amg' requiere pyamg (pip install pyamg).")
    if solver not in ("lu", "amg"):
        raise ValueError(f"solver desconocido: {solver!r}")

    sistema = {"solver": solver, "K_lf": part["K_lf"], "K_f": part["K_f"],
               "idx_l": part["idx_l"], "idx_f": part["idx_f"],
               "ndof": part["ndof"], "iters": [], "res_rel": [],
               "carpeta_tmp": part.get("carpeta_tmp")}

    if n_l == 0:
        sistema["op"] = None
        return sistema

    if solver == "lu":
        sistema["op"] = splu(sp.csr_matrix(K_ll).tocsc())
    else:
        sistema["op"] = pyamg.smoothed_aggregation_solver(
            K_ll, max_coarse=AMG_MAX_COARSE,
            smooth=AMG_SMOOTH, strength=AMG_STRENGTH)
        sistema["K_ll"] = K_ll
    return sistema


def factorizar(K, nodos, borde_mask, solver="auto"):
    """
    Particiona los gdl en libres (l) y prescritos (f) y prepara UNA sola vez
    el operador con el que se resuelven los seis casos de carga. Los seis
    fijan EXACTAMENTE los mismos gdl (los del contorno) y solo cambian los
    valores prescritos, de modo que K_ll es común a los seis: con LU se
    reutiliza la factorización y con AMG se reutiliza la jerarquía.

    solver:
      "lu"   -> LU disperso (SuperLU). Exacto, pero el relleno crece muy
                rápido en 3D: una malla de 116 304 gdl libres necesitó más
                de 6 GB y fue abortada por falta de memoria.
      "amg"  -> multigrid algebraico por agregación (pyamg) con CG.
                Para esa misma malla: 2.8 GB de pico, 42-43 iteraciones por
                caso y residuo relativo ~1e-12.
      "auto" -> LU si hay pocos gdl libres (< UMBRAL_GDL_LU) y AMG si no,
                cayendo a LU cuando pyamg no está instalado.
    """
    ndof = 3 * len(nodos)
    ix, iy, iz = indices_gdl(nodos)

    fijo = np.zeros(ndof, dtype=bool)
    fijo[ix[borde_mask]] = True
    fijo[iy[borde_mask]] = True
    fijo[iz[borde_mask]] = True

    idx_l = np.where(~fijo)[0]
    idx_f = np.where(fijo)[0]

    if solver == "auto":
        if len(idx_l) < UMBRAL_GDL_LU or not HAY_PYAMG:
            solver = "lu"
        else:
            solver = "amg"
    if solver == "amg" and not HAY_PYAMG:
        raise RuntimeError(
            "solver='amg' requiere pyamg (pip install pyamg).")
    if solver not in ("lu", "amg"):
        raise ValueError(f"solver desconocido: {solver!r}")

    # Se guardan tambien las FILAS prescritas, que son las unicas que hacen
    # falta para las reacciones del contorno. Asi el llamador puede liberar
    # la K completa en cuanto esta funcion retorna, en vez de mantenerla
    # viva junto con K_ll durante los seis casos de carga.
    K_lf = K[idx_l][:, idx_f].tocsr()
    sistema = {"solver": solver, "K_lf": K_lf, "K_f": K[idx_f].tocsr(),
               "idx_l": idx_l, "idx_f": idx_f, "ndof": ndof,
               "iters": [], "res_rel": []}

    if len(idx_l) == 0:
        sistema["op"] = None
        return sistema

    if solver == "lu":
        sistema["op"] = splu(K[idx_l][:, idx_l].tocsc())
    else:
        K_ll = K[idx_l][:, idx_l].tocsr()
        sistema["op"] = pyamg.smoothed_aggregation_solver(
            K_ll, max_coarse=AMG_MAX_COARSE,
            smooth=AMG_SMOOTH, strength=AMG_STRENGTH)
        sistema["K_ll"] = K_ll
    return sistema


def resolver(sistema, u_f):
    """Resuelve K_ll u_l = -K_lf u_f reutilizando el operador preparado."""
    U = np.zeros(sistema["ndof"])
    U[sistema["idx_f"]] = u_f
    idx_l = sistema["idx_l"]
    if len(idx_l) == 0:
        return U

    b = -(sistema["K_lf"] @ u_f)

    if sistema["solver"] == "lu":
        U[idx_l] = sistema["op"].solve(b)
        return U

    residuos = []
    u_l = sistema["op"].solve(b, tol=TOL_AMG, accel="cg",
                              residuals=residuos, maxiter=500)
    nb = np.linalg.norm(b)
    rel = (np.linalg.norm(b - sistema["K_ll"] @ u_l) / nb
           if nb > 0 else 0.0)
    sistema["iters"].append(max(len(residuos) - 1, 0))
    sistema["res_rel"].append(float(rel))
    U[idx_l] = u_l
    return U


def post_proceso(U, elementos, info):
    """Deformación y esfuerzo constantes por elemento (proyección VEM)."""
    ne = len(elementos)
    eps = np.zeros((ne, 6))
    sig = np.zeros((ne, 6))
    for i in range(ne):
        ue = U[info[i]["gdl"]]
        e = (info[i]["dpi"] @ info[i]["piast"]) @ ue
        eps[i] = e
        sig[i] = info[i]["C"] @ e
    return eps, sig


# ══════════════════════════════════════════════════════════════════════════════
# 4. HOMOGENIZACIÓN
# ══════════════════════════════════════════════════════════════════════════════

def C_isotropica_3D(E, nu):
    """Tensor constitutivo isotrópico 3D en notación de Voigt (ingenieril)."""
    f = E / ((1 + nu) * (1 - 2 * nu))
    G = E / (2 * (1 + nu))
    C = np.zeros((6, 6))
    C[0, 0] = C[1, 1] = C[2, 2] = f * (1 - nu)
    C[0, 1] = C[0, 2] = C[1, 2] = f * nu
    C[1, 0] = C[2, 0] = C[2, 1] = f * nu
    C[3, 3] = C[4, 4] = C[5, 5] = G
    return C


def extraer_E_nu_directo(C):
    """
    Extracción directa que usaba el script 3D anterior:
        nu = C12 / (C11 + C12),   E = 2*C44*(1 + nu)
    con C11, C12 y C44 promediados sobre las entradas equivalentes.
    Solo es exacta si C es isotrópica; se reporta para comparar.
    """
    C11 = np.mean([C[0, 0], C[1, 1], C[2, 2]])
    C12 = np.mean([C[0, 1], C[0, 2], C[1, 2], C[1, 0], C[2, 0], C[2, 1]])
    C44 = np.mean([C[3, 3], C[4, 4], C[5, 5]])
    nu = C12 / (C11 + C12) if (C11 + C12) != 0 else 0.0
    E = 2 * C44 * (1 + nu)
    return float(E), float(nu)


def ajustar_E_nu(C):
    """
    Ajusta por mínimos cuadrados el par (E, nu) cuya C isotrópica mejor
    aproxima la C calculada, usando las 36 entradas de la matriz.
    """
    def residuo(p):
        return (C_isotropica_3D(p[0], p[1]) - C).ravel()

    E0, nu0 = extraer_E_nu_directo(C)
    if not np.isfinite(E0) or E0 <= 0:
        E0 = max(abs(C[0, 0]), 1.0)
    nu0 = float(np.clip(nu0, -0.3, 0.45))
    sol = least_squares(residuo, [E0, nu0],
                        bounds=([1e-6, -0.95], [np.inf, 0.49]))
    return float(sol.x[0]), float(sol.x[1])


def anisotropia(C):
    """
    Medida escalar de la desviación respecto a la mejor C isotrópica:
        ||C - C_iso(E_eq, nu_eq)||_max / max|C|
    No es un índice estándar; sirve para detectar mallas cuya C efectiva
    no admite una lectura isotrópica razonable.
    """
    E, nu = ajustar_E_nu(C)
    escala = np.abs(C).max()
    if escala == 0:
        return 0.0
    return float(np.abs(C - C_isotropica_3D(E, nu)).max() / escala)


def fraccion_agregado(info, codigos):
    """
    Fracción de volumen ocupada por agregado.
    El código COD_MATRIZ (= 0) es la matriz; cualquier otro es agregado.
    """
    V_tot = sum(info[i]["V"] for i in range(len(info)))
    V_agg = sum(info[i]["V"] for i in range(len(info))
                if int(codigos[i]) != COD_MATRIZ)
    return (V_agg / V_tot if V_tot else 0.0), V_tot


def esfuerzo_por_fuerzas(sistema, U, nodos, borde_mask, V):
    """
    Esfuerzo medio a partir de las reacciones en el contorno:

        <sigma_ij> = (1/V) * sum_(nodos borde) F_i * x_j

    Es la otra lectura del teorema del esfuerzo medio. Debe coincidir con el
    promedio volumétrico; se calculan las dos para poder contrastarlas.
    Las componentes cortantes se simetrizan.
    """
    # Solo hacen falta las reacciones en los gdl prescritos (los del
    # contorno), que son los unicos que intervienen en las sumas de abajo;
    # por eso basta con las filas prescritas de K y no la K completa.
    F = np.zeros(sistema["ndof"])   # no hay fuerzas de cuerpo
    F[sistema["idx_f"]] = sistema["K_f"] @ U
    ix, iy, iz = indices_gdl(nodos)
    Fx, Fy, Fz = F[ix], F[iy], F[iz]
    x, y, z = nodos[:, 3], nodos[:, 4], nodos[:, 5]

    b = borde_mask
    sxx = np.dot(Fx[b], x[b]) / V
    syy = np.dot(Fy[b], y[b]) / V
    szz = np.dot(Fz[b], z[b]) / V
    sxy = 0.5 * (np.dot(Fx[b], y[b]) + np.dot(Fy[b], x[b])) / V
    sxz = 0.5 * (np.dot(Fx[b], z[b]) + np.dot(Fz[b], x[b])) / V
    syz = 0.5 * (np.dot(Fy[b], z[b]) + np.dot(Fz[b], y[b])) / V
    return np.array([sxx, syy, szz, sxy, sxz, syz])


def homogenizar(nodos, elementos, info, sistema, borde_mask, V_bbox):
    """
    Resuelve los seis casos de carga y arma la C 6x6 por dos vías
    independientes: promedio volumétrico de esfuerzos y reacciones de borde.
    Recibe el `sistema` ya preparado por `factorizar`, no la K completa:
    asi el llamador puede liberar K antes de resolver y no se mantienen K y
    K_ll vivas a la vez.

    Devuelve (C_vol, C_fuerzas, resultados_por_caso, info_solver), donde
    info_solver recoge el solver usado y, con AMG, las iteraciones y el
    residuo relativo de cada caso, para dejarlos registrados en el resumen.
    """
    C      = np.zeros((6, 6))
    C_fuer = np.zeros((6, 6))
    resultados = {}

    vols = np.array([info[i]["V"] for i in range(len(info))])
    V_tot = vols.sum()

    idx_f = sistema["idx_f"]

    for col, (nombre, exx, eyy, ezz, exy, exz, eyz) in enumerate(CASOS_CARGA):
        u_teo = campo_lineal(nodos, exx, eyy, ezz, exy, exz, eyz)
        U = resolver(sistema, u_teo[idx_f])
        eps, sig = post_proceso(U, elementos, info)

        # Vía 1: promedio de esfuerzos ponderado por volumen
        sigma_prom = (sig * vols[:, None]).sum(axis=0) / V_tot
        C[:, col] = sigma_prom

        # Vía 2: reacciones en el contorno
        sigma_fuer = esfuerzo_por_fuerzas(sistema, U, nodos, borde_mask,
                                          V_bbox)
        C_fuer[:, col] = sigma_fuer

        # Desviación respecto al campo lineal impuesto
        denom = np.abs(u_teo).max()
        err_rel = np.abs(U - u_teo).max() / denom if denom > 0 else 0.0

        resultados[nombre] = {"U": U, "eps": eps, "sig": sig,
                              "sigma_prom": sigma_prom,
                              "sigma_fuerzas": sigma_fuer,
                              "err_rel": err_rel}

    info_solver = {
        "nombre": sistema["solver"],
        "n_gdl_libres": int(len(sistema["idx_l"])),
        "iters_max": (max(sistema["iters"]) if sistema["iters"] else 0),
        "res_rel_max": (max(sistema["res_rel"]) if sistema["res_rel"]
                        else 0.0),
    }
    return C, C_fuer, resultados, info_solver


# ══════════════════════════════════════════════════════════════════════════════
# 5. ESCRITURA DE RESULTADOS POR MALLA
# ══════════════════════════════════════════════════════════════════════════════

def guardar_resultados_malla(carpeta, nodos, C, C_fuer, resultados, meta,
                             guardar_campos=True):
    """
    Escribe los .txt de una réplica:
      C_equivalente.txt, C_equivalente_fuerzas.txt, resumen.txt,
      U_<caso>.txt, deformaciones_<caso>.txt, esfuerzos_<caso>.txt
    """
    os.makedirs(carpeta, exist_ok=True)
    ix, iy, iz = indices_gdl(nodos)

    if guardar_campos:
        for nombre, r in resultados.items():
            U = r["U"]
            np.savetxt(os.path.join(carpeta, f"U_{nombre}.txt"),
                       np.column_stack([U[ix], U[iy], U[iz]]),
                       fmt="%.12e", header="ux  uy  uz", comments="# ")
            np.savetxt(os.path.join(carpeta, f"deformaciones_{nombre}.txt"),
                       np.column_stack([np.arange(len(r["eps"])),
                                        meta["volumenes"], r["eps"]]),
                       fmt=["%.0f"] + ["%.12e"] * 7,
                       header="elem  V  " + "  ".join(COMP_EPS),
                       comments="# ")
            np.savetxt(os.path.join(carpeta, f"esfuerzos_{nombre}.txt"),
                       np.column_stack([np.arange(len(r["sig"])),
                                        meta["volumenes"], r["sig"]]),
                       fmt=["%.0f"] + ["%.12e"] * 7,
                       header="elem  V  " + "  ".join(COMP_SIGMA),
                       comments="# ")

    etiq_c = [c[0] for c in CASOS_CARGA]

    # ── C_equivalente.txt (promedio volumétrico) ─────────────────────
    with open(os.path.join(carpeta, "C_equivalente.txt"), "w",
              encoding="utf-8") as f:
        f.write("# Matriz constitutiva homogenizada C (Pa) - VEM 3D\n")
        f.write("# Obtenida por promedio de esfuerzos ponderado por volumen\n")
        f.write("# Voigt: sigma = [sxx, syy, szz, txy, txz, tyz]\n")
        f.write("#        eps   = [exx, eyy, ezz, gxy, gxz, gyz]\n")
        f.write("# Columnas: " + ", ".join(etiq_c) + "\n")
        for i in range(6):
            f.write("  ".join(f"{C[i, j]:+.12e}" for j in range(6)) + "\n")

    # ── C_equivalente_fuerzas.txt (reacciones de borde) ──────────────
    with open(os.path.join(carpeta, "C_equivalente_fuerzas.txt"), "w",
              encoding="utf-8") as f:
        f.write("# Matriz constitutiva homogenizada C (Pa) - VEM 3D\n")
        f.write("# Obtenida por reacciones en el contorno: "
                "sigma_ij = (1/V) sum F_i x_j\n")
        f.write(f"# Volumen usado (caja): {meta['vol_bbox']:.12e}\n")
        f.write("# Columnas: " + ", ".join(etiq_c) + "\n")
        for i in range(6):
            f.write("  ".join(f"{C_fuer[i, j]:+.12e}" for j in range(6)) + "\n")

    # ── resumen.txt ──────────────────────────────────────────────────
    L = []
    L.append("=" * 100)
    L.append(f"RESUMEN VEM 3D - {meta['malla']}")
    L.append("=" * 100)
    L.append("")
    L.append(f"Modo                 : {meta['modo']}")
    L.append(f"Nodos                : {meta['n_nodos']:,}")
    L.append(f"Aristas              : {meta['n_aristas']:,}")
    L.append(f"Caras                : {meta['n_caras']:,}")
    L.append(f"Elementos            : {meta['n_elementos']:,}")
    L.append(f"Grados de libertad   : {meta['n_dofs']:,}")
    L.append(f"Nodos en borde       : {meta['n_borde']:,}")
    L.append(f"  gdl prescritos     : {3*meta['n_borde']:,}")
    L.append(f"  gdl libres         : {meta['n_gdl_libres']:,}")
    L.append("")
    L.append(f"Solver               : {meta['solver']}")
    if meta["solver"] == "amg":
        L.append(f"  iteraciones (max)  : {meta['solver_iters_max']}")
        L.append(f"  residuo rel. (max) : {meta['solver_res_rel']:.3e}")
    L.append("")
    L.append(f"Volumen (suma elem.) : {meta['vol_total']:.6e}")
    L.append(f"Volumen (caja)       : {meta['vol_bbox']:.6e}")
    L.append(f"  discrepancia rel.  : {meta['dif_volumen']:.3e}")
    L.append(f"Dominio Lx, Ly, Lz   : {meta['Lx']:.6e}, "
             f"{meta['Ly']:.6e}, {meta['Lz']:.6e}")
    L.append(f"Fraccion agregado    : {100*meta['frac_agg']:.2f} %"
             f"   (codigo matriz = {COD_MATRIZ})")
    L.append("")
    if meta.get("memoria_pico_gb"):
        L.append(f"Memoria pico         : {meta['memoria_pico_gb']:.2f} GB")
        L.append("")
    L.append(f"Tiempo total         : {meta['tiempo_s']:.1f} s")
    L.append(f"  lectura            : {meta['t_lectura']:.1f} s")
    L.append(f"  infocaras          : {meta['t_infocaras']:.1f} s")
    L.append(f"  ensamblaje         : {meta['t_ensamblaje']:.1f} s")
    L.append(f"  solver (6 casos)   : {meta['t_solver']:.1f} s")
    L.append("")
    L.append("-" * 100)
    L.append("MATRIZ C HOMOGENIZADA (Pa) - promedio de esfuerzos por volumen")
    L.append("-" * 100)
    L.append(f"{'':8s}" + "".join(f"{e:>18s}" for e in etiq_c))
    for i in range(6):
        L.append(f"{COMP_SIGMA[i]:<8s}"
                 + "".join(f"{C[i, j]:+18.6e}" for j in range(6)))
    L.append("")
    L.append("-" * 100)
    L.append("MATRIZ C HOMOGENIZADA (Pa) - reacciones en el contorno")
    L.append("-" * 100)
    L.append(f"{'':8s}" + "".join(f"{e:>18s}" for e in etiq_c))
    for i in range(6):
        L.append(f"{COMP_SIGMA[i]:<8s}"
                 + "".join(f"{C_fuer[i, j]:+18.6e}" for j in range(6)))
    L.append("")
    L.append(f"  Discrepancia relativa maxima entre ambas vias: "
             f"{meta['dif_vias']:.3e}")
    L.append("")
    L.append("-" * 100)
    L.append("CONSTANTES EFECTIVAS")
    L.append("-" * 100)
    L.append("  Ajuste por minimos cuadrados sobre las 36 entradas de C:")
    L.append(f"    E_eq  = {meta['E_eq']:.6e} Pa")
    L.append(f"    nu_eq = {meta['nu_eq']:.6f}")
    L.append("  Extraccion directa C11/C12/C44 (valida solo si C es isotropica):")
    L.append(f"    E_dir  = {meta['E_dir']:.6e} Pa")
    L.append(f"    nu_dir = {meta['nu_dir']:.6f}")
    L.append(f"  Desviacion de C respecto a la mejor C isotropica: "
             f"{meta['anisotropia']:.3e}")
    L.append("")
    L.append("-" * 100)
    if meta["modo"] == "homogeneo":
        L.append("PATCH TEST (campo lineal impuesto)")
        L.append("-" * 100)
        L.append("  El medio es homogeneo, asi que el campo lineal del contorno")
        L.append("  es la solucion exacta: estos valores deben ser ~1e-14.")
    else:
        L.append("DESVIACION RESPECTO AL CAMPO LINEAL DEL CONTORNO")
        L.append("-" * 100)
        L.append("  NO es un patch test: en un medio heterogeneo el campo interno")
        L.append("  no es lineal, de modo que un valor no nulo es lo esperado.")
    L.append("")
    L.append("  Maximo relativo del desplazamiento, por caso de carga:")
    for nombre, r in resultados.items():
        L.append(f"    {nombre:<8s} {r['err_rel']:.3e}")
    L.append("")

    if meta["modo"] == "homogeneo":
        L.append("-" * 100)
        L.append("VERIFICACION CONTRA LA C TEORICA (material uniforme)")
        L.append("-" * 100)
        Ct = meta["C_teorica"]
        L.append(f"  E  = {E_HOM:.6e} Pa      nu = {NU_HOM}")
        L.append("")
        L.append(f"  {'Entrada':<10s}{'Calculado':>20s}{'Teorico':>20s}"
                 f"{'Err rel':>14s}")
        for et, (i, j) in [("C11", (0, 0)), ("C22", (1, 1)), ("C33", (2, 2)),
                           ("C12", (0, 1)), ("C13", (0, 2)), ("C23", (1, 2)),
                           ("C44", (3, 3)), ("C55", (4, 4)), ("C66", (5, 5)),
                           ("C14", (0, 3)), ("C25", (1, 4)), ("C36", (2, 5))]:
            calc, teo = C[i, j], Ct[i, j]
            err = abs(calc - teo) / abs(teo) if abs(teo) > 0 else abs(calc)
            L.append(f"  {et:<10s}{calc:>20.6e}{teo:>20.6e}{err:>14.3e}")
        L.append("")
        L.append(f"  Error relativo maximo : {meta['err_C_max']:.3e}")
        L.append(f"  Tolerancia            : {meta['tol_C']:.1e}")
        estado = "COINCIDE" if meta["verif_ok"] else "NO COINCIDE"
        L.append(f"  Resultado             : {estado}")
        L.append("")

    L.append("-" * 100)
    L.append("ESTADISTICAS DE ESFUERZOS POR CASO")
    L.append("-" * 100)
    for nombre, r in resultados.items():
        L.append(f"\n  Caso: {nombre}")
        L.append(f"  {'Comp':<6s}{'Min':>16s}{'Max':>16s}{'Media':>16s}"
                 f"{'Desv.Std':>16s}")
        L.append("  " + "-" * 70)
        for k, comp in enumerate(COMP_SIGMA):
            v = r["sig"][:, k]
            L.append(f"  {comp:<6s}{v.min():>16.4e}{v.max():>16.4e}"
                     f"{v.mean():>16.4e}{v.std():>16.4e}")
    L.append("")
    L.append("=" * 100)

    with open(os.path.join(carpeta, "resumen.txt"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(L))


# ══════════════════════════════════════════════════════════════════════════════
# 6. EJECUCIÓN DE UNA MALLA
# ══════════════════════════════════════════════════════════════════════════════

def correr_malla(carpeta_vem, carpeta_salida, nombre_malla, modo,
                 tol_C=1e-6, guardar_campos=True, solver="auto",
                 base_tmp=None):
    """
    Corre la homogenización de una réplica.

    modo: "homogeneo"   -> E=1e6, nu=0.3 en todos los elementos y se
                           verifica C contra la teórica.
          "heterogeneo" -> se leen las propiedades del Propiedades.txt de
                           la propia malla.

    Devuelve un dict con el resultado (ok / error y los datos del resumen).
    """
    t0 = time.perf_counter()
    try:
        # ── lectura ──────────────────────────────────────────────────
        t = time.perf_counter()
        f_nod = _buscar_archivo(carpeta_vem, "Input_nodos.txt",
                                "input_nodos.txt")
        f_ari = _buscar_archivo(carpeta_vem, "Input_aristas.txt",
                                "input_aristas.txt")
        f_car = _buscar_archivo(carpeta_vem, "Input_caras.txt",
                                "input_caras.txt")
        f_ele = _buscar_archivo(carpeta_vem, "input_elements.txt",
                                "Input_elements.txt", "input_elementos.txt",
                                "Input_elementos.txt")

        nodos = leer_nodos(f_nod)
        aristas = leer_aristas(f_ari)
        caras = leer_caras(f_car)
        elementos, codigos = leer_elementos(f_ele)

        if modo == "homogeneo":
            asignar_propiedades_uniformes(elementos, E_HOM, NU_HOM)
        else:
            f_pro = _buscar_archivo(carpeta_vem, "Propiedades.txt",
                                    "propiedades.txt", "propiedades2.txt")
            asignar_propiedades(elementos, leer_propiedades(f_pro), codigos)
        t_lectura = time.perf_counter() - t

        # ── geometría de caras ───────────────────────────────────────
        t = time.perf_counter()
        infocaras = construir_infocaras(nodos, aristas, caras)
        t_infocaras = time.perf_counter() - t

        # ── ensamblaje ───────────────────────────────────────────────
        # Se impone el contorno DURANTE el ensamblaje: nunca se construye la
        # K completa, solo los bloques que el solver necesita.
        borde_mask = detectar_borde(nodos)
        t = time.perf_counter()
        info = {}
        part = ensamblar_particionado(nodos, aristas, caras, elementos,
                                      infocaras, info, borde_mask,
                                      base_tmp=base_tmp)
        t_ensamblaje = time.perf_counter() - t
        del infocaras

        # ── homogenización ───────────────────────────────────────────
        x, y, z = nodos[:, 3], nodos[:, 4], nodos[:, 5]
        Lx = x.max() - x.min()
        Ly = y.max() - y.min()
        Lz = z.max() - z.min()
        V_bbox = Lx * Ly * Lz

        t = time.perf_counter()
        sistema = factorizar_particionado(part, solver=solver)
        del part
        C, C_fuer, resultados, info_solver = homogenizar(
            nodos, elementos, info, sistema, borde_mask, V_bbox)
        liberar_sistema(sistema)
        del sistema
        t_solver = time.perf_counter() - t

        E_eq, nu_eq = ajustar_E_nu(C)
        E_dir, nu_dir = extraer_E_nu_directo(C)
        frac_agg, V_tot = fraccion_agregado(info, codigos)

        escala_C = np.abs(C).max()
        dif_vias = (float(np.abs(C - C_fuer).max() / escala_C)
                    if escala_C > 0 else 0.0)
        dif_vol = (abs(V_tot - V_bbox) / V_bbox) if V_bbox > 0 else 0.0

        meta = {
            "malla": nombre_malla,
            "modo": modo,
            "n_nodos": len(nodos),
            "n_aristas": len(aristas),
            "n_caras": len(caras),
            "n_elementos": len(elementos),
            "n_dofs": 3 * len(nodos),
            "n_borde": int(borde_mask.sum()),
            "vol_total": V_tot,
            "vol_bbox": V_bbox,
            "dif_volumen": dif_vol,
            "Lx": Lx, "Ly": Ly, "Lz": Lz,
            "frac_agg": frac_agg,
            "E_eq": E_eq, "nu_eq": nu_eq,
            "E_dir": E_dir, "nu_dir": nu_dir,
            "anisotropia": anisotropia(C),
            "dif_vias": dif_vias,
            "tol_C": tol_C,
            "t_lectura": t_lectura,
            "t_infocaras": t_infocaras,
            "t_ensamblaje": t_ensamblaje,
            "t_solver": t_solver,
            "solver": info_solver["nombre"],
            "n_gdl_libres": info_solver["n_gdl_libres"],
            "solver_iters_max": info_solver["iters_max"],
            "solver_res_rel": info_solver["res_rel_max"],
            "volumenes": np.array([info[i]["V"] for i in range(len(info))]),
        }

        if modo == "homogeneo":
            Ct = C_isotropica_3D(E_HOM, NU_HOM)
            escala = np.abs(Ct).max()
            err_max = float(np.abs(C - Ct).max() / escala)
            meta["C_teorica"] = Ct
            meta["err_C_max"] = err_max
            meta["verif_ok"] = err_max <= tol_C

        meta["memoria_pico_gb"] = memoria_pico_gb()
        meta["tiempo_s"] = time.perf_counter() - t0
        guardar_resultados_malla(carpeta_salida, nodos, C, C_fuer,
                                 resultados, meta, guardar_campos)

        meta.pop("volumenes")       # no viaja entre procesos
        return {"ok": True, "malla": nombre_malla,
                "C": C, "C_fuerzas": C_fuer, "meta": meta,
                "err_patch": max(r["err_rel"] for r in resultados.values()),
                "stats": _estadisticas(resultados),
                "tiempo_s": meta["tiempo_s"]}

    except Exception as ex:
        import traceback
        tb = traceback.format_exc()
        try:
            os.makedirs(carpeta_salida, exist_ok=True)
            with open(os.path.join(carpeta_salida, "ERROR.txt"), "w",
                      encoding="utf-8") as f:
                f.write(tb)
        except Exception:
            pass
        return {"ok": False, "malla": nombre_malla, "error": str(ex),
                "traceback": tb,
                "tiempo_s": time.perf_counter() - t0}


def _estadisticas(resultados):
    """min/max/media/desviación de cada componente de esfuerzo, por caso."""
    stats = {}
    for nombre, r in resultados.items():
        for k, comp in enumerate(COMP_SIGMA):
            v = r["sig"][:, k]
            stats[f"{nombre}_{comp}_min"]  = float(v.min())
            stats[f"{nombre}_{comp}_max"]  = float(v.max())
            stats[f"{nombre}_{comp}_mean"] = float(v.mean())
            stats[f"{nombre}_{comp}_std"]  = float(v.std())
    return stats


# ══════════════════════════════════════════════════════════════════════════════
# 7. DESCUBRIMIENTO DE MALLAS
# ══════════════════════════════════════════════════════════════════════════════

def _tiene_nodos(carpeta):
    try:
        return any(n.lower().startswith("input_nodos")
                   for n in os.listdir(carpeta))
    except OSError:
        return False


# ══════════════════════════════════════════════════════════════════════════════
# 7b. MEMORIA Y PARALELISMO
# ══════════════════════════════════════════════════════════════════════════════

def memoria_pico_gb():
    """
    Pico de memoria residente del proceso actual, en GB. Devuelve 0.0 si no
    se puede medir. En Windows usa psutil (peak_wset) y en Linux/macOS el
    ru_maxrss de resource, cuya unidad cambia entre sistemas.
    """
    try:
        import psutil
        mi = psutil.Process().memory_info()
        if hasattr(mi, "peak_wset"):          # Windows
            return float(mi.peak_wset) / 1024**3
    except Exception:
        pass
    try:
        import resource
        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux reporta KiB; macOS, bytes.
        return float(r) / (1024**3 if sys.platform == "darwin" else 1024**2)
    except Exception:
        return 0.0


def memoria_disponible_gb():
    """RAM disponible ahora mismo, en GB. 0.0 si no se puede averiguar."""
    try:
        import psutil
        return psutil.virtual_memory().available / 1024**3
    except Exception:
        return 0.0


def nucleos_fisicos():
    """Nucleos fisicos (sin hyperthreading); cae a los logicos si no se sabe."""
    try:
        import psutil
        n = psutil.cpu_count(logical=False)
        if n:
            return int(n)
    except Exception:
        pass
    return os.cpu_count() or 1


def procesos_recomendados(pico_gb, n_tareas=None, fraccion=0.75):
    """
    Cuantos procesos caben, dado el pico de memoria medido en UNA malla.

    Se reserva (1 - fraccion) de la RAM disponible para el sistema y para
    que el pico real de alguna malla pueda superar al medido. El limite
    duro son los nucleos fisicos: por encima de eso no hay ganancia en este
    tipo de trabajo, que esta limitado por ancho de banda de memoria.

    Devuelve (n_procesos, explicacion).
    """
    cores = nucleos_fisicos()
    disp = memoria_disponible_gb()

    if pico_gb <= 0 or disp <= 0:
        return 1, ("no se pudo medir la memoria (instala psutil para que "
                   "esto funcione); se usa 1 proceso por seguridad")

    caben = int((disp * fraccion) // pico_gb)
    n = max(1, min(cores, caben))
    if n_tareas:
        n = max(1, min(n, int(n_tareas)))

    razon = (f"{disp:.1f} GB disponibles x {fraccion:.0%} / {pico_gb:.2f} GB "
             f"por malla = {caben} caben; nucleos fisicos = {cores}")
    if caben == 0:
        razon += "  (ni una sola malla cabe con holgura: cuidado)"
    return n, razon


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
    return (_contar("Input_nodos.txt", "input_nodos.txt"),
            _contar("input_elements.txt", "Input_elements.txt",
                    "input_elementos.txt"))


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


# ══════════════════════════════════════════════════════════════════════════════
# 8. SALIDAS GLOBALES
# ══════════════════════════════════════════════════════════════════════════════

def escribir_resumen_global(ruta, filas, modo, tiempo_total, tol_C=1e-6):
    """RESUMEN_GLOBAL.txt con una línea por malla."""
    L = []
    L.append("=" * 140)
    L.append(f"RESUMEN GLOBAL - CASO {modo.upper()} - VEM 3D")
    L.append("=" * 140)
    L.append("")

    if modo == "homogeneo":
        L.append(f"{'Malla':<24s}{'Estado':<10s}{'Tiempo':>10s}"
                 f"{'Nodos':>10s}{'Caras':>10s}{'Elem':>10s}"
                 f"{'Err C':>12s}{'Err patch':>12s}"
                 f"{'E eq':>15s}{'nu eq':>10s}{'Verif':>13s}")
        L.append("-" * 140)
        for f in filas:
            if not f["ok"]:
                L.append(f"{f['malla']:<24s}{'ERROR':<10s}"
                         f"{f['tiempo_s']:>9.1f}s")
                continue
            m = f["meta"]
            L.append(f"{f['malla']:<24s}{'OK':<10s}{m['tiempo_s']:>9.1f}s"
                     f"{m['n_nodos']:>10,d}{m['n_caras']:>10,d}"
                     f"{m['n_elementos']:>10,d}"
                     f"{m['err_C_max']:>12.2e}{f['err_patch']:>12.2e}"
                     f"{m['E_eq']:>15.4e}{m['nu_eq']:>10.4f}"
                     f"{('COINCIDE' if m['verif_ok'] else 'NO COINCIDE'):>13s}")
    else:
        L.append(f"{'Malla':<24s}{'Estado':<10s}{'Tiempo':>10s}"
                 f"{'Nodos':>10s}{'Caras':>10s}{'Elem':>10s}"
                 f"{'Agg(%)':>10s}{'E eq':>15s}{'nu eq':>10s}"
                 f"{'Dif vias':>12s}{'Anisotr':>12s}")
        L.append("-" * 140)
        for f in filas:
            if not f["ok"]:
                L.append(f"{f['malla']:<24s}{'ERROR':<10s}"
                         f"{f['tiempo_s']:>9.1f}s")
                continue
            m = f["meta"]
            L.append(f"{f['malla']:<24s}{'OK':<10s}{m['tiempo_s']:>9.1f}s"
                     f"{m['n_nodos']:>10,d}{m['n_caras']:>10,d}"
                     f"{m['n_elementos']:>10,d}"
                     f"{100*m['frac_agg']:>10.2f}{m['E_eq']:>15.4e}"
                     f"{m['nu_eq']:>10.4f}{m['dif_vias']:>12.2e}"
                     f"{m['anisotropia']:>12.2e}")

    ok   = [f for f in filas if f["ok"]]
    fail = [f for f in filas if not f["ok"]]
    L.append("")
    L.append("-" * 140)
    L.append(f"Mallas procesadas : {len(filas)}    OK: {len(ok)}    "
             f"ERROR: {len(fail)}")

    if ok:
        L.append(f"Peor err_patch    : "
                 f"{max(f['err_patch'] for f in ok):.3e}")
        L.append(f"Peor dif_vias     : "
                 f"{max(f['meta']['dif_vias'] for f in ok):.3e}")
        L.append(f"Peor dif_volumen  : "
                 f"{max(f['meta']['dif_volumen'] for f in ok):.3e}")

    if modo == "homogeneo" and ok:
        n_bien = sum(1 for f in ok if f["meta"]["verif_ok"])
        peor = max(f["meta"]["err_C_max"] for f in ok)
        L.append(f"Verificacion C    : {n_bien}/{len(ok)} coinciden "
                 f"(tolerancia {tol_C:.1e})")
        L.append(f"Peor error en C   : {peor:.3e}")

    if fail:
        L.append("")
        L.append("Mallas con error:")
        for f in fail:
            L.append(f"  {f['malla']}: {f.get('error', 'desconocido')}")

    L.append("")
    L.append(f"TIEMPO TOTAL: {tiempo_total:.1f}s ({tiempo_total/60:.1f} min)")
    L.append("=" * 140)

    with open(ruta, "w", encoding="utf-8") as f:
        f.write("\n".join(L))


def escribir_csv_global(ruta, filas, modo):
    """RESUMEN_PROPIEDADES.csv con una fila por malla."""
    campos = ["malla", "caso", "replica", "estado", "pct_real(%)",
              "n_nodos", "n_aristas", "n_caras", "n_elementos", "n_dofs",
              "n_borde", "n_gdl_libres", "solver", "solver_iters_max",
              "solver_res_rel", "memoria_pico_gb",
              "vol_total", "vol_bbox", "dif_volumen",
              "Lx", "Ly", "Lz"]
    campos += [f"C{i+1}{j+1}" for i in range(6) for j in range(6)]
    campos += [f"F{i+1}{j+1}" for i in range(6) for j in range(6)]
    campos += ["E_eq", "nu_eq", "E_dir", "nu_dir", "anisotropia",
               "err_patch", "dif_vias",
               "tiempo_s", "t_lectura", "t_infocaras", "t_ensamblaje",
               "t_solver"]
    if modo == "homogeneo":
        campos += ["C11_teo", "C12_teo", "C44_teo", "err_C_max", "verif_ok"]

    for nombre, *_ in CASOS_CARGA:
        for comp in COMP_SIGMA:
            for st in ["min", "max", "mean", "std"]:
                campos.append(f"{nombre}_{comp}_{st}")

    with open(ruta, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=campos)
        w.writeheader()
        for f in filas:
            caso, _, rep = f["malla"].partition("/")
            fila = {c: "" for c in campos}
            fila["malla"]    = f["malla"]
            fila["caso"]     = caso
            fila["replica"]  = rep
            fila["estado"]   = "OK" if f["ok"] else "ERROR"
            fila["tiempo_s"] = round(f.get("tiempo_s", 0.0), 2)

            if f["ok"]:
                m, C, Cf = f["meta"], f["C"], f["C_fuerzas"]
                fila["pct_real(%)"] = round(100 * m["frac_agg"], 4)
                for k in ["n_nodos", "n_aristas", "n_caras", "n_elementos",
                          "n_dofs", "n_borde", "n_gdl_libres", "solver",
                          "solver_iters_max"]:
                    fila[k] = m[k]
                fila["solver_res_rel"] = f"{m['solver_res_rel']:.3e}"
                fila["memoria_pico_gb"] = round(
                    m.get("memoria_pico_gb", 0.0), 3)
                for k in ["vol_total", "vol_bbox", "Lx", "Ly", "Lz"]:
                    fila[k] = f"{m[k]:.10e}"
                fila["dif_volumen"] = f"{m['dif_volumen']:.4e}"
                for i in range(6):
                    for j in range(6):
                        fila[f"C{i+1}{j+1}"] = f"{C[i, j]:.10e}"
                        fila[f"F{i+1}{j+1}"] = f"{Cf[i, j]:.10e}"
                fila["E_eq"]        = f"{m['E_eq']:.10e}"
                fila["nu_eq"]       = f"{m['nu_eq']:.8f}"
                fila["E_dir"]       = f"{m['E_dir']:.10e}"
                fila["nu_dir"]      = f"{m['nu_dir']:.8f}"
                fila["anisotropia"] = f"{m['anisotropia']:.4e}"
                fila["err_patch"]   = f"{f['err_patch']:.4e}"
                fila["dif_vias"]    = f"{m['dif_vias']:.4e}"
                for k in ["t_lectura", "t_infocaras", "t_ensamblaje",
                          "t_solver"]:
                    fila[k] = round(m[k], 2)
                if modo == "homogeneo":
                    Ct = m["C_teorica"]
                    fila["C11_teo"]   = f"{Ct[0, 0]:.10e}"
                    fila["C12_teo"]   = f"{Ct[0, 1]:.10e}"
                    fila["C44_teo"]   = f"{Ct[3, 3]:.10e}"
                    fila["err_C_max"] = f"{m['err_C_max']:.4e}"
                    fila["verif_ok"]  = int(m["verif_ok"])
                for k, v in f["stats"].items():
                    fila[k] = f"{v:.10e}"
            w.writerow(fila)
