"""VEM 3D: element."""

import numpy as np

from .geometry import divergencia, nodos_e


def monomios3D(x, y, z, centroide, he):
    xi = (x - centroide[0]) / he
    eta = (y - centroide[1]) / he
    citha = (z - centroide[2]) / he
    monomio = np.array(
        [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [-eta, xi, 0],
            [citha, 0, -xi],
            [0, -citha, eta],
            [xi, 0, 0],
            [0, eta, 0],
            [0, 0, citha],
            [eta, xi, 0],
            [citha, 0, xi],
            [0, citha, eta],
        ]
    )
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
    C = np.array(
        [
            [C1, C2, C2, 0, 0, 0],
            [C2, C1, C2, 0, 0, 0],
            [C2, C2, C1, 0, 0, 0],
            [0, 0, 0, C3, 0, 0],
            [0, 0, 0, 0, C3, 0],
            [0, 0, 0, 0, 0, C3],
        ]
    )

    nodo_e = nodos_e(ne, nodos, aristas, caras, elementos)
    gdlx = nodo_e[:, 0].astype(int)
    gdly = nodo_e[:, 1].astype(int)
    gdlz = nodo_e[:, 2].astype(int)
    gdl = np.concatenate([gdlx, gdly, gdlz])

    x = nodo_e[:, 3]
    y = nodo_e[:, 4]
    z = nodo_e[:, 5]

    V, centroide, infocaras_e = divergencia(ne, aristas, caras, elementos, infocaras)
    ngdl = len(x)

    coords = np.stack([x, y, z], axis=1)
    diffs = coords[:, None, :] - coords[None, :, :]
    he = np.sqrt((diffs**2).sum(axis=2)).max()

    D = np.zeros((3 * ngdl, 12))
    for i in range(ngdl):
        mnodo = monomios3D(x[i], y[i], z[i], centroide, he)
        D[i, :] = mnodo[0]
        D[ngdl + i, :] = mnodo[1]
        D[2 * ngdl + i, :] = mnodo[2]

    e11 = 1 / he * np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
    e12 = 1 / he * np.array([0, 0, 0, -1, 0, 0, 0, 0, 0, 1, 0, 0])
    e13 = 1 / he * np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0])
    e21 = 1 / he * np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0])
    e22 = 1 / he * np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
    e23 = 1 / he * np.array([0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 1])
    e31 = 1 / he * np.array([0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 1, 0])
    e32 = 1 / he * np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1])
    e33 = 1 / he * np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0])

    # La deformación tiene seis componentes: mismo producto energético,
    # evaluado por BLAS en vez de 144 iteraciones de Python.
    dpi = np.array([e11, e22, e33, e12 + e21, e13 + e31, e23 + e32])
    G = V * (dpi.T @ C @ dpi)

    B = np.zeros((12, ngdl * 3))
    Bs = np.zeros((12, ngdl * 3))
    for k in range(len(infocaras_e)):
        gdlglob = infocaras_e[k]["gdlx"]
        piastcara = infocaras_e[k]["piast"]
        intvh = infocaras_e[k]["E"] * np.array([1, 0, 0]) @ piastcara
        gdlcara = np.array([np.where(gdlx == g)[0][0] for g in gdlglob], dtype=int)
        n = infocaras_e[k]["n"]
        N = np.array(
            [
                [n[0], 0, 0],
                [0, n[1], 0],
                [0, 0, n[2]],
                [n[1], n[0], 0],
                [n[2], 0, n[0]],
                [0, n[2], n[1]],
            ]
        )

        Bs[0, gdlcara] += intvh
        Bs[1, gdlcara + ngdl] += intvh
        Bs[2, gdlcara + ngdl * 2] += intvh
        Bs[3, gdlcara + ngdl] += n[0] * intvh
        Bs[3, gdlcara] += -n[1] * intvh
        Bs[4, gdlcara] += n[2] * intvh
        Bs[4, gdlcara + ngdl * 2] += -n[0] * intvh
        Bs[5, gdlcara + ngdl * 2] += n[1] * intvh
        Bs[5, gdlcara + ngdl] += -n[2] * intvh

        traction = dpi.T @ C @ N
        B[:, gdlcara] += traction[:, 0, None] * intvh
        B[:, gdlcara + ngdl] += traction[:, 1, None] * intvh
        B[:, gdlcara + 2 * ngdl] += traction[:, 2, None] * intvh

    Bs[6:, :] = B[6:, :]

    Gs = Bs @ D
    piast = np.linalg.solve(Gs, Bs)
    pi = D @ piast

    Kc = piast.T @ (G @ piast)
    alpha = np.trace(Kc) / ngdl
    # (I-DP)^T(I-DP) = I-DP-(DP)^T + P^T(D^TD)P.
    # D tiene sólo 12 columnas. Se evita un producto dxd por dxd:
    # coste O(12*d^2) en vez de O(d^3), d = 3*nodos del elemento.
    stabilization = piast.T @ (D.T @ D) @ piast
    stabilization -= pi
    stabilization -= pi.T
    stabilization.flat[:: 3 * ngdl + 1] += 1.0
    K = Kc + alpha * stabilization

    info[ne] = {
        "dpi": dpi,
        "piast": piast,
        "C": C,
        "gdl": gdl,
        "V": V,
        "he": he,
        "centroide": centroide,
    }
    return K, gdl
