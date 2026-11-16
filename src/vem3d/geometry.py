"""VEM 3D: geometry."""

import numpy as np

from .constants import R_mat


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
        E += x[i - 1] * y[i] - y[i - 1] * x[i]
        xe += (x[i - 1] + x[i]) * (x[i - 1] * y[i] - x[i] * y[i - 1])
        ye += (y[i - 1] + y[i]) * (x[i - 1] * y[i] - x[i] * y[i - 1])
    E /= 2.0
    ye /= 6 * E
    xe /= 6 * E

    he = max(
        (
            ((x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2) ** 0.5
            for i in range(ngdl)
            for j in range(ngdl)
        ),
        default=0.0,
    )
    centroideglobal = p0 + xe * u + ye * v

    D = monomios2D(x, y, xe, ye, he).T
    G = np.zeros((3, 3))
    for i in range(1, 3):
        G[i, i] = E / he**2
    G[0] = np.mean(D, axis=0)

    dm = np.array([[0, 0], [1 / he, 0], [0, 1 / he]])
    B = np.zeros((3, ngdl))
    for j in range(ngdl):
        x0, y0, x1, y1 = x[j - 1], y[j - 1], x[j], y[j]
        x2, y2 = (x[0], y[0]) if j + 1 == ngdl else (x[j + 1], y[j + 1])

        nant = R_mat @ np.array([x1 - x0, y1 - y0])
        nant = nant / np.linalg.norm(nant)
        Lant = np.hypot(x1 - x0, y1 - y0)

        nsig = R_mat @ np.array([x2 - x1, y2 - y1])
        nsig = nsig / np.linalg.norm(nsig)
        Lsig = np.hypot(x2 - x1, y2 - y1)

        for i in range(3):
            B[i, j] += np.dot(nant, dm[i]) * Lant / 2 + np.dot(nsig, dm[i]) * Lsig / 2
    B[0] = 1 / ngdl
    piast = np.linalg.solve(G, B)

    return {
        "node_ids": np.asarray(idnodos_c, dtype=np.int64),
        "gdlx": gdlx,
        "gdly": gdly,
        "gdlz": gdlz,
        "x": x,
        "y": y,
        "E": E,
        "xe": xe,
        "ye": ye,
        "he": he,
        "n": n,
        "cglobal": centroideglobal,
        "aristas": idaristas,
        "piast": piast,
    }


def construir_infocaras(nodos, aristas, caras):
    return [infocara(i, nodos, aristas, caras) for i in range(len(caras))]


def _copiar_cara(c):
    """Copia superficial con los campos que `orientar_cara` puede invertir."""
    return {
        "node_ids": c["node_ids"].copy(),
        "gdlx": c["gdlx"].copy(),
        "gdly": c["gdly"].copy(),
        "gdlz": c["gdlz"].copy(),
        "E": c["E"],
        "he": c["he"],
        "n": c["n"].copy(),
        "cglobal": c["cglobal"],
        "aristas": c["aristas"],
        "piast": c["piast"].copy(),
    }


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
                            if infocaras_e[k]["node_ids"][m] == nodoa:
                                if infocaras_e[k]["node_ids"][m - 1] == nodob:
                                    orientacion1 = 1
                        for j in range(len(infocaras_e)):
                            if a in infocaras_e[j]["aristas"]:
                                if j not in carascompletadas:
                                    carascompletadas.append(j)
                                    for i in range(len(infocaras_e[j]["gdlx"])):
                                        if infocaras_e[j]["node_ids"][i] == nodoa:
                                            if infocaras_e[j]["node_ids"][i - 1] == nodob:
                                                orientacion1 += 1
                                            if orientacion1 == 0 or orientacion1 == 2:
                                                infocaras_e[j]["node_ids"] = infocaras_e[j][
                                                    "node_ids"
                                                ][::-1]
                                                infocaras_e[j]["gdlx"] = infocaras_e[j]["gdlx"][
                                                    ::-1
                                                ]
                                                infocaras_e[j]["gdly"] = infocaras_e[j]["gdly"][
                                                    ::-1
                                                ]
                                                infocaras_e[j]["gdlz"] = infocaras_e[j]["gdlz"][
                                                    ::-1
                                                ]
                                                infocaras_e[j]["piast"] = infocaras_e[j]["piast"][
                                                    :, ::-1
                                                ]
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
            cara["node_ids"] = cara["node_ids"][::-1]
            cara["gdlx"] = cara["gdlx"][::-1]
            cara["gdly"] = cara["gdly"][::-1]
            cara["gdlz"] = cara["gdlz"][::-1]
            cara["piast"] = cara["piast"][:, ::-1]
            cara["n"] *= -1

    return V, C, infocaras_e
