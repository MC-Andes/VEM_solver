"""VEM 3D: assembly."""

import os
import shutil
import tempfile

import numpy as np
import scipy.sparse as sp

from .boundary import indices_gdl
from .element import elemento_VEM3D

MAX_ENTRADAS_BLOQUE = 12_000_000
ENTRADAS_BUFFER = 4_000_000
FILAS_POR_FRANJA = 1_500_000


def csr_offset(indptr, offset):
    """Suma desplazamientos CSR en int64 ANTES de asignar el resultado.

    El número de no ceros puede exceder 2**31 aun con menos filas/columnas.
    Se valida también el límite int64 para evitar desbordamiento silencioso.
    """
    offset = int(offset)
    if offset < 0 or offset > np.iinfo(np.int64).max:
        raise OverflowError("Desplazamiento CSR fuera de int64")
    pointers = np.asarray(indptr, dtype=np.int64)
    if pointers.size and (
        pointers.min() < 0 or int(pointers.max()) > np.iinfo(np.int64).max - offset
    ):
        raise OverflowError("Punteros CSR fuera de int64")
    return pointers + np.int64(offset)


def _acumular(K, filas, cols, vals, ndof):
    """Pasa las tripletas acumuladas a disperso y las suma a K."""
    if not vals:
        return K
    bloque = sp.coo_matrix(
        (np.concatenate(vals), (np.concatenate(filas), np.concatenate(cols))), shape=(ndof, ndof)
    ).tocsr()
    return bloque if K is None else (K + bloque)


def ensamblar(nodos, aristas, caras, elementos, infocaras, info, max_entradas=MAX_ENTRADAS_BLOQUE):
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
    if ndof > np.iinfo(np.int32).max:
        raise ValueError(
            "Las tripletas requieren menos de 2**31 grados de libertad; nnz sí admite int64"
        )
    nelem = len(elementos)

    K = None
    filas, cols, vals = [], [], []
    acumuladas = 0
    for n in range(nelem):
        Ke, gdl = elemento_VEM3D(n, nodos, aristas, caras, elementos, infocaras, info)
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
        self._f_i.close()
        self._f_j.close()
        self._f_v.close()

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
        M = sp.coo_matrix((mv, (mi - self.fila0, mj)), shape=(self.nfilas, ncols)).tocsr()
        del mi, mj, mv
        M.sum_duplicates()
        return M


def ensamblar_particionado(
    nodos,
    aristas,
    caras,
    elementos,
    infocaras,
    info,
    borde_mask,
    base_tmp=None,
    en_disco=True,
    compact_info=False,
):
    """
    Ensambla directamente el sistema particionado por las condiciones KUBC.

    Devuelve el mismo diccionario que `factorizar` espera recibir ya armado:
        {"K_ll", "K_lf", "K_f", "idx_l", "idx_f", "ndof"}

    `base_tmp` es la carpeta DENTRO de la cual se crea el directorio de
    trabajo (por defecto, el temporal del sistema). Necesita espacio en
    disco del orden de 16 bytes por entrada elemental, sum_e (3*n_e)^2, mas
    12–16 bytes por no-cero de K_ll más punteros para los arreglos definitivos.
    """
    ndof = 3 * len(nodos)
    if ndof > np.iinfo(np.int32).max:
        raise ValueError(
            "Las tripletas requieren menos de 2**31 grados de libertad; nnz sí admite int64"
        )
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
    franjas = []
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
            franjas.append(_Franja(carpeta_tmp, k, f0, min(filas_franja, n_l - f0)))

        lf_i, lf_j, lf_v = [], [], []
        ff_i, ff_j, ff_v = [], [], []

        for n in range(len(elementos)):
            Ke, gdl = elemento_VEM3D(n, nodos, aristas, caras, elementos, infocaras, info)
            if compact_info:
                record = info[n]
                record["strain"] = record.pop("dpi") @ record.pop("piast")
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
        # Elegir índices de salida compatibles con punteros globales grandes.
        index_dtype = np.int64 if sum(fr.n for fr in franjas) > np.iinfo(np.int32).max else np.int32
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
                b.indices.astype(index_dtype, copy=False).tofile(fi)
                indptr[fila + 1 : fila + 1 + b.shape[0]] = csr_offset(b.indptr[1:], total_nnz)
                total_nnz += int(b.nnz)
                fila += b.shape[0]
                fr.borrar()
                del b
        franjas.clear()

        if en_disco and total_nnz:
            datos = np.memmap(ruta_d, dtype=np.float64, mode="r", shape=(total_nnz,))
            indices = np.memmap(ruta_i, dtype=index_dtype, mode="r", shape=(total_nnz,))
        else:
            datos = np.fromfile(ruta_d, dtype=np.float64)
            indices = np.fromfile(ruta_i, dtype=index_dtype)
            os.remove(ruta_d)
            os.remove(ruta_i)

        K_ll = sp.csr_matrix((datos, indices, indptr), shape=(n_l, n_l))

        K_lf = _coo_de_listas(lf_i, lf_j, lf_v, (n_l, n_f))
        K_f = _coo_de_listas(ff_i, ff_j, ff_v, (n_f, ndof))

        ok = True
        return {
            "K_ll": K_ll,
            "K_lf": K_lf,
            "K_f": K_f,
            "idx_l": idx_l,
            "idx_f": idx_f,
            "ndof": ndof,
            "carpeta_tmp": carpeta_tmp if en_disco else None,
        }

    finally:
        for fr in franjas:
            for handle in (fr._f_i, fr._f_j, fr._f_v):
                if not handle.closed:
                    handle.close()
        # La carpeta solo se borra aqui si algo fallo o si no quedo nada
        # mapeado a ella; si K_ll vive en disco, la borra el llamador
        # cuando suelta el sistema (`liberar_sistema`).
        if propia and (not ok or not en_disco):
            shutil.rmtree(carpeta_tmp, ignore_errors=True)


def _coo_de_listas(li, lj, lv, forma):
    """CSR a partir de listas de tripletas (bloques chicos: K_lf y K_f)."""
    if not li:
        return sp.csr_matrix(forma)
    M = sp.coo_matrix(
        (
            np.concatenate(lv),
            (np.concatenate(li).astype(np.int32), np.concatenate(lj).astype(np.int32)),
        ),
        shape=forma,
    ).tocsr()
    li.clear()
    lj.clear()
    lv.clear()
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
