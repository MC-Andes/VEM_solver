# VEM 3D para homogenización elástica

El paquete resuelve elasticidad lineal sin fuerzas de cuerpo sobre dominios
prismáticos. Impone seis campos afines en el contorno (KUBC) y obtiene una
matriz efectiva de 6 × 6. Materiales isotrópicos por elemento, pequeñas
deformaciones, VEM conforme de primer orden y estabilización por traza.

La matriz se calcula mediante promedio volumétrico de esfuerzos y mediante
reacciones del contorno. La coincidencia de ambas vías, el balance volumétrico,
el residuo del solver y el patch test homogéneo son comprobaciones distintas.
Ninguna por sí sola demuestra convergencia de malla para el paper.

## Convenciones

- Coordenadas y módulos deben tener unidades consistentes. Se recomienda m, Pa.
- Voigt: `[xx, yy, zz, xy, xz, yz]`; deformaciones cortantes ingenieriles.
- Las cargas `gxy=1`, `gxz=1`, `gyz=1` usan componentes tensoriales 0.5.
- IDs y referencias de nodos, aristas y caras comienzan en cero y corresponden
  a posiciones de fila; los GDL forman una permutación de `0..3*N-1`.
- Material 0 es matriz; cualquier otro código se contabiliza como agregado.
- El contorno se detecta sobre las seis caras de la caja envolvente: esta
  implementación no prescribe KUBC sobre dominios de forma arbitraria.

## Entrada

Un directorio contiene:

| Archivo | Columnas |
| --- | --- |
| `Input_nodos.txt` | gdl_x gdl_y gdl_z x y z; BC opcionales se regeneran |
| `Input_aristas.txt` | id nodo_i nodo_j |
| `Input_caras.txt` | id arista_1 arista_2 ... en orden alrededor de la cara |
| `input_elements.txt` | id material material2 cara_1,cara_2,... o separado por espacios |
| `Propiedades.txt` | material_id E nu |

Para el modo homogéneo se sustituyen materiales por E=1e6 y nu=0.3.
El modo heterogéneo conserva las propiedades de la malla.

## Salida

`C_equivalente.txt`, `C_equivalente_fuerzas.txt`, `resumen.txt`, opcionalmente
campos de desplazamiento/deformación/esfuerzo. El comando `vem3d-run` añade
`run.json` con versiones, equipo, parámetros, SHA256 de las entradas y resultado.
Un fallo produce código de salida no nulo y `ERROR.txt`. El código 2 del comando
individual indica que el caso homogéneo terminó pero no pasó la tolerancia.
Los batch históricos conservan sus CSV y resúmenes.

Consulta [arquitectura](architecture.md), [Hypatia](hypatia.md) y
[revisión técnica](review.md).
