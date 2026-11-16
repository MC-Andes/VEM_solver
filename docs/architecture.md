# Arquitectura y migración

Estructura basada en `MC-Andes/template-python-mech` (commit
`9f3de3aaab5c69f23721f2f4fd47f3c8f2a42db8`) y `MC-Andes/standards`
(`b1bfa6317a60a52febe268056944933ec9a1e708`).

```text
src/vem3d/
  mesh.py             lectura, propiedades y descubrimiento de mallas
  boundary.py         GDL, detección de borde y campos afines
  geometry.py         geometría y orientación de caras/poliedros
  element.py          proyectores y rigidez VEM de orden 1
  assembly.py         ensamblaje global/particionado, CSR y temporales
  solvers.py          selección LU/AMG, factorización y convergencia
  materials.py        tensor isotrópico y ajuste de E, nu
  homogenization.py   seis cargas, esfuerzos y reacciones
  results.py          escritura de campos, resúmenes y CSV
  resources.py        medidas de memoria y procesos
  workflow.py         coordinación de una malla
  cli/                una malla y lotes homogéneos/heterogéneos
  constants.py        convenciones y parámetros físicos comunes
  compat.py           API histórica de transición
examples/             ejemplo analítico reproducible
tests/                regresiones numéricas y de índices
jobs/                 trabajos Slurm preparados para Hypatia
docs/                 formulación, arquitectura, revisión y uso
reference/original/   fuentes recibidas y SHA256, sólo referencia
```

Las dependencias van de la CLI al flujo de trabajo y de éste a módulos
numéricos; el elemento no conoce los lotes ni la salida. Se conservaron nombres
matemáticos y funciones para facilitar la comparación con el código recibido.
Los batch siguen separados para conservar sus opciones y formatos históricos.

Los cuatro nombres históricos en la raíz siguen funcionando como adaptadores.
`import vem3d_core` reexporta la API; para código nuevo conviene importar desde
el módulo correspondiente. Cambiar una constante reexportada no reconfigura su
módulo propietario: las constantes se configuran en `solvers` o `assembly`.
La nueva CLI individual escribe resultados en `--salida` y no junto al código.

## Estado de preparación para GitHub

No se ha publicado ni elegido licencia/autores en nombre de los investigadores.
Antes del primer release deben confirmarse licencia, autores, DOI/CITATION y
CODEOWNERS. No hay resultados inventados ni un DOI provisional.
El alcance de tipado se mantiene como deuda de migración; no se anuncia mypy
estricto. La validación de esta migración se ejecutó en Hypatia. Se aplica el
umbral de cobertura de la plantilla (70 %); la verificación final 803351
alcanzó 84.23 %, incluyendo reportes y batch.
