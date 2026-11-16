# VEM y FEM para elasticidad 2D y 3D

Este programa permite estimar las propiedades elásticas de un material formado
por varios componentes. Por ejemplo, puede representar un pequeño volumen de
concreto con una matriz y agregados, aplicar distintas deformaciones y calcular
cómo responde el conjunto. El resultado principal es una matriz de rigidez
**C**, que relaciona las deformaciones con los esfuerzos medios del material.
También se calculan un módulo de Young y un coeficiente de Poisson equivalentes.

Puedes realizar el cálculo con el **Método de Elementos Virtuales** (VEM),
en polígonos 2D o poliedros 3D, o con **elementos finitos** (FEM) mediante
FEniCSx, en triángulos 2D o tetraedros 3D.
La geometría se divide en pequeños elementos, que en conjunto forman
una *malla*. El programa lee esa malla y las propiedades de sus materiales;
no genera automáticamente la geometría de una muestra real. Sí incluye un
pequeño ejemplo que permite comprobar la instalación sin disponer de una malla
propia.

El modelo considera elasticidad lineal y pequeñas deformaciones. En el borde
exterior prescribe desplazamientos uniformes: dos extensiones y un corte en
2D, o tres extensiones y tres cortes en 3D. Esta condición se conoce como
**KUBC**. La implementación actual supone que el dominio completo tiene forma
de rectángulo en 2D o caja rectangular en 3D, sin huecos interiores.
No incluye cargas de volumen, plasticidad ni efectos dinámicos.

## Instalar el programa y ejecutar un ejemplo 3D

Necesitas Python 3.11 o posterior. Puedes usar el programa en tu propio equipo;
no requiere un clúster ni acceso a un servidor específico. Las mallas grandes
sí pueden necesitar mucha memoria y espacio en disco, por lo que conviene
comenzar con el ejemplo pequeño.

Descarga el repositorio desde GitHub con **Code → Download ZIP** y descomprímelo.
Abre una terminal en la carpeta que contiene este README. Crea un entorno para
instalar las dependencias del programa sin mezclarlas con otros proyectos:

```bash
python -m venv .venv
```

Actívalo con el comando correspondiente a tu sistema. En Windows, desde
el Símbolo del sistema (CMD):

```bat
.venv\Scripts\activate.bat
```

En Linux o macOS:

```bash
source .venv/bin/activate
```

Si tu equipo reconoce Python como `python3`, usa ese nombre en lugar de
`python` al crear el entorno. Una vez activado, instala el programa:

```bash
python -m pip install .
```

Este paso instala también las bibliotecas de cálculo necesarias. Entre ellas
está PyAMG 5.4.1 o posterior, requerido para trabajar con índices de matrices
grandes. No necesitas instalarlas una por una.

Ahora genera la malla del ejemplo. Es un cubo dividido en ocho cubos más
pequeños, con 27 nodos:

```bash
python examples/cube.py ejemplo/VEM
```

Ejecuta la comprobación asignando el mismo material a todos los elementos:

```bash
vem3d-run ejemplo/VEM --modo homogeneo --solver lu --sin-campos --salida resultados/ejemplo
```

`--modo homogeneo` usa un módulo de Young de 1 000 000 y un coeficiente de
Poisson de 0,3 en toda la malla. Como se conoce la respuesta teórica de ese
material, este caso sirve para comprobar el cálculo. `--solver lu` elige un
método de solución apropiado para este ejemplo pequeño. `--sin-campos` guarda
las propiedades y los resúmenes, sin generar los archivos completos de
desplazamientos y esfuerzos. `--salida` indica dónde quedarán los resultados.

Al terminar, la terminal mostrará los valores equivalentes de **E** y **nu**,
que deberían estar muy próximos a 1 000 000 y 0,3. También mostrará `err_C`,
el error relativo de la matriz calculada frente a la respuesta teórica. En
este ejemplo debe ser muy pequeño y menor que la tolerancia predeterminada
de 0,000001. Si la comprobación falla, el comando devuelve un estado de error.

## Usar una malla 3D propia

Una malla describe los puntos de la geometría —los *nodos*—, las aristas que
los unen, las caras y los elementos que forman el volumen. El programa espera
estos cinco archivos de texto en una misma carpeta:

| Archivo | Qué contiene |
| --- | --- |
| `Input_nodos.txt` | Las coordenadas de los nodos y los índices de sus desplazamientos. |
| `Input_aristas.txt` | Los dos nodos que forman cada arista. |
| `Input_caras.txt` | Las aristas que rodean cada cara. |
| `input_elements.txt` | Las caras de cada elemento y su código de material. |
| `Propiedades.txt` | El módulo de Young y el coeficiente de Poisson de cada material. |

Cada línea representa un nodo, arista, cara, elemento o material. Usa las
siguientes columnas, separadas por espacios y sin encabezado:

```text
Input_nodos.txt:    gdl_x gdl_y gdl_z x y z
Input_aristas.txt:  id nodo_i nodo_j
Input_caras.txt:    id arista_1 arista_2 ...
input_elements.txt: id material material2 cara_1 cara_2 ...
Propiedades.txt:    material E nu
```

Los grados de libertad (`gdl_x`, `gdl_y`, `gdl_z`) identifican los tres
desplazamientos de cada nodo. Entre todos los nodos deben aparecer los números
de 0 a 3N−1, sin repetir, donde N es el número de nodos. Los IDs de nodos,
aristas, caras y elementos comienzan en cero y coinciden con su posición de
fila; en el archivo de nodos esa posición es su ID implícito. Las aristas de
cada cara deben seguir el orden alrededor de ella. `material2` se conserva
por el formato de entrada, pero no se usa: puedes poner cero. El código de
`material` debe existir en `Propiedades.txt`; el código cero representa la
matriz y los demás se cuentan como agregados.

No basta con cambiar el nombre de archivos procedentes de otro programa.
Las unidades deben ser consistentes; se recomienda usar metros para las
coordenadas y pascales para los módulos elásticos. Puedes usar los archivos
generados por el ejemplo como referencia del formato.

Primero puedes comprobar tu malla con `--modo homogeneo`, igual que en el
ejemplo anterior. Para calcular después la respuesta con sus materiales
reales, cambia a `--modo heterogeneo`. Por ejemplo, si los archivos están en
`mis_mallas/muestra_01/VEM`:

```bash
vem3d-run mis_mallas/muestra_01/VEM --modo heterogeneo --solver amg --sin-campos --salida resultados/muestra_01
```

Aquí `amg` utiliza un método iterativo pensado para sistemas más grandes.
En el modo heterogéneo, el programa lee las propiedades de `Propiedades.txt`.
Si una ruta contiene espacios, escríbela entre comillas. Puedes consultar todas
las opciones con `vem3d-run --help`.

## Entender los resultados

Abre `resumen.txt` en la carpeta de salida para ver las dimensiones del problema,
las propiedades equivalentes, los tiempos y las comprobaciones numéricas.
`C_equivalente.txt` contiene la matriz C calculada a partir de los esfuerzos
medios. `C_equivalente_fuerzas.txt` contiene otra estimación obtenida a partir
de las reacciones en el borde. Ambas deberían ser próximas; su diferencia
relativa aparece como `dif_vias` y ayuda a detectar problemas en el cálculo.

El archivo `run.json` registra las versiones de las bibliotecas, los parámetros,
las huellas de los archivos de entrada y los resultados. Conviene conservarlo
junto con la malla para poder reproducir una ejecución. Si ocurre un fallo
durante el cálculo, `ERROR.txt` guarda el detalle técnico.

Si necesitas visualizar desplazamientos, deformaciones o esfuerzos, repite el
comando sin `--sin-campos`. Se generarán archivos adicionales para cada caso
de carga, que pueden ocupar bastante espacio en mallas grandes.

En un material heterogéneo, los valores equivalentes de E y nu se obtienen
ajustando una respuesta isotrópica a la matriz C. Si el material responde de
forma muy distinta según la dirección, esos dos valores no describen toda su
respuesta: conviene analizar la matriz completa y la medida de anisotropía
del resumen. Pasar la comprobación homogénea tampoco sustituye un estudio de
convergencia al refinar la malla.


Las filas y columnas de C siguen el orden `xx, yy, zz, xy, xz, yz`. Las
componentes de deformación de corte son ingenieriles: por ejemplo, `gxy=1`
corresponde a una componente tensorial de 0,5.

## Ejecutar varias mallas VEM 3D

Para procesar varias mallas en lote, organízalas en carpetas de caso y
réplica. Dentro de cada réplica, la carpeta `VEM` contiene los cinco archivos
de entrada:

```text
mis_mallas/
  caso_01/
    caso_01_R01/
      VEM/
    caso_01_R02/
      VEM/
```

Primero consulta cuáles se han encontrado:

```bash
vem3d-batch-homogeneous --raiz mis_mallas --listar
```

Para comprobarlas con el material uniforme o calcular sus materiales reales:

```bash
vem3d-batch-homogeneous --raiz mis_mallas --procesos 1 --sin-campos
vem3d-batch-heterogeneous --raiz mis_mallas --procesos 1 --sin-campos
```

Los resultados se guardan bajo esa raíz en `Resultados_Homogeneos` o
`Resultados_Heterogeneos`. Empieza con un solo proceso: cada malla necesita su
propia memoria. Usa `--help` en cualquiera de los comandos para consultar
sus opciones.


## Trabajar en 2D con VEM

La misma instalación incluye `vem2d-run`. Para empezar, genera un cuadrado
dividido en cuatro elementos y resuélvelo con material uniforme:

```bash
python examples/square.py ejemplo_2d/VEM
vem2d-run ejemplo_2d/VEM --modo homogeneo --solver lu --sin-campos --salida resultados/ejemplo_2d
```

También aquí debes obtener E próximo a 1 000 000 y nu próximo a 0,3. La matriz
C tiene ahora tres filas y tres columnas, en el orden `xx, yy, xy`, y relaciona
los esfuerzos con `exx, eyy, gamma_xy`. Se considera espesor unitario.

Por defecto se usa **tensión plana**, apropiada para idealizar una lámina
delgada sin esfuerzos fuera del plano. Si el desplazamiento fuera del plano
está restringido, elige **deformación plana** con
`--hipotesis deformacion-plana`. Son modelos diferentes: la elección depende
del problema físico, no del método de solución.

Para una malla propia, el formato 2D del generador MC-Andes utiliza sólo tres
archivos. Cada elemento enumera sus nodos alrededor del perímetro, en sentido
horario o antihorario:

```text
Input_nodos.txt:    gdl_x gdl_y x y
input_elements.txt: id material material2 nodo_1 nodo_2 nodo_3 ...
Propiedades.txt:    material E nu
```

Los nodos de cada polígono pueden separarse por espacios o por comas. Los IDs
comienzan en cero y coinciden con su posición de fila. Los grados de libertad
deben cubrir de 0 a 2N−1 sin repeticiones. Las columnas adicionales de condiciones
de borde de `Input_nodos.txt` se ignoran: el programa prescribe KUBC.
`material2` se conserva por compatibilidad con el formato y no se utiliza.
No necesitas archivos de aristas ni de caras. Los polígonos deben ser simples
y formar una malla conforme del rectángulo: las interfaces deben compartir
los mismos nodos y aristas, sin superposiciones ni huecos.

Para calcular los materiales reales:

```bash
vem2d-run mis_mallas_2d/muestra/VEM --modo heterogeneo --hipotesis tension-plana --solver amg --sin-campos --salida resultados/muestra_2d
```

Las salidas tienen los mismos nombres que en 3D: `resumen.txt`, las dos
matrices C y `run.json`. Sin `--sin-campos`, se guardan también desplazamientos,
deformaciones y esfuerzos para las tres cargas. Usa una carpeta de salida
vacía para no mezclar ejecuciones. En 2D el ensamblaje se guarda en RAM;
no utiliza el ensamblaje por franjas en disco del solver 3D.

Para varias mallas organizadas como `raíz/caso/réplica/VEM`, usa
`vem2d-batch-homogeneous --raiz mis_mallas_2d --sin-campos` o
`vem2d-batch-heterogeneous --raiz mis_mallas_2d --sin-campos`. Estos comandos
procesan una malla a la vez y aceptan `--hipotesis`, `--solver`, `--salida`
y `--listar`.

## Resolver el mismo problema con FEM y FEniCSx

El comando `fem-run` aplica las mismas cargas y materiales mediante elementos
finitos lineales. Usa las mallas triangulares o tetraédricas `.msh` que genera
Gmsh, como las que el generador MC-Andes guarda en
`FEM_physical/gruesa/mesh.msh`. No convierte los polígonos o poliedros VEM:
lee la malla FEM disponible para ese caso.

FEniCSx requiere un entorno adicional. Puedes crearlo con Conda en Linux o
macOS; en Windows usa un entorno Linux mediante WSL. Desde la carpeta del
repositorio, una instalación con la versión 0.9 usada para verificar el solver es:

```bash
conda create -n fem-vem -c conda-forge python=3.12 fenics-dolfinx=0.9.0 mpich pip
conda activate fem-vem
python -m pip install . gmsh==4.14.1
```

La [guía oficial de instalación de FEniCSx](https://fenicsproject.org/download/)
explica otras opciones. `fem-run` utiliza PETSc con escalares reales y, en esta
versión, un solo proceso MPI: ejecútalo directamente, sin `mpirun`.
No necesitas este entorno adicional para usar únicamente VEM.

Por ejemplo, para un caso 2D:

```bash
fem-run muestra_2d/FEM_physical/gruesa/mesh.msh --propiedades muestra_2d/Propiedades.txt --dimension 2 --hipotesis tension-plana --modo heterogeneo --sin-campos --salida resultados/fem_2d
```

Y para uno 3D:

```bash
fem-run muestra_3d/FEM_physical/gruesa/mesh.msh --propiedades muestra_3d/Propiedades.txt --dimension 3 --modo heterogeneo --sin-campos --salida resultados/fem_3d
```

En 3D se usa elasticidad tridimensional y no se selecciona hipótesis plana.
En ambas dimensiones, `--modo homogeneo` permite comprobar el caso conocido
E=1 000 000, nu=0,3. `--solver auto` elige LU para problemas pequeños y AMG
para los grandes; también puedes seleccionar `lu` o `amg` explícitamente.

Cada región de la malla FEM debe pertenecer a un único grupo físico de material.
Se reconocen los nombres `matriz_tipo_0`, `agregado_tipo_1`, etc.: el número
final corresponde al código de `Propiedades.txt`. La etiqueta numérica de Gmsh
no se confunde con ese código. Si tu malla usa otros nombres, proporciona
`--mapa-materiales materiales.json`, con un objeto como `{"1": 0, "2": 1}`
que asocia cada etiqueta física con su material.

FEM guarda `resumen.txt`, `run.json`, `C_equivalente.txt` y
`C_equivalente_fuerzas.txt`. Al omitir `--sin-campos`, escribe archivos XDMF/HDF5
con desplazamientos y componentes escalares de deformación y esfuerzo que
puedes abrir en ParaView (`sxx`, `txy`, `exx`, `gamma_xy`, etc.).
El directorio de salida debe estar vacío.

Para comparar FEM y VEM, usa el mismo caso, propiedades, unidades, hipótesis
2D y condiciones de borde. En un material uniforme, ambos deben reproducir
la matriz teórica. En uno heterogéneo puede haber diferencias porque las mallas
y aproximaciones son distintas; comprueba también las fracciones de material
y estudia el refinamiento antes de interpretar esas diferencias como errores.
