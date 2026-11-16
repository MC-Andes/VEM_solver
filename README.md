# VEM 3D

Este programa permite estimar las propiedades elásticas de un material formado
por varios componentes. Por ejemplo, puede representar un pequeño volumen de
concreto con una matriz y agregados, aplicar distintas deformaciones y calcular
cómo responde el conjunto. El resultado principal es una matriz de rigidez
**C**, que relaciona las deformaciones con los esfuerzos medios del material.
También se calculan un módulo de Young y un coeficiente de Poisson equivalentes.

Para realizar el cálculo se usa el **Método de Elementos Virtuales**, o VEM.
La geometría se divide en pequeños elementos poliédricos, que en conjunto forman
una *malla*. El programa lee esa malla y las propiedades de sus materiales;
no genera automáticamente la geometría de una muestra real. Sí incluye un
pequeño ejemplo que permite comprobar la instalación sin disponer de una malla
propia.

El modelo considera elasticidad lineal y pequeñas deformaciones. En el borde
exterior prescribe desplazamientos que corresponden a seis deformaciones
uniformes: tres de extensión y tres de corte. Esta condición se conoce como
**KUBC**. La implementación actual supone que el dominio completo tiene forma
de caja rectangular.

## Empezar con un ejemplo

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

## Usar una malla propia

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

## Ejecutar varias mallas

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
