# Revisión técnica y evidencia

## Error reportado: punteros CSR

En la línea 942 del núcleo recibido se evaluaba:

```python
indptr[...] = b.indptr[1:] + total_nnz
```

SciPy puede representar una franja con `b.indptr` int32. El acumulado global de
no ceros puede superar 2,147,483,647 aunque la malla sólo tenga 8,294,124 GDL.
El dtype del destino no cambia el dtype con que NumPy evalúa la suma.
Con NumPy 2 un escalar fuera de int32 produce OverflowError; antes de cruzar
ese límite también puede haber wraparound del resultado de la suma.

`assembly.csr_offset` convierte el operando a int64 **antes** de sumar y comprueba
el límite int64. Los índices de columna escritos a disco se amplían cuando la
cota de tripletas supera int32, evitando que el CSR global exija una conversión
completa int32 → int64 al construirse. La cota es conservadora: SciPy aún puede
reducirlos para matrices pequeñas. El almacenamiento final puede ser 16 bytes
por no cero más los punteros; ya no debe presupuestarse siempre 12 bytes.

La regresión usa desplazamientos a ambos lados de `2**31` y cruza el límite sin
reservar miles de millones de entradas. Esto verifica la operación que falla;
no equivale a haber resuelto la malla original completa.

## Segunda barrera: PyAMG 5.3.0

El trabajo **803331** repitió las 31 pruebas con NumPy 2.1.3, SciPy 1.15.3 y
PyAMG 5.3.0: pasaron para matrices pequeñas. Sin embargo, el probe de índices
int64 falló con `TypeError` en `symmetric_strength_of_connection`, cuyo kernel
en esa versión acepta únicamente int32. El mismo probe con PyAMG 5.4.1
(trabajo **803328**) pasó manteniendo int64 en todos los niveles y alcanzó
residuo relativo 2.15e-13. Por ello se fija **pyamg>=5.4.1** como dependencia
y se añade un diagnóstico explícito si se ejecuta manualmente con una versión
antigua y un CSR de 64 bits. No se hace downcast de una matriz grande.

El entorno antiguo sirve sólo para diagnóstico, no como instalación soportada.
Después, el trabajo **803337** conservó NumPy 2.1.3 y SciPy 1.15.3, actualizando
únicamente PyAMG a 5.4.1: **32 pruebas aprobadas**, 84.16 % de cobertura y probe
int64 aceptado con residuo 2.08e-13. No fue necesario actualizar NumPy/SciPy
para esas pruebas.
No se ha ejecutado Windows/Python 3.13: estas comparaciones fueron en Linux,
Python 3.11.10, conservando las tres versiones numéricas del reporte.

## Otros defectos corregidos

1. `en_disco=True` intentaba mapear archivos vacíos si no había GDL libres.
2. Archivos abiertos podían impedir limpiar un ensamblaje fallido.
3. El flujo no liberaba temporales si fallaba la preparación o resolución.
4. AMG devolvía resultados al agotar iteraciones sin exigir tolerancia.
5. El batch homogéneo podía devolver éxito aunque fallara su verificación de C.
6. La orientación comparaba índices GDL-x con IDs de nodo; sólo era válida
   para ciertas numeraciones. Ahora usa IDs de nodo explícitos.
7. `--sin-campos` ahorraba disco, pero retenía los seis campos en RAM.

El solver automático deja de recurrir silenciosamente a LU para sistemas
mayores al umbral si falta AMG; comunica la dependencia faltante.

## Optimización algebraica

Sea `d = 3*n_e`, D de tamaño d×12 y P=piast de tamaño 12×d. La estabilización
original es α(I−DP)ᵀ(I−DP). Se evalúa como:

```text
α [ I − DP − (DP)ᵀ + Pᵀ(DᵀD)P ]
```

Se evita el producto d×d por d×d: O(d³) → O(12 d²). La matriz elemental sigue
siendo densa, por lo que su memoria sigue siendo O(d²). La complejidad global
del solver AMG y la conectividad de la malla no cambian por esta identidad.
G se obtiene como `V * dpi.T @ C @ dpi`, y las tracciones se evalúan para los
12 monomios simultáneamente. Son operaciones equivalentes en aritmética exacta;
las pruebas comparan error relativo y toleran redondeo de punto flotante.

En el flujo de producción se almacena `strain = dpi @ piast` (6×d), en lugar
de `dpi` (6×12) y `piast` (12×d). El ensamblaje expone `compact_info` para conservar
el formato histórico cuando se necesita inspeccionar los proyectores. El
postproceso acepta ambas representaciones. Sin campos, se guardan sólo
estadísticas por carga y se liberan los arreglos completos entre cargas.

## Verificación en Hypatia

Trabajos Slurm **803342** y **803343**, Python 3.11.10: **35 pruebas aprobadas**
en cada entorno, cobertura total **84.23 %**. Incluye matrices vacías, partición en múltiples franjas,
limpieza ante fallos inyectados, seis cargas homogéneas con LU y AMG,
numeración por bloques e intercalada, materiales heterogéneos, CSV, CLI y
comparación contra el código recibido. Los prismas sesgados de 6, 16 y 48
vértices reproducen la rigidez original con error relativo inferior a 1e-12,
conservando simetría y los seis modos rígidos.

La revisión final **803351** volvió a aprobar las 35 pruebas con 84.23 % de
cobertura después de ajustar textos de salida y la portabilidad de una prueba.

Benchmark **803326**, mismo nodo, BLAS a un hilo, mediana de 20 muestras por
método tras calentamiento y orden alternado:

| Vértices por elemento | Original (ms) | Optimizado (ms) | Aceleración | Arreglos de info antes/después (bytes) |
| --- | ---: | ---: | ---: | ---: |
| 8 | 2.728 | 0.931 | 2.93× | 3384 / 1656 |
| 32 | 15.788 | 2.898 | 5.45× | 10872 / 5688 |
| 64 | 13.800 | 5.919 | 2.33× | 20856 / 11064 |

La reducción de arreglos de info es 47–51 %, excluyendo la sobrecarga de objetos
Python. Es un microbenchmark del kernel; no es una medida de aceleración de
toda la malla, y el nodo compartido introduce variabilidad temporal.

## Límites que permanecen

- Los archivos de conectividad y propiedades suponen índices base cero y
  correspondencia entre ID y posición; falta un validador integral de mallas.
- La detección de borde supone un dominio prismático completo.
- Las tolerancias geométricas heredadas no están normalizadas en todas las
  operaciones; conviene verificar cambios extremos de unidades o escala.
- Los bloques de borde se acumulan en RAM y la geometría de todas las caras
  permanece cargada durante el ensamblaje. Memmap no elimina el coste de RAM
  de AMG ni el espacio temporal de tripletas.
- Las tripletas soportan menos de `2**31` GDL; los punteros globales de no ceros
  sí soportan int64. Se exige PyAMG 5.4.1 o posterior para
  matrices con índices de 64 bits; el probe no reserva más de 2³¹ no ceros.
- Para el paper faltan un estudio de convergencia, tolerancias físicas de
  aceptación de casos heterogéneos y metadatos editoriales confirmados.

Los originales están preservados en `reference/original`, con SHA256. Los
fallos iniciales de preparación del entorno se conservaron en los logs:
803322 falló antes del solver por ausencia de `/usr/bin/time` en el nodo;
803323/803324 por un error sintáctico del generador, corregido antes de 803325.

Lint, formato, construcción de sdist/wheel, importación del wheel y documentación
estricta completados en la revisión final **803352**. El probe **803336** reproduce
el OverflowError original y el wraparound previo al límite, y confirma la
corrección con punteros `[2147483648, 2147483649, 2147483653]`.

## Inicio afín y corrida grande

AMG utiliza ahora el campo afín macroscópico como aproximación inicial, también
en modo heterogéneo. El residuo se calcula contra la matriz ensamblada y debe
satisfacer la misma tolerancia 1e-12. En un medio homogéneo, el campo afín puede
satisfacer ya el sistema: en ese caso cero iteraciones es un resultado válido,
no un criterio de convergencia omitido. El ensamblaje, los esfuerzos y las
reacciones siguen calculándose. Para comparar con el método anterior existe
`--inicio-cero`. La comparación de ambas inicializaciones en medios homogéneos
y heterogéneos forma parte de las 35 pruebas; se comprueba además RHS cero con
una aproximación inicial no nula.

La primera corrida grande, **803327**, completó lectura, geometría y ensamblaje,
pero se interrumpió deliberadamente después de 35 min 51 s para sustituir el
inicio nulo. Una inspección no bloqueante del proceso mostró la primera carga
en la iteración 103, con residuo relativo cercano a 5e-6. No se presenta como
una corrida completada ni se usa para calcular una aceleración global.
Se conservaron sus logs y el perfil; se eliminaron sólo sus temporales propios.

El trabajo de reemplazo **803344** terminó con estado COMPLETED y código 0,
usando inicio afín y disco temporal local del nodo, manteniendo solver y
tolerancia. Se usó la malla **CA4_A1_G1_R01**, disponible en el clúster, con
498 131 nodos, 155 534 elementos y 1 494 393 GDL (1 428 933 libres). La malla
CA50_A2_G101_R03 del reporte original no estaba disponible; esta prueba no
reproduce su tamaño ni valida por sí sola un CSR con más de 2³¹ no ceros.

| Medida | Resultado |
| --- | ---: |
| Lectura | 2.64 s |
| Geometría | 203.99 s |
| Ensamblaje | 273.52 s |
| Solver y homogenización de seis cargas | 408.17 s |
| Tiempo del flujo / trabajo Slurm | 889.75 s / 14 min 52 s |
| Iteraciones máximas por carga | 8 |
| Residuo relativo máximo del sistema | 9.403e-13 |
| Error relativo de C frente al material uniforme | 6.572e-15 |
| Discrepancia entre esfuerzos medios y reacciones | 2.300e-11 |
| Discrepancia relativa de volumen | 1.311e-14 |
| Error relativo máximo de desplazamiento frente al campo afín | 3.837e-8 |
| E equivalente / nu equivalente | 999 999.9999999997 Pa / 0.2999999999999999 |

El proceso registró un pico de memoria residente de **35.30 GiB** mediante
`resource.getrusage`. Slurm registró MaxRSS de 28 480 220 KiB (**27.16 GiB**);
son mediciones distintas y su muestreo puede no capturar el mismo pico. Para
presupuestar memoria se conserva la mayor medida. El campo histórico
`memoria_pico_gb` usa divisores binarios, por lo que sus valores son GiB.

El error del campo afín no es una promesa de precisión de los esfuerzos
locales. Por ejemplo, para ezz=1 el máximo esfuerzo de corte tyz fue
21 995.5 Pa aunque su valor teórico es cero. La matriz C promediada pasó las
verificaciones, pero para publicar campos locales hace falta estudiar
condicionamiento, calidad de los elementos y convergencia. No se validó una
homogenización heterogénea de este tamaño; los casos heterogéneos de regresión
son pequeños.

El ensamblaje anterior tomó 542.72 s con temporales en almacenamiento compartido.
La comparación no aísla una sola mejora: también cambió la ubicación de los
temporales. Tampoco se calcula una aceleración global frente a la corrida
interrumpida.

En el paquete entregado, `validation/results/803344_homogeneo/run.json` conserva
los resultados, las huellas de entrada y del código de esa corrida. Los logs
y `validation/slurm-accounting.txt` conservan el estado de Slurm. Después de
la corrida se corrigieron únicamente una afirmación excesiva de precisión en
el texto del resumen y un docstring del batch; no cambió el cálculo numérico.
Por eso esas dos huellas de fuentes difieren de la entrega final. El resumen
bruto se conserva sin editar.

## Ejemplo de entrada para usuarios nuevos

El trabajo **803339** ejecutó los comandos del ejemplo del README: un cubo con
27 nodos y ocho elementos, material uniforme, LU y salida sin campos. Obtuvo
E=1 000 000, nu=0.3, `err_C=3.459e-16`, discrepancia entre vías `1.211e-15` y
error del campo lineal `2.220e-16`. Las instrucciones del README son generales
para equipos personales; Hypatia es sólo el lugar donde se hicieron estas pruebas.
