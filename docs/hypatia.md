# Ejecución en Hypatia

Esta guía es opcional y está dirigida a quienes tengan acceso a Hypatia.
Las validaciones documentadas de esta migración se realizaron allí. Para ejecutar
cálculos en el clúster, usar Slurm para obtener un nodo de cálculo; la instalación
general en un equipo personal se explica en el README.

## Instalación aislada

```bash
rsync -az --exclude .venv --exclude results --exclude logs \
  /Users/nuseche/Desktop/VEM_solver/ hypatia:proyectos/vem3d_20261116/
ssh hypatia
cd ~/proyectos/vem3d_20261116
module load python/3.11
python -m venv .venv
source .venv/bin/activate
python -m pip install -c environments/hypatia.txt -e '.[dev,docs]'
mkdir -p logs
sbatch jobs/verify.sbatch
```

Para repetir el entorno medido, usar el archivo `environment.txt` del trabajo
de verificación. Las cotas de dependencias en pyproject permiten instalar; el
freeze registra las versiones exactas realmente usadas.

## Una malla

```bash
export VEM_MESH=/ruta/absoluta/a/los/archivos/VEM
export VEM_MODE=homogeneo
sbatch --export=ALL jobs/run.sbatch
```

El script reserva por defecto 1 CPU, 64 GiB y 4 horas, con BLAS a un hilo.
Ajustar memoria y tiempo al tamaño; por ejemplo `--partition=bigmem --mem=192G`
para una malla grande después de medir. Estos recursos no garantizan que una
malla de 8.3 millones de GDL termine; usar PyAMG 5.4.1 o posterior para índices de 64 bits.

Cambiar `VEM_MODE=heterogeneo` usa las propiedades de entrada. No modificar
los archivos de malla; resultados y temporales van al directorio del proyecto.
Los nombres incluyen el job ID. Para usar disco local del nodo, comprobar primero
el espacio disponible y exportar `VEM_TMP_BASE=/tmp` antes de `sbatch`; el script
crea un directorio temporal único. Sin esa opción usa el temporal de Slurm o
el directorio `work` del proyecto. Consultar `squeue -j ID` y
`sacct -j ID --format=JobID,State,ExitCode,Elapsed,MaxRSS`.

## Lotes

Dentro de una asignación Slurm:

```bash
vem3d-batch-homogeneous --raiz /ruta/3D --listar
vem3d-batch-homogeneous --raiz /ruta/3D --procesos 1 --sin-campos --tmp /ruta/scratch
vem3d-batch-heterogeneous --raiz /ruta/3D --procesos 1 --sin-campos --tmp /ruta/scratch
```

Los batch históricos escriben `Resultados_Homogeneos` o
`Resultados_Heterogeneos` bajo `--raiz`. Usar una raíz de trabajo si el conjunto
es de sólo lectura. El valor por defecto se redujo de 4 a 1 proceso.

## Checks

El trabajo `jobs/verify.sbatch` recoge las pruebas y versiones. El trabajo
`jobs/quality.sbatch` ejecuta lint, formato, build de paquete y documentación.
No se configuró ejecución automática del solver en GitHub Actions: durante
esta migración se solicitó realizar todas las corridas en Hypatia.
Un futuro runner privado requiere configuración explícita del equipo; no debe
concederse ejecución de PR públicos directamente sobre el clúster.
