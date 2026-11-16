#!/bin/bash -l
#SBATCH --job-name=vem3d-readme-example
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00
#SBATCH --output=logs/example-%j.log
set -euo pipefail
module load python/3.11
source .venv/bin/activate
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
OUT="results/example-${SLURM_JOB_ID}"
python examples/cube.py "$OUT/ejemplo/VEM"
vem3d-run "$OUT/ejemplo/VEM" --modo homogeneo --solver lu --sin-campos --salida "$OUT/resultados/ejemplo"
