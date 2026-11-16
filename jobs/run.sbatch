#!/bin/bash -l
#SBATCH --job-name=vem3d-mesh
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/mesh-%j.log
set -euo pipefail
module load python/3.11
source .venv/bin/activate
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
: "${VEM_MESH:?Definir VEM_MESH con la carpeta de entrada}"
MODE="${VEM_MODE:-homogeneo}"
OUT="results/${SLURM_JOB_ID}_${MODE}"
VEM_SCRATCH_BASE="${VEM_TMP_BASE:-${SLURM_TMPDIR:-$PWD/work}}"
mkdir -p "$OUT" "$VEM_SCRATCH_BASE"
SCRATCH=$(mktemp -d "$VEM_SCRATCH_BASE/vem-${SLURM_JOB_ID}-XXXXXX")
trap 'rmdir "$SCRATCH" 2>/dev/null || true' EXIT
printf 'Temporales: %s\n' "$SCRATCH"
hostname
time vem3d-run "$VEM_MESH" --modo "$MODE" --solver amg \
    --sin-campos --salida "$OUT" --tmp "$SCRATCH"
