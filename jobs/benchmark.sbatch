#!/bin/bash -l
#SBATCH --job-name=vem3d-benchmark
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=logs/benchmark-%j.log
set -euo pipefail
module load python/3.11
source .venv/bin/activate
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "results/benchmark-${SLURM_JOB_ID}"
python examples/benchmark.py "results/benchmark-${SLURM_JOB_ID}/benchmark.json"
