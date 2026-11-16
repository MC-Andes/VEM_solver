#!/bin/bash -l
#SBATCH --job-name=vem3d-verify
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=logs/verify-%j.log
set -euo pipefail
module load python/3.11
source .venv/bin/activate
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
mkdir -p "results/verify-${SLURM_JOB_ID}"
hostname
python --version
python -m pip freeze > "results/verify-${SLURM_JOB_ID}/environment.txt"
python -m pytest --cov=vem3d --cov-report=term-missing --junitxml="results/verify-${SLURM_JOB_ID}/junit.xml"
