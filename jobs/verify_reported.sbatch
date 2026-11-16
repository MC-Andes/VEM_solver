#!/bin/bash -l
#SBATCH --job-name=vem3d-reported-env
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=logs/reported-%j.log
set -euo pipefail
module load python/3.11
source .venv-reported/bin/activate
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "results/reported-${SLURM_JOB_ID}"
python -m pip freeze > "results/reported-${SLURM_JOB_ID}/environment.txt"
python -m pytest --junitxml="results/reported-${SLURM_JOB_ID}/junit.xml"
python examples/index_probe.py
