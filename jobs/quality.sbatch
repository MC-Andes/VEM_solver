#!/bin/bash -l
#SBATCH --job-name=vem3d-quality
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=logs/quality-%j.log
set -euo pipefail
module load python/3.11
source .venv/bin/activate
ruff check .
ruff format --check .
python -m build
mkdocs build --strict
WHEEL_DIR="$PWD/work/wheel-${SLURM_JOB_ID}"
mkdir -p "$WHEEL_DIR"
python -m pip install --no-deps --target "$WHEEL_DIR" dist/*.whl
cd "$WHEEL_DIR"
python -c 'import pathlib, vem3d, vem3d_core; assert pathlib.Path(vem3d.__file__).is_relative_to(pathlib.Path.cwd()); print("Wheel import OK:", vem3d.__file__)'
