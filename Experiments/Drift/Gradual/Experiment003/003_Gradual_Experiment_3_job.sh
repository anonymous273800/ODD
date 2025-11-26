#!/bin/bash
#SBATCH -J 002_Incremental_Experiment_3_shaira-odd-gpu
#SBATCH -o 002_Incremental_Experiment_3_shaira-odd-gpu.%j.out
#SBATCH -e 002_Incremental_Experiment_3_shaira-odd-gpu.%j.err
#SBATCH -p gpu-a100
#SBATCH --mem=64G
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 48:00:00
#SBATCH -A CCR25027
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ShairaAbu-Shaira@my.unt.edu
# If your site requires explicit GPUs, uncomment ONE of these per policy:
# #SBATCH --gpus-per-node=1
# #SBATCH --gres=gpu:a100:1

set -euo pipefail

module load python3/3.9.7
module load cuda/11.4

PROJECT_ROOT="/work/10879/mabushaira/ls6/001ODD/ODD"
ENV_DIR="$PROJECT_ROOT/shaira_odd_env"
SCRIPT_DIR="$PROJECT_ROOT/Experiments/Drift/Incremental/Experiment002"

source "$ENV_DIR/bin/activate"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"


# --- quick preflight (optional) ---
echo "Python: $(which python)"
python -V
echo "PYTHONPATH=$PYTHONPATH"
python -c "import sys, os; print('CUDA_VISIBLE_DEVICES=', os.getenv('CUDA_VISIBLE_DEVICES')); import Datasets; print('Datasets OK:', Datasets.__file__)"



# --- run experiment ---
cd "$SCRIPT_DIR"
echo "Running main experiment: python 002_Incremental_Experiment_3.py"
srun python 002_Incremental_Experiment_3.py

echo "Job completed."
