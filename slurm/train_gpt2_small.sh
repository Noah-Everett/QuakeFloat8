#!/bin/bash
#SBATCH --job-name=qf8-gpt2
#SBATCH --output=slurm/logs/%j.out
#SBATCH --error=slurm/logs/%j.err
#SBATCH --partition=gpu_test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00

set -euo pipefail

echo "=== Job $SLURM_JOB_ID on $(hostname) ==="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
date

# Load modules — adjust these to match your FASRC setup
module load python/3.11.5-fasrc01 2>/dev/null || true
module load cuda/12.4.1-fasrc01 2>/dev/null || true

# Create venv if needed
if [ ! -d .venv-slurm ]; then
    python3 -m venv .venv-slurm
    source .venv-slurm/bin/activate
    pip install --upgrade pip
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
    pip install tiktoken datasets numpy
else
    source .venv-slurm/bin/activate
fi

echo "Python: $(python3 --version)"
echo "PyTorch: $(python3 -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python3 -c 'import torch; print(torch.cuda.is_available())')"
echo ""

# Run training
PYTHONUNBUFFERED=1 python3 src/train_gpt2_small.py

echo ""
echo "=== Done ==="
date
