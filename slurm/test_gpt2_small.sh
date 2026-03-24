#!/bin/bash
#SBATCH --job-name=qf8-test
#SBATCH --account=schwartz_lab
#SBATCH --partition=gpu_test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/gpt2_test_%j.out
#SBATCH --error=slurm_logs/gpt2_test_%j.err
#SBATCH --mail-user=neverett@g.harvard.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# =============================================================================
# QF8 Training — SMOKE TEST
# =============================================================================
# Short run to validate the full pipeline: download, tokenize, train, save.
# 500 steps x 3 runs, should finish in ~30 min on H100.
#
# Before submitting:
#   mkdir -p slurm_logs
#
# Usage:
#   sbatch slurm/test_gpt2_small.sh
# =============================================================================

echo "==========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPU: ${CUDA_VISIBLE_DEVICES:-none}"
echo "Start: $(date)"
echo "==========================================="

# ── Environment ──
module load python
set +eu  # mamba activate can fail with strict mode
mamba activate quake-float-8
set -eu

export HF_HOME="/n/holylabs/schwartz_lab/Lab/neverett/quake-float-8/huggingface_cache"
mkdir -p "$HF_HOME"

echo "Python: $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA: $(python -c 'import torch; print(torch.cuda.get_device_name() if torch.cuda.is_available() else "N/A")')"
echo "HF cache: $HF_HOME"

# ── Run ──
cd "$SLURM_SUBMIT_DIR"

export PYTHONUNBUFFERED=1
python src/train_gpt2_small.py --dataset openwebtext --steps 500
EXIT_CODE=$?

echo "==========================================="
echo "Completed with exit code: $EXIT_CODE"
echo "$(date)"
echo "==========================================="

exit $EXIT_CODE
