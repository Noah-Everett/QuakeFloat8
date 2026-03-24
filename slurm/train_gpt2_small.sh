#!/bin/bash
#SBATCH --job-name=qf8-gpt2
#SBATCH --account=schwartz_lab
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/gpt2_%j.out
#SBATCH --error=slurm_logs/gpt2_%j.err
#SBATCH --mail-user=neverett@g.harvard.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# =============================================================================
# QF8 Training Experiment — GPT-2 Small
# =============================================================================
# Trains GPT-2 Small (162M) from scratch with FP32 / FP8-E4M3 / QF8
# quantization-aware training (STE), 20K steps on OpenWebText.
#
# Before submitting:
#   mkdir -p slurm_logs
#
# Usage:
#   sbatch slurm/train_gpt2_small.sh
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
python src/train_gpt2_small.py --dataset openwebtext --steps 20000
EXIT_CODE=$?

echo "==========================================="
echo "Completed with exit code: $EXIT_CODE"
echo "$(date)"
echo "==========================================="

exit $EXIT_CODE
