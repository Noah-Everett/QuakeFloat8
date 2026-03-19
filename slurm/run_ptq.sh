#!/bin/bash
#SBATCH --job-name=qf8-ptq
#SBATCH --account=schwartz_lab
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=slurm_logs/ptq_%j.out
#SBATCH --error=slurm_logs/ptq_%j.err
#SBATCH --mail-user=neverett@g.harvard.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# =============================================================================
# QF8 PTQ Perplexity + SQNR Experiment
# =============================================================================
# Runs weight-only PTQ on 7B+ models, measures SQNR and WikiText-2 perplexity.
# Default: Phi-2 (2.7B), LLaMA-2-7B, Mistral-7B, LLaMA-2-13B.
#
# Usage:
#   sbatch slurm/run_ptq.sh                                      # all defaults
#   QF8_MODELS="meta-llama/Llama-2-7b-hf" sbatch slurm/run_ptq.sh  # one model
#   QF8_SQNR_ONLY=1 sbatch --partition=shared --gres="" slurm/run_ptq.sh  # SQNR only, no GPU
#
# For 13B you may need more VRAM:
#   sbatch --mem=128G slurm/run_ptq.sh
# =============================================================================

set -euo pipefail

echo "==========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPU: ${CUDA_VISIBLE_DEVICES:-none}"
echo "Start: $(date)"
echo "==========================================="

# ── Environment ──
module load python
mamba activate quake-float-8

# HF cache on lab storage (model weights are ~60GB)
export HF_HOME="/n/holylabs/schwartz_lab/Lab/neverett/quake-float-8/huggingface_cache"
mkdir -p "$HF_HOME"

# All models are fully open — no HF auth needed

echo "Python: $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA: $(python -c 'import torch; print(torch.cuda.get_device_name() if torch.cuda.is_available() else "N/A")')"
echo "HF cache: $HF_HOME"

# ── Run ──
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm_logs results/scaling

CMD="python src/ptq_scale.py --output-dir results/scaling"

if [ -n "${QF8_MODELS:-}" ]; then
    CMD="$CMD --models $QF8_MODELS"
fi

if [ -n "${QF8_SQNR_ONLY:-}" ]; then
    CMD="$CMD --sqnr-only"
fi

if [ -n "${QF8_PPL_ONLY:-}" ]; then
    CMD="$CMD --ppl-only"
fi

echo "Running: $CMD"
eval $CMD

EXIT_CODE=$?

echo "==========================================="
echo "Completed with exit code: $EXIT_CODE"
echo "$(date)"
echo "==========================================="

exit $EXIT_CODE
