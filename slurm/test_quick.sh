#!/bin/bash
#SBATCH --job-name=qf8-test
#SBATCH --account=schwartz_lab
#SBATCH --partition=gpu_test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=slurm_logs/test_%j.out
#SBATCH --error=slurm_logs/test_%j.err

# Quick smoke test on gpu_test partition.
# SQNR on Pythia-160M + perplexity on Pythia-160M.
# Should finish in <10 min.

set -euo pipefail

echo "==========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPU: ${CUDA_VISIBLE_DEVICES:-none}"
echo "Start: $(date)"
echo "==========================================="

module load python
mamba activate quake-float-8

export HF_HOME="/n/holylabs/schwartz_lab/Lab/neverett/quake-float-8/huggingface_cache"
mkdir -p "$HF_HOME"

echo "Python: $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA: $(python -c 'import torch; print(torch.cuda.get_device_name() if torch.cuda.is_available() else "N/A")')"
echo "VRAM: $(python -c 'import torch; print(f"{torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB") if torch.cuda.is_available() else print("N/A")')"

cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm_logs results/scaling

echo "=== SQNR test ==="
python src/ptq_scale.py --sqnr-only --models EleutherAI/pythia-160m --output-dir results/scaling

echo "=== Perplexity test ==="
python src/ptq_scale.py --ppl-only --models EleutherAI/pythia-160m --output-dir results/scaling

EXIT_CODE=$?
echo "==========================================="
echo "Completed with exit code: $EXIT_CODE"
echo "$(date)"
echo "==========================================="
exit $EXIT_CODE
