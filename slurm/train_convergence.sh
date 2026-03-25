#!/bin/bash
#SBATCH --job-name=qf8-conv
#SBATCH --account=schwartz_lab
#SBATCH --partition=gpu_h200
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=3-00:00:00
#SBATCH --output=slurm_logs/convergence_%j.out
#SBATCH --error=slurm_logs/convergence_%j.err
#SBATCH --mail-user=neverett@g.harvard.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# =============================================================================
# QF8 Convergence Training — GPT-2 Small (3 formats × 3 seeds)
# =============================================================================
# Runs 9 jobs (3 formats × 3 seeds) across GPUs using GNU parallel.
#
# For testing, override SBATCH above and tweak these variables:
#   partition=gpu_test, gres=gpu:4, time=00:30:00, mem=64G, cpus=4
#   STEPS=50, BATCH=4, GRAD_ACCUM=1, SEQ_LEN=256, MAX_DOCS=5000
#
# Usage:
#   mkdir -p slurm_logs
#   sbatch slurm/train_convergence.sh
# =============================================================================

# ── Config (edit for testing) ──
STEPS=200000
BATCH=32
GRAD_ACCUM=8
SEQ_LEN=1024
MAX_DOCS=0
N_GPUS=4
RESULTS_DIR="results/training/convergence"

# ── Setup ──
set -euo pipefail

echo "==========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPUs: $(nvidia-smi -L 2>/dev/null | wc -l)"
echo "Start: $(date)"
echo "Config: steps=${STEPS} batch=${BATCH} grad_accum=${GRAD_ACCUM} seq=${SEQ_LEN} max_docs=${MAX_DOCS}"
echo "==========================================="

module load python
set +eu
mamba activate quake-float-8
set -eu

export HF_HOME="/n/holylabs/schwartz_lab/Lab/neverett/quake-float-8/huggingface_cache"
mkdir -p "$HF_HOME"

echo "Python: $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA: $(python -c 'import torch; print(torch.cuda.get_device_name() if torch.cuda.is_available() else "N/A")')"

cd "$SLURM_SUBMIT_DIR"
export PYTHONUNBUFFERED=1
mkdir -p "$RESULTS_DIR"

# ── Run function ──
run_one() {
    local fmt=$1
    local seed=$2
    local gpu=$3
    local outfile="${RESULTS_DIR}/${fmt}_seed${seed}.json"
    local max_docs_flag=""
    [ "$MAX_DOCS" -gt 0 ] && max_docs_flag="--max-docs $MAX_DOCS"

    echo "[$(date +%H:%M:%S)] Starting ${fmt} seed=${seed} on GPU ${gpu}"
    CUDA_VISIBLE_DEVICES=$gpu python src/train_gpt2_small.py \
        --dataset openwebtext \
        --format "$fmt" \
        --steps $STEPS \
        --batch $BATCH \
        --grad-accum $GRAD_ACCUM \
        --seq-len $SEQ_LEN \
        --seed "$seed" \
        --output "$outfile" \
        $max_docs_flag \
        > "${RESULTS_DIR}/${fmt}_seed${seed}.log" 2>&1
    local rc=$?

    echo "[$(date +%H:%M:%S)] Finished ${fmt} seed=${seed} (exit ${rc})"
    return $rc
}
export -f run_one
export STEPS BATCH GRAD_ACCUM SEQ_LEN MAX_DOCS RESULTS_DIR HF_HOME

# ── Schedule 9 jobs across N GPUs ──
parallel --jobs "$N_GPUS" --colsep ' ' \
    'run_one {1} {2} $(( {%} - 1 ))' \
    ::: fp32 qf8 fp8 \
    ::: 42 123 7

echo ""
echo "==========================================="
echo "All runs completed: $(date)"
echo "Results in: ${RESULTS_DIR}/"
ls -la "${RESULTS_DIR}/"
echo "==========================================="
