#!/bin/bash
#SBATCH --job-name=qf8-test
#SBATCH --account=schwartz_lab
#SBATCH --partition=gpu_test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:10:00
#SBATCH --output=slurm_logs/test_conv_%j.out
#SBATCH --error=slurm_logs/test_conv_%j.err

# =============================================================================
# Smoke test for convergence training script
# =============================================================================
# Runs each format for 50 steps on wikitext2 to verify the --format and
# --output flags work correctly on the cluster.
#
# gpu_test: 8 A100 MIG 3g.20GB, 12h limit, max 2 jobs
#
# Usage:
#   mkdir -p slurm_logs
#   sbatch slurm/test_convergence.sh
# =============================================================================

set -euo pipefail

echo "==========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi -L 2>/dev/null | head -1)"
echo "Start: $(date)"
echo "==========================================="

# ── Environment ──
module load python
set +eu
mamba activate quake-float-8
set -eu

export HF_HOME="/n/holylabs/schwartz_lab/Lab/neverett/quake-float-8/huggingface_cache"
mkdir -p "$HF_HOME"

echo "Python: $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"

cd "$SLURM_SUBMIT_DIR"
export PYTHONUNBUFFERED=1

TEST_DIR="results/training/test_convergence"
mkdir -p "$TEST_DIR"

PASSED=0
FAILED=0

for fmt in fp32 qf8 fp8; do
    echo ""
    echo "--- Testing --format ${fmt} ---"
    outfile="${TEST_DIR}/${fmt}_test.json"

    if python src/train_gpt2_small.py \
        --dataset openwebtext \
        --max-docs 5000 \
        --format "$fmt" \
        --steps 50 \
        --batch 4 \
        --grad-accum 1 \
        --seq-len 256 \
        --seed 42 \
        --output "$outfile"; then

        if [ -f "$outfile" ]; then
            echo "  PASS: ${fmt} — output written to ${outfile}"
            PASSED=$((PASSED + 1))
        else
            echo "  FAIL: ${fmt} — ran but no output file"
            FAILED=$((FAILED + 1))
        fi
    else
        echo "  FAIL: ${fmt} — exited with error"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "==========================================="
echo "Results: ${PASSED} passed, ${FAILED} failed"
echo "$(date)"
echo "==========================================="

[ $FAILED -eq 0 ] && exit 0 || exit 1
