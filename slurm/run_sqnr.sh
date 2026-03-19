#!/bin/bash
#SBATCH --job-name=qf8-sqnr
#SBATCH --account=schwartz_lab
#SBATCH --partition=shared
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/sqnr_%j.out
#SBATCH --error=slurm_logs/sqnr_%j.err
#SBATCH --mail-user=neverett@g.harvard.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# =============================================================================
# QF8 SQNR-Only Experiment (no GPU needed)
# =============================================================================
# Just loads weights and measures quantization error. Fast.
# For 70B: sbatch --mem=200G slurm/run_sqnr.sh

export QF8_SQNR_ONLY=1
exec bash slurm/run_ptq.sh
