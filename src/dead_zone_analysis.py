#!/usr/bin/env python3
"""
dead_zone_analysis.py — Measure how many weight values fall in each format's
"dead zone" (between zero and the smallest positive representable value)
when using E8M0 block scaling.

For each block of 32 elements:
  1. Compute the E8M0 shared scale (power of 2 matching the block max)
  2. Compute the smallest positive value representable at that scale
  3. Count how many nonzero values in the block fall below that threshold
     (these get rounded to zero — information loss)

Compares QF8, FP8 E4M3, FP8 E5M2, and INT8.
"""

import gc
import math
import sys

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import AutoModel, AutoModelForCausalLM

plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'font.serif': ['Computer Modern Roman'],
    'axes.labelsize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
})

# ─── Format parameters ───────────────────────────────────────────────────────

BLOCK_SIZE = 32

# QF8: smallest positive per-element magnitude = 2^((1-64)/16) = 2^(-3.9375)
# Dynamic range within a block: max/min = 2^(63/16) / 2^(-63/16) = 2^(7.875) ≈ 235:1
QF8_BIAS = 64
QF8_MAX_CODE = 127
QF8_MAX_MAG = 2.0 ** ((QF8_MAX_CODE - QF8_BIAS) / 16.0)   # ≈ 15.32
QF8_MIN_MAG = 2.0 ** ((1 - QF8_BIAS) / 16.0)               # ≈ 0.0652
QF8_DYNAMIC_RATIO = QF8_MAX_MAG / QF8_MIN_MAG               # ≈ 235

# FP8 E4M3FN: smallest positive = 2^(-9) (smallest subnormal)
# Max = 448. Dynamic range ≈ 229,376:1
FP8E4M3_MIN_MAG = 2.0**(-6) * (1.0 / 8.0)  # = 2^(-9)
FP8E4M3_MAX_MAG = 448.0
FP8E4M3_DYNAMIC_RATIO = FP8E4M3_MAX_MAG / FP8E4M3_MIN_MAG

# FP8 E5M2: smallest positive = 2^(-16) (smallest subnormal)
# Max = 57344. Dynamic range ≈ 3.75 billion:1
FP8E5M2_MIN_MAG = 2.0**(-14) * (1.0 / 4.0)  # = 2^(-16)
FP8E5M2_MAX_MAG = 2.0**15 * 1.75  # = 57344
FP8E5M2_DYNAMIC_RATIO = FP8E5M2_MAX_MAG / FP8E5M2_MIN_MAG

# INT8: smallest positive magnitude = 1 (code=1). Max = 127.
# Dynamic range = 127:1
INT8_MIN_MAG = 1.0
INT8_MAX_MAG = 127.0
INT8_DYNAMIC_RATIO = INT8_MAX_MAG / INT8_MIN_MAG

FORMATS = [
    (r'\textsc{Int8}',     INT8_MAX_MAG,     INT8_MIN_MAG,     '#7f8c8d'),
    (r'\textsc{Fp8} E5M2', FP8E5M2_MAX_MAG,  FP8E5M2_MIN_MAG,  '#e67e22'),
    (r'\textsc{Fp8} E4M3', FP8E4M3_MAX_MAG,  FP8E4M3_MIN_MAG,  '#c0392b'),
    (r'\textbf{QF8}',      QF8_MAX_MAG,       QF8_MIN_MAG,       '#2471a3'),
]

# ─── Models ───────────────────────────────────────────────────────────────────

MODELS = [
    ("openai-community/gpt2",          "GPT2-S"),
    ("openai-community/gpt2-medium",   "GPT2-M"),
    ("openai-community/gpt2-large",    "GPT2-L"),
    ("facebook/opt-125m",              "OPT-125M"),
    ("facebook/opt-350m",              "OPT-350M"),
    ("EleutherAI/pythia-160m",         "Pythia-160M"),
    ("EleutherAI/pythia-410m",         "Pythia-410M"),
    ("google-bert/bert-base-uncased",  "BERT-B"),
    ("google-bert/bert-large-uncased", "BERT-L"),
    ("FacebookAI/roberta-base",        "RoBERTa-B"),
    ("google-t5/t5-small",            "T5-S"),
    ("google-t5/t5-base",             "T5-B"),
]

# ─── Analysis ─────────────────────────────────────────────────────────────────

def compute_dead_zone_fractions(weights_flat: torch.Tensor, fmt_max: float,
                                 fmt_min: float) -> np.ndarray:
    """For each block of 32, compute fraction of nonzero values in the dead zone.

    The dead zone is: values whose magnitude is nonzero but smaller than
    (block_scale × fmt_min), where block_scale = 2^floor(log2(block_max / fmt_max)).

    Returns array of shape (n_blocks,) with values in [0, 1].
    """
    n = weights_flat.numel()
    pad = (-n) % BLOCK_SIZE
    if pad > 0:
        weights_flat = torch.nn.functional.pad(weights_flat, (0, pad))

    blocks = weights_flat.reshape(-1, BLOCK_SIZE)
    abs_blocks = blocks.abs()

    # E8M0 block scale: 2^floor(log2(block_max / fmt_max))
    block_maxes = abs_blocks.max(dim=1).values  # (n_blocks,)

    # Skip all-zero blocks
    nonzero_mask = block_maxes > 0
    results = np.zeros(blocks.shape[0], dtype=np.float64)

    if nonzero_mask.sum() == 0:
        return results

    bm = block_maxes[nonzero_mask]
    ab = abs_blocks[nonzero_mask]

    # scale = 2^floor(log2(bm / fmt_max))
    log2_scale = torch.floor(torch.log2(bm / fmt_max))
    scale = (2.0 ** log2_scale).unsqueeze(1)  # (n_blocks, 1)

    # Smallest representable positive at this scale
    threshold = scale * fmt_min  # (n_blocks, 1)

    # Count nonzero values below threshold per block
    is_nonzero = ab > 0
    is_in_dead_zone = is_nonzero & (ab < threshold)
    n_nonzero = is_nonzero.sum(dim=1).float()
    n_dead = is_in_dead_zone.sum(dim=1).float()

    # Fraction of nonzero values that land in dead zone
    frac = torch.where(n_nonzero > 0, n_dead / n_nonzero, torch.zeros_like(n_dead))
    results[nonzero_mask.numpy()] = frac.numpy()

    return results


def analyze_model(hf_name: str, label: str):
    """Load a model, analyze dead zone fractions for all formats."""
    print(f"  Loading {label} ({hf_name})...")
    try:
        model = AutoModelForCausalLM.from_pretrained(hf_name, torch_dtype=torch.float32)
    except (ValueError, OSError):
        model = AutoModel.from_pretrained(hf_name, torch_dtype=torch.float32)
    model.eval()

    # Collect all weight tensors (skip biases and layernorms)
    all_weights = []
    for name, param in model.named_parameters():
        if param.ndim >= 2:  # weight matrices only
            all_weights.append(param.data.reshape(-1))

    weights_flat = torch.cat(all_weights)
    n_params = weights_flat.numel()
    print(f"    {n_params:,} weight parameters in {len(all_weights)} tensors")

    results = {}
    for fmt_name, fmt_max, fmt_min, _ in FORMATS:
        fracs = compute_dead_zone_fractions(weights_flat, fmt_max, fmt_min)
        results[fmt_name] = fracs
        mean_frac = fracs.mean() * 100
        max_frac = fracs.max() * 100
        pct_blocks_any = (fracs > 0).mean() * 100
        print(f"    {fmt_name:20s}: mean={mean_frac:.4f}%  max={max_frac:.2f}%  "
              f"blocks_with_any={pct_blocks_any:.2f}%")

    del model, all_weights, weights_flat
    gc.collect()

    return results, n_params


def main():
    models = MODELS
    if len(sys.argv) >= 2 and sys.argv[1] == "--quick":
        quick = ["openai-community/gpt2", "google-bert/bert-base-uncased",
                 "google-t5/t5-small", "facebook/opt-125m"]
        models = [m for m in MODELS if m[0] in quick]
        print("Running quick test (4 models)")

    # Collect per-block dead zone fractions across all models
    all_fracs = {name: [] for name, _, _, _ in FORMATS}
    model_labels = []
    model_means = {name: [] for name, _, _, _ in FORMATS}

    print("Dead zone analysis across models:")
    print("=" * 70)

    for hf_name, label in models:
        results, n_params = analyze_model(hf_name, label)
        model_labels.append(label)
        for fmt_name, _, _, _ in FORMATS:
            fracs = results[fmt_name]
            all_fracs[fmt_name].append(fracs)
            model_means[fmt_name].append(fracs.mean() * 100)

    # ─── Summary stats ────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY: Mean dead zone fraction across all blocks and models")
    print("=" * 70)
    for fmt_name, _, _, _ in FORMATS:
        all_blocks = np.concatenate(all_fracs[fmt_name])
        mean = all_blocks.mean() * 100
        median = np.median(all_blocks) * 100
        p99 = np.percentile(all_blocks, 99) * 100
        max_val = all_blocks.max() * 100
        pct_any = (all_blocks > 0).mean() * 100
        print(f"  {fmt_name:20s}: mean={mean:.4f}%  median={median:.4f}%  "
              f"p99={p99:.2f}%  max={max_val:.1f}%  blocks_w_any={pct_any:.1f}%")

    # ─── Plot 1: Per-model mean dead zone fraction (bar chart) ────────────

    n_models = len(model_labels)
    n_formats = len(FORMATS)
    x = np.arange(n_models)
    width = 0.8 / n_formats

    fig, ax = plt.subplots(figsize=(6.5, 3.0))
    for i, (fmt_name, _, _, color) in enumerate(FORMATS):
        means = model_means[fmt_name]
        # Clamp tiny values to a visible minimum for log scale
        means_vis = [max(m, 1e-4) for m in means]
        ax.bar(x + i * width - 0.4 + width/2, means_vis, width,
               label=fmt_name, color=color, alpha=0.85)

    ax.set_yscale('log')
    ax.set_xticks(x)
    ax.set_xticklabels(model_labels, rotation=45, ha='right', fontsize=7)
    ax.set_ylabel(r'Mean dead zone (\%)', fontsize=9)
    ax.set_title(r'Fraction of nonzero weights rounded to zero per block', fontsize=10)
    ax.legend(fontsize=7, ncol=2)
    ax.set_ylim(bottom=5e-5, top=5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # Add a reference line at 1%
    ax.axhline(1.0, color='#999999', linewidth=0.5, linestyle=':')

    plt.tight_layout()
    plt.savefig('repo/paper/fig_dead_zone_bars.pdf', bbox_inches='tight', pad_inches=0.03)
    plt.savefig('repo/paper/figures/fig_dead_zone_bars.png', bbox_inches='tight', dpi=300)
    print("\nSaved: fig_dead_zone_bars.pdf")
    plt.close(fig)

    # ─── Plot 2: CDF of per-block dead zone fraction ─────────────────────

    fig, ax = plt.subplots(figsize=(6.5, 3.0))
    for fmt_name, _, _, color in FORMATS:
        all_blocks = np.concatenate(all_fracs[fmt_name])
        total = len(all_blocks)
        pct_any = (all_blocks > 0).sum() / total * 100

        sorted_vals = np.sort(all_blocks[all_blocks > 0]) * 100
        if len(sorted_vals) == 0:
            # Format has zero dead zone — show as annotation
            ax.annotate(fmt_name + r' ($0\%$)', xy=(0.5, 0.02),
                        fontsize=7, color=color, ha='left',
                        xycoords=('data', 'axes fraction'))
            continue
        cdf = np.arange(1, len(sorted_vals) + 1) / total
        ax.plot(sorted_vals, cdf * 100, color=color, linewidth=1.2,
                label=fmt_name + rf' ({pct_any:.1f}\% of blocks)')

    ax.set_xlabel(r'Dead zone fraction per block (\%)', fontsize=9)
    ax.set_ylabel(r'Cumulative \% of all blocks', fontsize=9)
    ax.set_title(r'Distribution of dead zone impact across all blocks', fontsize=10)
    ax.legend(fontsize=7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig('repo/paper/fig_dead_zone_cdf.pdf', bbox_inches='tight', pad_inches=0.03)
    plt.savefig('repo/paper/figures/fig_dead_zone_cdf.png', bbox_inches='tight', dpi=300)
    print("Saved: fig_dead_zone_cdf.pdf")
    plt.close(fig)

    # ─── Plot 3: Histogram of within-block dynamic range ─────────────────
    # Shows what dynamic range blocks actually need vs what each format provides.

    print("\nComputing within-block dynamic range distribution...")
    all_dynamic_ranges = []
    for hf_name, label in models:
        try:
            model = AutoModelForCausalLM.from_pretrained(hf_name, torch_dtype=torch.float32)
        except (ValueError, OSError):
            model = AutoModel.from_pretrained(hf_name, torch_dtype=torch.float32)
        model.eval()

        weights = []
        for name, param in model.named_parameters():
            if param.ndim >= 2:
                weights.append(param.data.reshape(-1))
        w = torch.cat(weights)
        n = w.numel()
        pad = (-n) % BLOCK_SIZE
        if pad > 0:
            w = torch.nn.functional.pad(w, (0, pad))
        blocks = w.reshape(-1, BLOCK_SIZE)
        abs_blocks = blocks.abs()
        block_max = abs_blocks.max(dim=1).values
        # Min of nonzero values per block
        abs_blocks_masked = abs_blocks.clone()
        abs_blocks_masked[abs_blocks_masked == 0] = float('inf')
        block_min_nonzero = abs_blocks_masked.min(dim=1).values
        # Dynamic range = max / min (only for blocks with nonzero min)
        valid = (block_max > 0) & (block_min_nonzero < float('inf'))
        dr = (block_max[valid] / block_min_nonzero[valid]).numpy()
        all_dynamic_ranges.append(dr)
        del model, weights, w, blocks
        gc.collect()

    all_dr = np.concatenate(all_dynamic_ranges)
    log2_dr = np.log2(all_dr)

    fig, ax = plt.subplots(figsize=(6.5, 3.0))
    ax.hist(log2_dr, bins=100, density=True, color='#c8d6e5', alpha=0.8, edgecolor='none')

    # Vertical lines for each format's max dynamic range, with annotations
    y_top = ax.get_ylim()[1]
    annot_positions = [0.95, 0.85, 0.75, 0.65]  # stagger labels vertically
    for i, (fmt_name, fmt_max, fmt_min, color) in enumerate(FORMATS):
        dr = fmt_max / fmt_min
        log2_dr_val = np.log2(dr)
        dr_str = f'{dr:.0f}'
        ax.axvline(log2_dr_val, color=color, linewidth=2.0, linestyle='--', alpha=0.9,
                   label=fmt_name + r' ($' + dr_str + r'{:}1$)')
        ax.annotate(fmt_name, xy=(log2_dr_val, y_top * annot_positions[i]),
                    fontsize=6.5, color=color, fontweight='bold',
                    ha='left', va='top',
                    xytext=(3, 0), textcoords='offset points')

    ax.set_xlabel(r'Within-block dynamic range ($\log_2$ max/min)', fontsize=9)
    ax.set_ylabel(r'Density', fontsize=9)
    ax.set_title(r'Weight dynamic range per block vs.\ format capacity', fontsize=10)
    ax.legend(fontsize=6.5, loc='upper right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig('repo/paper/fig_dead_zone_dynamic_range.pdf', bbox_inches='tight', pad_inches=0.03)
    plt.savefig('repo/paper/figures/fig_dead_zone_dynamic_range.png', bbox_inches='tight', dpi=300)
    print("Saved: fig_dead_zone_dynamic_range.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
