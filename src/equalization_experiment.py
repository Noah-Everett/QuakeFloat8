#!/usr/bin/env python3
"""
equalization_experiment.py — Cross-model SQNR stability experiment.

Tests the equalization property: QF8's SQNR should be ~constant across all
models and layers, while FP8/INT8 SQNR varies with weight distribution.

This is the key experiment that turns Theorem 2.1 from math into evidence.
No inference needed — just loads weights, quantizes, measures SQNR.
"""

import gc
import json
import math
import os
import sys
import time

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import OrderedDict

# ======================================================================
# Quantization round-trips (copied from sqnr_benchmark.py for standalone use)
# ======================================================================

BLOCK_SIZE = 32
QF8_BIAS = 64
QF8_FRAC_BITS = 4
QF8_MAX_CODE = 127
QF8_MIN_POSITIVE_MAG = 2.0 ** ((1 - QF8_BIAS) / (1 << QF8_FRAC_BITS))
E8M0_MIN_EXP = -127
E8M0_MAX_EXP = 127
FP8_BIAS = 7
FP8_MAX_VAL = 448.0
FP8_LOG2_MAX = math.log2(FP8_MAX_VAL)


def qf8_roundtrip(x: torch.Tensor) -> torch.Tensor:
    """Block-scaled QF8 quantize-dequantize round-trip."""
    orig_shape = x.shape
    x_flat = x.reshape(-1).float()
    n = x_flat.numel()
    pad = (-n) % BLOCK_SIZE
    if pad > 0:
        x_flat = torch.nn.functional.pad(x_flat, (0, pad))

    blocks = x_flat.reshape(-1, BLOCK_SIZE)
    amax = blocks.abs().amax(dim=-1)
    nz = amax > 0
    exps = torch.zeros_like(amax)
    exps[nz] = torch.ceil(torch.log2(amax[nz]) - (QF8_MAX_CODE - QF8_BIAS) / 16.0)
    exps = exps.clamp(E8M0_MIN_EXP, E8M0_MAX_EXP)
    scales = torch.exp2(exps).clamp(min=1e-45)

    scaled = blocks / scales.unsqueeze(-1)
    signs = (scaled < 0).float()
    abs_s = scaled.abs()

    log2_v = torch.log2(abs_s.clamp(min=1e-45))
    raw = torch.round(QF8_BIAS + (1 << QF8_FRAC_BITS) * log2_v)
    nonzero_mask = abs_s >= (QF8_MIN_POSITIVE_MAG * 0.5)
    codes = torch.where(nonzero_mask, raw.clamp(1, QF8_MAX_CODE), torch.zeros_like(raw))

    mags = torch.where(codes > 0, torch.exp2((codes - QF8_BIAS) / 16.0), torch.zeros_like(codes))
    mags = mags * scales.unsqueeze(-1)
    result = torch.where(signs.bool(), -mags, mags)
    return result.reshape(-1)[:n].reshape(orig_shape)


def _build_fp8e4m3_lut():
    lut = torch.zeros(128, dtype=torch.float32)
    for c in range(128):
        exp = (c >> 3) & 0xF
        man = c & 0x7
        if exp == 15 and man == 7:
            lut[c] = FP8_MAX_VAL
        elif exp == 0:
            lut[c] = man * 2.0 ** (1 - FP8_BIAS - 3)
        else:
            lut[c] = (1.0 + man / 8.0) * 2.0 ** (exp - FP8_BIAS)
    return lut

_FP8_LUT = _build_fp8e4m3_lut()
_FP8_MID = (_FP8_LUT[:-1] + _FP8_LUT[1:]) / 2.0


def mxfp8_roundtrip(x: torch.Tensor) -> torch.Tensor:
    """Block-scaled FP8 E4M3 quantize-dequantize round-trip."""
    orig_shape = x.shape
    x_flat = x.reshape(-1).float()
    n = x_flat.numel()
    pad = (-n) % BLOCK_SIZE
    if pad > 0:
        x_flat = torch.nn.functional.pad(x_flat, (0, pad))

    blocks = x_flat.reshape(-1, BLOCK_SIZE)
    amax = blocks.abs().amax(dim=-1)
    nz = amax > 0
    exps = torch.zeros_like(amax)
    exps[nz] = torch.ceil(torch.log2(amax[nz]) - FP8_LOG2_MAX)
    exps = exps.clamp(E8M0_MIN_EXP, E8M0_MAX_EXP)
    scales = torch.exp2(exps).clamp(min=1e-45)

    scaled = blocks / scales.unsqueeze(-1)
    signs = torch.sign(scaled)
    abs_s = scaled.abs()

    fp8_mid = _FP8_MID.to(abs_s.device)
    fp8_lut = _FP8_LUT.to(abs_s.device)
    codes = torch.searchsorted(fp8_mid, abs_s.reshape(-1))
    codes = codes.clamp(0, 126)

    mags = fp8_lut[codes].reshape(abs_s.shape) * scales.unsqueeze(-1)
    result = signs * mags
    return result.reshape(-1)[:n].reshape(orig_shape)


def int8_roundtrip(x: torch.Tensor) -> torch.Tensor:
    """Block-scaled symmetric INT8 quantize-dequantize round-trip."""
    orig_shape = x.shape
    x_flat = x.reshape(-1).float()
    n = x_flat.numel()
    pad = (-n) % BLOCK_SIZE
    if pad > 0:
        x_flat = torch.nn.functional.pad(x_flat, (0, pad))

    blocks = x_flat.reshape(-1, BLOCK_SIZE)
    amax = blocks.abs().amax(dim=-1, keepdim=True)
    scales = (amax / 127.0).clamp(min=1e-45)
    q = torch.round(blocks / scales).clamp(-127, 127)
    result = q * scales
    return result.reshape(-1)[:n].reshape(orig_shape)


def compute_sqnr(original: torch.Tensor, quantized: torch.Tensor) -> float:
    signal_power = (original ** 2).sum().item()
    noise_power = ((original - quantized) ** 2).sum().item()
    if noise_power == 0:
        return float("inf")
    if signal_power == 0:
        return 0.0
    return 10.0 * math.log10(signal_power / noise_power)


# ======================================================================
# Model list — diverse architectures, sizes, training procedures
# ======================================================================

# Organized by family. Each entry: (hf_name, short_label, arch_type)
# All fit in 16GB RAM in float32 except where noted.
MODELS = [
    # GPT-2 family (autoregressive, OpenAI)
    ("openai-community/gpt2",          "GPT2-S",      "GPT"),
    ("openai-community/gpt2-medium",   "GPT2-M",      "GPT"),
    ("openai-community/gpt2-large",    "GPT2-L",      "GPT"),

    # OPT family (autoregressive, Meta)
    ("facebook/opt-125m",     "OPT-125M",    "OPT"),
    ("facebook/opt-350m",     "OPT-350M",    "OPT"),
    ("facebook/opt-1.3b",     "OPT-1.3B",    "OPT"),

    # Pythia family (autoregressive, EleutherAI, different training stages)
    ("EleutherAI/pythia-160m",  "Pythia-160M", "Pythia"),
    ("EleutherAI/pythia-410m",  "Pythia-410M", "Pythia"),
    ("EleutherAI/pythia-1b",    "Pythia-1B",   "Pythia"),

    # BERT family (bidirectional encoder)
    ("google-bert/bert-base-uncased",   "BERT-B",   "BERT"),
    ("google-bert/bert-large-uncased",  "BERT-L",   "BERT"),

    # RoBERTa (different pretraining)
    ("FacebookAI/roberta-base", "RoBERTa-B", "RoBERTa"),

    # DistilBERT (distilled)
    ("distilbert/distilbert-base-uncased", "DistilBERT", "DistilBERT"),

    # T5 (encoder-decoder)
    ("google-t5/t5-small", "T5-S", "T5"),
    ("google-t5/t5-base",  "T5-B", "T5"),
]


def is_weight_tensor(name: str, param: torch.Tensor) -> bool:
    """Filter to 2D+ weight tensors (skip biases, norms, embeddings)."""
    if param.ndim < 2:
        return False
    low = name.lower()
    skip = ["embed", "wte", "wpe", "position", "layernorm", "layer_norm", "ln_"]
    return not any(s in low for s in skip)


def measure_model(model_name: str, label: str) -> list[dict]:
    """Load a model's weights and measure per-layer SQNR for all formats."""
    from transformers import AutoModel, AutoModelForCausalLM

    print(f"\n{'='*60}")
    print(f"  {label} ({model_name})")
    print(f"{'='*60}")

    # Try causal LM first, fall back to base model
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float32,
            trust_remote_code=True, low_cpu_mem_usage=True
        )
    except (ValueError, KeyError):
        model = AutoModel.from_pretrained(
            model_name, torch_dtype=torch.float32,
            trust_remote_code=True, low_cpu_mem_usage=True
        )
    model.eval()

    rows = []
    for name, param in model.named_parameters():
        if not is_weight_tensor(name, param):
            continue

        w = param.data.float()

        # Weight distribution stats
        std = w.std().item()
        abs_vals = w.abs()
        nz = abs_vals[abs_vals > 0]
        log2_range = math.log2(nz.max().item() / nz.min().item()) if len(nz) > 0 else 0
        kurtosis_val = ((w - w.mean()) ** 4).mean().item() / (std ** 4) if std > 0 else 0

        # Quantize
        qf8_w = qf8_roundtrip(w)
        fp8_w = mxfp8_roundtrip(w)
        int8_w = int8_roundtrip(w)

        qf8_sqnr = compute_sqnr(w, qf8_w)
        fp8_sqnr = compute_sqnr(w, fp8_w)
        int8_sqnr = compute_sqnr(w, int8_w)

        row = {
            "model": label,
            "model_hf": model_name,
            "layer": name,
            "numel": w.numel(),
            "std": std,
            "log2_range": log2_range,
            "kurtosis": kurtosis_val,
            "qf8_sqnr": round(qf8_sqnr, 2),
            "mxfp8_sqnr": round(fp8_sqnr, 2),
            "int8_sqnr": round(int8_sqnr, 2),
        }
        rows.append(row)

    qf8_vals = [r["qf8_sqnr"] for r in rows]
    fp8_vals = [r["mxfp8_sqnr"] for r in rows]
    int8_vals = [r["int8_sqnr"] for r in rows]

    print(f"  {len(rows)} weight tensors")
    print(f"  QF8:   mean={np.mean(qf8_vals):.1f}  std={np.std(qf8_vals):.2f}  "
          f"range=[{min(qf8_vals):.1f}, {max(qf8_vals):.1f}]")
    print(f"  MXFP8: mean={np.mean(fp8_vals):.1f}  std={np.std(fp8_vals):.2f}  "
          f"range=[{min(fp8_vals):.1f}, {max(fp8_vals):.1f}]")
    print(f"  INT8:  mean={np.mean(int8_vals):.1f}  std={np.std(int8_vals):.2f}  "
          f"range=[{min(int8_vals):.1f}, {max(int8_vals):.1f}]")

    # Free memory
    del model
    gc.collect()

    return rows


# ======================================================================
# Plotting
# ======================================================================

def plot_equalization(all_rows: list[dict], out_dir: str):
    """Generate the key plots showing the equalization property."""

    models_seen = list(OrderedDict.fromkeys(r["model"] for r in all_rows))

    # --- Plot 1: Per-layer SQNR strip plot across all models ---
    fig, ax = plt.subplots(figsize=(14, 6))

    model_positions = {m: i for i, m in enumerate(models_seen)}
    jitter = 0.08

    for r in all_rows:
        x = model_positions[r["model"]]
        ax.scatter(x - jitter * 3, r["qf8_sqnr"],  color="#2ecc71", s=6, alpha=0.5, zorder=3)
        ax.scatter(x,              r["mxfp8_sqnr"], color="#e74c3c", s=6, alpha=0.5, zorder=3)
        ax.scatter(x + jitter * 3, r["int8_sqnr"],  color="#3498db", s=6, alpha=0.5, zorder=3)

    # Per-model means
    for m in models_seen:
        x = model_positions[m]
        rows_m = [r for r in all_rows if r["model"] == m]
        ax.plot(x - jitter * 3, np.mean([r["qf8_sqnr"] for r in rows_m]),
                "D", color="#27ae60", markersize=7, zorder=5)
        ax.plot(x, np.mean([r["mxfp8_sqnr"] for r in rows_m]),
                "D", color="#c0392b", markersize=7, zorder=5)
        ax.plot(x + jitter * 3, np.mean([r["int8_sqnr"] for r in rows_m]),
                "D", color="#2980b9", markersize=7, zorder=5)

    ax.set_xticks(range(len(models_seen)))
    ax.set_xticklabels(models_seen, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("SQNR (dB)")
    ax.set_title("Per-Layer SQNR Across Models — The Equalization Property")

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2ecc71", markersize=8, label="QF8"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#e74c3c", markersize=8, label="MXFP8 E4M3"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#3498db", markersize=8, label="INT8"),
    ]
    ax.legend(handles=legend_elements, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "equalization_strip.png"), dpi=200)
    fig.savefig(os.path.join(out_dir, "equalization_strip.pdf"))
    plt.close(fig)
    print(f"  Saved equalization_strip.png/pdf")

    # --- Plot 2: SQNR std dev (variance) per model ---
    fig, ax = plt.subplots(figsize=(12, 5))
    qf8_stds, fp8_stds, int8_stds = [], [], []
    for m in models_seen:
        rows_m = [r for r in all_rows if r["model"] == m]
        qf8_stds.append(np.std([r["qf8_sqnr"] for r in rows_m]))
        fp8_stds.append(np.std([r["mxfp8_sqnr"] for r in rows_m]))
        int8_stds.append(np.std([r["int8_sqnr"] for r in rows_m]))

    x = np.arange(len(models_seen))
    w = 0.25
    ax.bar(x - w, qf8_stds,  width=w, color="#2ecc71", label="QF8")
    ax.bar(x,     fp8_stds,  width=w, color="#e74c3c", label="MXFP8 E4M3")
    ax.bar(x + w, int8_stds, width=w, color="#3498db", label="INT8")

    ax.set_xticks(x)
    ax.set_xticklabels(models_seen, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("SQNR Std Dev (dB)")
    ax.set_title("SQNR Variance Across Layers — Lower = More Stable")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "equalization_variance.png"), dpi=200)
    fig.savefig(os.path.join(out_dir, "equalization_variance.pdf"))
    plt.close(fig)
    print(f"  Saved equalization_variance.png/pdf")

    # --- Plot 3: Histogram of all SQNR values across all models ---
    fig, ax = plt.subplots(figsize=(10, 5))
    qf8_all = [r["qf8_sqnr"] for r in all_rows]
    fp8_all = [r["mxfp8_sqnr"] for r in all_rows]
    int8_all = [r["int8_sqnr"] for r in all_rows]

    bins_qf8 = np.linspace(min(qf8_all) - 0.5, max(qf8_all) + 0.5, 40)
    bins_fp8 = np.linspace(min(fp8_all) - 0.5, max(fp8_all) + 0.5, 40)
    bins_int8 = np.linspace(min(int8_all) - 1, max(int8_all) + 1, 40)

    ax.hist(qf8_all, bins=bins_qf8, alpha=0.7, color="#2ecc71", label=f"QF8 (std={np.std(qf8_all):.2f} dB)")
    ax.hist(fp8_all, bins=bins_fp8, alpha=0.7, color="#e74c3c", label=f"MXFP8 (std={np.std(fp8_all):.2f} dB)")
    ax.hist(int8_all, bins=bins_int8, alpha=0.7, color="#3498db", label=f"INT8 (std={np.std(int8_all):.2f} dB)")

    ax.set_xlabel("SQNR (dB)")
    ax.set_ylabel("Count (layers)")
    ax.set_title(f"SQNR Distribution Across All {len(all_rows)} Weight Tensors from {len(models_seen)} Models")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "equalization_histogram.png"), dpi=200)
    fig.savefig(os.path.join(out_dir, "equalization_histogram.pdf"))
    plt.close(fig)
    print(f"  Saved equalization_histogram.png/pdf")

    # --- Plot 4: SQNR vs weight kurtosis (distribution shape) ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for ax_i, (key, label, color) in zip(axes, [
        ("qf8_sqnr", "QF8", "#2ecc71"),
        ("mxfp8_sqnr", "MXFP8 E4M3", "#e74c3c"),
        ("int8_sqnr", "INT8", "#3498db"),
    ]):
        kurtoses = [r["kurtosis"] for r in all_rows]
        sqnrs = [r[key] for r in all_rows]
        ax_i.scatter(kurtoses, sqnrs, s=8, alpha=0.5, color=color)
        ax_i.set_xlabel("Kurtosis")
        ax_i.set_title(label)
        ax_i.set_xscale("log")
        ax_i.grid(alpha=0.3)
        # Correlation
        if len(kurtoses) > 2:
            log_k = np.log10(np.clip(kurtoses, 1e-6, None))
            corr = np.corrcoef(log_k, sqnrs)[0, 1]
            ax_i.text(0.05, 0.95, f"r = {corr:.3f}", transform=ax_i.transAxes,
                     va="top", fontsize=10, bbox=dict(boxstyle="round", fc="white", alpha=0.8))

    axes[0].set_ylabel("SQNR (dB)")
    fig.suptitle("SQNR vs Weight Distribution Shape (Kurtosis)", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "equalization_vs_kurtosis.png"), dpi=200)
    fig.savefig(os.path.join(out_dir, "equalization_vs_kurtosis.pdf"))
    plt.close(fig)
    print(f"  Saved equalization_vs_kurtosis.png/pdf")


def print_summary(all_rows: list[dict]):
    """Print the headline numbers."""
    models_seen = list(OrderedDict.fromkeys(r["model"] for r in all_rows))

    qf8_all = [r["qf8_sqnr"] for r in all_rows]
    fp8_all = [r["mxfp8_sqnr"] for r in all_rows]
    int8_all = [r["int8_sqnr"] for r in all_rows]

    print(f"\n{'='*70}")
    print(f"  EQUALIZATION EXPERIMENT SUMMARY")
    print(f"  {len(all_rows)} weight tensors from {len(models_seen)} models")
    print(f"{'='*70}")
    print(f"                    Mean     Std      Min      Max      Range")
    print(f"  QF8:            {np.mean(qf8_all):6.1f}   {np.std(qf8_all):5.2f}   "
          f"{min(qf8_all):6.1f}   {max(qf8_all):6.1f}   {max(qf8_all)-min(qf8_all):5.2f}")
    print(f"  MXFP8 E4M3:     {np.mean(fp8_all):6.1f}   {np.std(fp8_all):5.2f}   "
          f"{min(fp8_all):6.1f}   {max(fp8_all):6.1f}   {max(fp8_all)-min(fp8_all):5.2f}")
    print(f"  INT8:           {np.mean(int8_all):6.1f}   {np.std(int8_all):5.2f}   "
          f"{min(int8_all):6.1f}   {max(int8_all):6.1f}   {max(int8_all)-min(int8_all):5.2f}")
    print()
    print(f"  QF8 advantage over MXFP8:  {np.mean(qf8_all) - np.mean(fp8_all):+.1f} dB (mean)")
    print()
    print(f"  Key finding: QF8 std = {np.std(qf8_all):.2f} dB across ALL layers and models.")
    print(f"  Compare:     MXFP8 std = {np.std(fp8_all):.2f} dB, INT8 std = {np.std(int8_all):.2f} dB.")
    if np.std(qf8_all) < np.std(fp8_all) and np.std(qf8_all) < np.std(int8_all):
        print(f"  --> QF8 is the most stable format (equalization property confirmed).")
    print(f"{'='*70}")


# ======================================================================
# Main
# ======================================================================

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_dir = os.path.dirname(script_dir)
    out_dir = os.path.join(repo_dir, "results", "equalization")
    os.makedirs(out_dir, exist_ok=True)

    # Allow subset via CLI: python equalization_experiment.py 0 5
    # (runs models[0:5] only, for quick testing)
    models = MODELS
    if len(sys.argv) >= 3:
        start, end = int(sys.argv[1]), int(sys.argv[2])
        models = MODELS[start:end]
        print(f"Running subset: models[{start}:{end}]")
    elif len(sys.argv) == 2 and sys.argv[1] == "--quick":
        # Quick test: just 4 diverse models
        quick = ["openai-community/gpt2", "google-bert/bert-base-uncased",
                 "google-t5/t5-small", "facebook/opt-125m"]
        models = [m for m in MODELS if m[0] in quick]
        print("Running quick test (4 models)")

    all_rows = []
    t0 = time.time()

    for model_name, label, arch in models:
        try:
            rows = measure_model(model_name, label)
            all_rows.extend(rows)
        except Exception as e:
            print(f"  FAILED: {e}")
            continue

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.0f}s ({elapsed/60:.1f} min)")

    if not all_rows:
        print("No results collected!")
        return

    # Save raw data
    json_path = os.path.join(out_dir, "equalization_results.json")
    with open(json_path, "w") as f:
        json.dump(all_rows, f, indent=2)
    print(f"Raw data saved to {json_path}")

    # Summary
    print_summary(all_rows)

    # Plots
    print("\nGenerating plots...")
    plot_equalization(all_rows, out_dir)

    print(f"\nAll outputs in {out_dir}/")


if __name__ == "__main__":
    main()
