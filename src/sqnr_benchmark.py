#!/usr/bin/env python3
"""
sqnr_benchmark.py — Per-layer SQNR benchmark on real pretrained GPT-2 weights.

Compares QF8 vs MXFP8 (block-scaled FP8 E4M3) vs INT8 quantization.
Reports SQNR, clip rate, and zero fraction per layer.

No training needed — just loads pretrained weights and measures quantization fidelity.
"""

import math
import torch
import json
import sys
from collections import OrderedDict

# ======================================================================
# Configuration
# ======================================================================

BLOCK_SIZE = 32

# QF8 parameters
QF8_BIAS = 64
QF8_FRAC_BITS = 4
QF8_MAX_CODE = 127
QF8_MIN_POSITIVE_MAG = 2.0 ** ((1 - QF8_BIAS) / (1 << QF8_FRAC_BITS))

# E8M0 block scaling shared parameters
E8M0_MIN_EXP = -127
E8M0_MAX_EXP = 127

# FP8 E4M3 parameters
FP8_BIAS = 7
FP8_MAX_VAL = 448.0
FP8_LOG2_MAX = math.log2(FP8_MAX_VAL)


# ======================================================================
# QF8 Round-Trip
# ======================================================================

def qf8_roundtrip(x: torch.Tensor) -> tuple[torch.Tensor, dict]:
    """Block-scaled QF8 round-trip. Returns (quantized, diagnostics)."""
    orig_shape = x.shape
    x_flat = x.reshape(-1).float()
    n = x_flat.numel()

    # Pad to multiple of BLOCK_SIZE
    pad = (-n) % BLOCK_SIZE
    if pad > 0:
        x_flat = torch.nn.functional.pad(x_flat, (0, pad))

    n_blocks = x_flat.numel() // BLOCK_SIZE
    blocks = x_flat.reshape(n_blocks, BLOCK_SIZE)

    # E8M0 shared scale per block
    amax = blocks.abs().amax(dim=-1)
    nz = amax > 0
    exps = torch.zeros_like(amax)
    exps[nz] = torch.ceil(torch.log2(amax[nz]) - (QF8_MAX_CODE - QF8_BIAS) / 16.0)
    exps = exps.clamp(E8M0_MIN_EXP, E8M0_MAX_EXP)
    scales = torch.exp2(exps)

    safe_scales = scales.clamp(min=1e-45)
    scaled = blocks / safe_scales.unsqueeze(-1)

    signs = (scaled < 0).float()
    abs_s = scaled.abs()

    # Encode
    log2_v = torch.log2(abs_s.clamp(min=1e-45))
    raw = torch.round(QF8_BIAS + (1 << QF8_FRAC_BITS) * log2_v)

    threshold = QF8_MIN_POSITIVE_MAG * 0.5
    nonzero_mask = abs_s >= threshold
    codes = torch.where(nonzero_mask, raw.clamp(1, QF8_MAX_CODE), torch.zeros_like(raw))

    # Diagnostics: count clips and zeros (exclude padding)
    valid = torch.zeros_like(codes, dtype=torch.bool)
    valid.reshape(-1)[:n] = True
    clip_high = (raw > QF8_MAX_CODE) & nonzero_mask & valid
    clip_low = (~nonzero_mask) & (abs_s > 0) & valid
    zero_codes = (codes == 0) & valid

    # Decode
    mags_unscaled = torch.exp2((codes - QF8_BIAS) / 16.0)
    mags_unscaled = torch.where(codes > 0, mags_unscaled, torch.zeros_like(mags_unscaled))
    mags = mags_unscaled * safe_scales.unsqueeze(-1)
    result = torch.where(signs.bool(), -mags, mags)

    result = result.reshape(-1)[:n].reshape(orig_shape)

    diag = {
        "clip_high_rate": clip_high.sum().item() / n,
        "clip_low_rate": clip_low.sum().item() / n,
        "zero_code_rate": zero_codes.sum().item() / n,
    }
    return result, diag


# ======================================================================
# MXFP8 E4M3 Round-Trip (block-scaled)
# ======================================================================

def _build_fp8e4m3_lut():
    """Build all 128 positive FP8-E4M3 magnitudes."""
    lut = torch.zeros(128, dtype=torch.float32)
    for c in range(128):
        exp = (c >> 3) & 0xF
        man = c & 0x7
        if exp == 15 and man == 7:
            lut[c] = FP8_MAX_VAL  # NaN -> clamp to max
        elif exp == 0:
            lut[c] = man * 2.0 ** (1 - FP8_BIAS - 3)  # subnormal
        else:
            lut[c] = (1.0 + man / 8.0) * 2.0 ** (exp - FP8_BIAS)
    return lut


_FP8_LUT = _build_fp8e4m3_lut()
_FP8_MID = (_FP8_LUT[:-1] + _FP8_LUT[1:]) / 2.0


def mxfp8_roundtrip(x: torch.Tensor) -> tuple[torch.Tensor, dict]:
    """Block-scaled FP8 E4M3 round-trip. Returns (quantized, diagnostics)."""
    orig_shape = x.shape
    x_flat = x.reshape(-1).float()
    n = x_flat.numel()

    pad = (-n) % BLOCK_SIZE
    if pad > 0:
        x_flat = torch.nn.functional.pad(x_flat, (0, pad))

    n_blocks = x_flat.numel() // BLOCK_SIZE
    blocks = x_flat.reshape(n_blocks, BLOCK_SIZE)

    # E8M0 shared scale
    amax = blocks.abs().amax(dim=-1)
    nz = amax > 0
    exps = torch.zeros_like(amax)
    exps[nz] = torch.ceil(torch.log2(amax[nz]) - FP8_LOG2_MAX)
    exps = exps.clamp(E8M0_MIN_EXP, E8M0_MAX_EXP)
    scales = torch.exp2(exps)

    safe_scales = scales.clamp(min=1e-45)
    scaled = blocks / safe_scales.unsqueeze(-1)

    signs = torch.sign(scaled)
    abs_s = scaled.abs()

    # Encode via searchsorted on midpoints (move LUT to input device)
    fp8_mid = _FP8_MID.to(abs_s.device)
    fp8_lut = _FP8_LUT.to(abs_s.device)
    flat_abs = abs_s.reshape(-1)
    codes = torch.searchsorted(fp8_mid, flat_abs)
    codes = codes.clamp(0, 126)

    # Diagnostics (exclude padding)
    valid = torch.zeros(flat_abs.numel(), dtype=torch.bool, device=flat_abs.device)
    valid[:n] = True
    clip_high = (flat_abs > FP8_MAX_VAL) & valid
    zero_codes = (codes == 0) & valid

    # Decode
    mags_unscaled = fp8_lut[codes].reshape(abs_s.shape)
    mags = mags_unscaled * safe_scales.unsqueeze(-1)
    result = signs * mags

    result = result.reshape(-1)[:n].reshape(orig_shape)

    diag = {
        "clip_high_rate": clip_high.sum().item() / n,
        "clip_low_rate": 0.0,
        "zero_code_rate": zero_codes.sum().item() / n,
    }
    return result, diag


# ======================================================================
# INT8 Round-Trip (block-scaled, symmetric)
# ======================================================================

def int8_roundtrip(x: torch.Tensor) -> tuple[torch.Tensor, dict]:
    """Block-scaled symmetric INT8 round-trip. Returns (quantized, diagnostics)."""
    orig_shape = x.shape
    x_flat = x.reshape(-1).float()
    n = x_flat.numel()

    pad = (-n) % BLOCK_SIZE
    if pad > 0:
        x_flat = torch.nn.functional.pad(x_flat, (0, pad))

    n_blocks = x_flat.numel() // BLOCK_SIZE
    blocks = x_flat.reshape(n_blocks, BLOCK_SIZE)

    # Per-block scale: amax / 127
    amax = blocks.abs().amax(dim=-1, keepdim=True)
    scales = amax / 127.0
    safe_scales = scales.clamp(min=1e-45)

    # Quantize
    q = torch.round(blocks / safe_scales).clamp(-127, 127)

    # Diagnostics (exclude padding)
    valid = torch.zeros_like(q, dtype=torch.bool)
    valid.reshape(-1)[:n] = True
    clip_count = (((blocks.abs() / safe_scales) > 127.5) & valid).sum().item()
    zero_count = ((q == 0) & valid).sum().item()

    # Dequantize
    result = q * safe_scales
    result = result.reshape(-1)[:n].reshape(orig_shape)

    diag = {
        "clip_high_rate": clip_count / n,
        "clip_low_rate": 0.0,
        "zero_code_rate": zero_count / n,
    }
    return result, diag


# ======================================================================
# SQNR Computation
# ======================================================================

def compute_sqnr(original: torch.Tensor, quantized: torch.Tensor) -> float:
    """Compute Signal-to-Quantization-Noise Ratio in dB."""
    signal_power = (original ** 2).sum().item()
    noise_power = ((original - quantized) ** 2).sum().item()
    if noise_power == 0:
        return float("inf")
    if signal_power == 0:
        return 0.0
    return 10.0 * math.log10(signal_power / noise_power)


def compute_msre(original: torch.Tensor, quantized: torch.Tensor) -> float:
    """Compute Mean Squared Relative Error (on nonzero elements)."""
    mask = original.abs() > 1e-30
    if mask.sum() == 0:
        return 0.0
    rel_err = ((original[mask] - quantized[mask]) / original[mask]) ** 2
    return rel_err.mean().item()


# ======================================================================
# Main Benchmark
# ======================================================================

def benchmark_model(model_name: str = "gpt2"):
    """Load a pretrained model and benchmark all weight tensors."""
    from transformers import AutoModelForCausalLM

    print(f"\nLoading {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    model.eval()

    quantizers = OrderedDict([
        ("QF8", qf8_roundtrip),
        ("MXFP8", mxfp8_roundtrip),
        ("INT8", int8_roundtrip),
    ])

    results = []

    # Only benchmark weight tensors (not biases, layernorms, embeddings)
    for name, param in model.named_parameters():
        if param.ndim < 2:
            continue  # skip biases, layernorm scales
        if "embed" in name.lower() or "wte" in name.lower() or "wpe" in name.lower():
            continue  # skip embeddings

        w = param.data.float()
        row = {"layer": name, "shape": list(w.shape), "numel": w.numel()}

        # Basic stats
        row["mean"] = w.mean().item()
        row["std"] = w.std().item()
        row["min"] = w.min().item()
        row["max"] = w.max().item()
        row["log2_range"] = math.log2(w.abs().max().item() / w.abs()[w.abs() > 0].min().item()) if (w.abs() > 0).any() else 0

        for qname, qfn in quantizers.items():
            q_w, diag = qfn(w)
            sqnr = compute_sqnr(w, q_w)
            msre = compute_msre(w, q_w)
            row[f"{qname}_sqnr"] = round(sqnr, 2)
            row[f"{qname}_msre"] = msre
            row[f"{qname}_clip_high"] = diag["clip_high_rate"]
            row[f"{qname}_clip_low"] = diag["clip_low_rate"]
            row[f"{qname}_zero_rate"] = diag["zero_code_rate"]

        # Advantage
        row["QF8_vs_MXFP8_dB"] = round(row["QF8_sqnr"] - row["MXFP8_sqnr"], 2)
        row["QF8_vs_INT8_dB"] = round(row["QF8_sqnr"] - row["INT8_sqnr"], 2)

        results.append(row)
        print(f"  {name:50s}  QF8={row['QF8_sqnr']:6.1f}  MXFP8={row['MXFP8_sqnr']:6.1f}  "
              f"INT8={row['INT8_sqnr']:6.1f}  Δ(QF8-MXFP8)={row['QF8_vs_MXFP8_dB']:+5.1f} dB")

    # Summary
    qf8_sqnrs = [r["QF8_sqnr"] for r in results]
    mxfp8_sqnrs = [r["MXFP8_sqnr"] for r in results]
    int8_sqnrs = [r["INT8_sqnr"] for r in results]
    deltas = [r["QF8_vs_MXFP8_dB"] for r in results]

    print(f"\n{'='*70}")
    print(f"SUMMARY: {model_name} ({len(results)} weight tensors)")
    print(f"{'='*70}")
    print(f"  QF8  SQNR:  mean={sum(qf8_sqnrs)/len(qf8_sqnrs):.1f}  "
          f"min={min(qf8_sqnrs):.1f}  max={max(qf8_sqnrs):.1f}")
    print(f"  MXFP8 SQNR: mean={sum(mxfp8_sqnrs)/len(mxfp8_sqnrs):.1f}  "
          f"min={min(mxfp8_sqnrs):.1f}  max={max(mxfp8_sqnrs):.1f}")
    print(f"  INT8 SQNR:  mean={sum(int8_sqnrs)/len(int8_sqnrs):.1f}  "
          f"min={min(int8_sqnrs):.1f}  max={max(int8_sqnrs):.1f}")
    print(f"  QF8 vs MXFP8: mean={sum(deltas)/len(deltas):+.1f}  "
          f"min={min(deltas):+.1f}  max={max(deltas):+.1f}")
    print(f"  QF8 vs INT8:  mean={sum(r['QF8_vs_INT8_dB'] for r in results)/len(results):+.1f}")

    # Clipping summary
    qf8_clips = [r["QF8_clip_high"] for r in results]
    mxfp8_clips = [r["MXFP8_clip_high"] for r in results]
    print(f"\n  Clip rates (high):")
    print(f"    QF8:   mean={sum(qf8_clips)/len(qf8_clips):.4%}  max={max(qf8_clips):.4%}")
    print(f"    MXFP8: mean={sum(mxfp8_clips)/len(mxfp8_clips):.4%}  max={max(mxfp8_clips):.4%}")

    return results


def main():
    models = ["gpt2"]  # Start with GPT-2 Small (124M)
    if len(sys.argv) > 1:
        models = sys.argv[1:]

    all_results = {}
    for model_name in models:
        results = benchmark_model(model_name)
        all_results[model_name] = results

    # Save raw results
    import os
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, "sqnr_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nRaw results saved to {out_path}")


if __name__ == "__main__":
    main()
