#!/usr/bin/env python3
"""
ptq_scale.py — Scaling PTQ perplexity + SQNR experiment for 7B+ models.

Runs weight-only post-training quantization on large models (LLaMA-2, Mistral,
Phi-2, etc.) and measures both SQNR and WikiText-2 perplexity.

Designed for SLURM / HPC with GPUs. Uses float16 inference, quantizes in float32
for accuracy, streams large models with low_cpu_mem_usage=True.

Usage:
    python src/ptq_scale.py                              # default model list
    python src/ptq_scale.py --models meta-llama/Llama-2-7b-hf
    python src/ptq_scale.py --sqnr-only                  # skip perplexity (fast)
    python src/ptq_scale.py --models all                 # everything
"""

import argparse
import gc
import json
import math
import os
import sys
import time

import torch
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict

# ======================================================================
# Quantization round-trips (self-contained, same as sqnr_benchmark.py)
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
    orig_shape = x.shape
    x_flat = x.reshape(-1).float()
    n = x_flat.numel()
    pad = (-n) % BLOCK_SIZE
    if pad > 0:
        x_flat = F.pad(x_flat, (0, pad))
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
    orig_shape = x.shape
    x_flat = x.reshape(-1).float()
    n = x_flat.numel()
    pad = (-n) % BLOCK_SIZE
    if pad > 0:
        x_flat = F.pad(x_flat, (0, pad))
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
    orig_shape = x.shape
    x_flat = x.reshape(-1).float()
    n = x_flat.numel()
    pad = (-n) % BLOCK_SIZE
    if pad > 0:
        x_flat = F.pad(x_flat, (0, pad))
    blocks = x_flat.reshape(-1, BLOCK_SIZE)
    amax = blocks.abs().amax(dim=-1, keepdim=True)
    scales = (amax / 127.0).clamp(min=1e-45)
    q = torch.round(blocks / scales).clamp(-127, 127)
    result = q * scales
    return result.reshape(-1)[:n].reshape(orig_shape)


def compute_sqnr(original: torch.Tensor, quantized: torch.Tensor) -> float:
    signal_power = (original.float() ** 2).sum().item()
    noise_power = ((original.float() - quantized.float()) ** 2).sum().item()
    if noise_power == 0:
        return float("inf")
    if signal_power == 0:
        return 0.0
    return 10.0 * math.log10(signal_power / noise_power)


# ======================================================================
# Model definitions
# ======================================================================

# (hf_id, short_label, model_type)
# model_type: "causal" or "seq2seq"
MODELS_DEFAULT = [
    ("microsoft/phi-2",                    "Phi-2 (2.7B)",       "causal"),
    ("mistralai/Mistral-7B-v0.1",         "Mistral-7B",         "causal"),
    ("tiiuae/falcon-7b",                   "Falcon-7B",          "causal"),
    ("EleutherAI/pythia-6.9b",             "Pythia-6.9B",        "causal"),
]

MODELS_ALL = MODELS_DEFAULT + [
    ("mistralai/Mistral-7B-Instruct-v0.1", "Mistral-7B-Inst",   "causal"),
    ("EleutherAI/pythia-12b",               "Pythia-12B",        "causal"),
    ("tiiuae/falcon-40b",                   "Falcon-40B",        "causal"),
]


def is_weight_tensor(name: str, param: torch.Tensor) -> bool:
    if param.ndim < 2:
        return False
    low = name.lower()
    skip = ["embed", "wte", "wpe", "position", "layernorm", "layer_norm",
            "ln_", "norm", "lm_head"]
    return not any(s in low for s in skip)


# ======================================================================
# SQNR measurement (no inference needed)
# ======================================================================

def measure_sqnr(model_name: str, label: str) -> list[dict]:
    """Load model weights and measure per-layer SQNR for all formats."""
    from transformers import AutoModelForCausalLM

    print(f"\n{'='*60}")
    print(f"  SQNR: {label} ({model_name})")
    print(f"{'='*60}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.float32,
        low_cpu_mem_usage=True, trust_remote_code=True,
    )
    model.eval()

    rows = []
    for name, param in model.named_parameters():
        if not is_weight_tensor(name, param):
            continue
        w = param.data.float()
        qf8_sqnr = compute_sqnr(w, qf8_roundtrip(w))
        fp8_sqnr = compute_sqnr(w, mxfp8_roundtrip(w))
        int8_sqnr = compute_sqnr(w, int8_roundtrip(w))
        rows.append({
            "model": label, "layer": name, "numel": w.numel(),
            "qf8_sqnr": round(qf8_sqnr, 2),
            "mxfp8_sqnr": round(fp8_sqnr, 2),
            "int8_sqnr": round(int8_sqnr, 2),
        })

    qf8_vals = [r["qf8_sqnr"] for r in rows]
    fp8_vals = [r["mxfp8_sqnr"] for r in rows]
    int8_vals = [r["int8_sqnr"] for r in rows]
    print(f"  {len(rows)} tensors")
    print(f"  QF8:   mean={np.mean(qf8_vals):.1f}  std={np.std(qf8_vals):.2f}")
    print(f"  MXFP8: mean={np.mean(fp8_vals):.1f}  std={np.std(fp8_vals):.2f}")
    print(f"  INT8:  mean={np.mean(int8_vals):.1f}  std={np.std(int8_vals):.2f}")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return rows


# ======================================================================
# PTQ perplexity
# ======================================================================

def quantize_model_weights(model, quant_fn):
    """Apply quantization round-trip to all linear layer weights in-place."""
    n_quantized = 0
    with torch.no_grad():
        for name, param in model.named_parameters():
            if not is_weight_tensor(name, param):
                continue
            # Quantize in float32 for accuracy, then cast back
            w32 = param.data.float()
            q32 = quant_fn(w32)
            param.data.copy_(q32.to(param.dtype))
            n_quantized += 1
    print(f"  Quantized {n_quantized} weight tensors")


def load_wikitext2(tokenizer_name: str):
    """Load WikiText-2 test set, tokenized with the model's tokenizer."""
    from datasets import load_dataset
    from transformers import AutoTokenizer

    print(f"Loading WikiText-2 (tokenizer: {tokenizer_name})...")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n".join(t for t in ds["text"] if t.strip())

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    tokens = tokenizer.encode(text)
    print(f"  {len(tokens):,} tokens")
    return torch.tensor(tokens, dtype=torch.long)


@torch.no_grad()
def evaluate_perplexity(model, tokens, device, stride=512, max_length=None):
    """Sliding-window perplexity evaluation."""
    model.eval()

    # When using device_map="auto", inputs go to the model's input device
    if hasattr(model, "hf_device_map"):
        input_device = next(model.parameters()).device
    else:
        input_device = device

    if max_length is None:
        # Use model's max position embeddings
        config = model.config
        max_length = getattr(config, "max_position_embeddings", 2048)
        max_length = min(max_length, 2048)  # cap for memory

    n = tokens.numel()
    nlls = []
    counts = []
    t0 = time.time()
    n_windows = 0

    for begin in range(0, n - 1, stride):
        end = min(begin + max_length, n)
        input_ids = tokens[begin:end].unsqueeze(0).to(input_device)
        target_ids = input_ids.clone()

        if begin > 0:
            target_ids[:, :-stride] = -100

        outputs = model(input_ids, labels=target_ids)
        n_valid = (target_ids != -100).sum().item()
        nlls.append(outputs.loss.item() * n_valid)
        counts.append(n_valid)
        n_windows += 1

        if n_windows % 20 == 0:
            elapsed = time.time() - t0
            ppl_so_far = math.exp(sum(nlls) / sum(counts)) if sum(counts) > 0 else 0
            print(f"    {n_windows} windows, {elapsed:.0f}s, ppl~{ppl_so_far:.2f}")

    total_nll = sum(nlls)
    total_count = sum(counts)
    ppl = math.exp(total_nll / total_count) if total_count > 0 else float("inf")

    elapsed = time.time() - t0
    print(f"    Done: {n_windows} windows, {total_count} tokens, {elapsed:.0f}s")
    return ppl


def run_ptq_perplexity(model_name: str, label: str, device: str) -> list[dict]:
    """Run PTQ perplexity for one model across all formats."""
    from transformers import AutoModelForCausalLM

    # Determine tokenizer (usually same as model)
    tokenizer_name = model_name
    test_tokens = load_wikitext2(tokenizer_name)

    # Inference dtype: float16 on GPU for speed+memory, float32 on CPU
    infer_dtype = torch.float16 if device in ("cuda", "mps") else torch.float32

    quantizers = OrderedDict([
        ("FP16 baseline", None),
        ("QF8", qf8_roundtrip),
        ("MXFP8 E4M3", mxfp8_roundtrip),
        ("INT8", int8_roundtrip),
    ])

    results = []
    for qname, qfn in quantizers.items():
        print(f"\n{'='*60}")
        print(f"  PPL: {label} — {qname}")
        print(f"{'='*60}")

        model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=infer_dtype,
            low_cpu_mem_usage=True, trust_remote_code=True,
            device_map="auto" if device == "cuda" else None,
        )
        model.eval()

        if qfn is not None:
            print(f"  Quantizing weights...")
            quantize_model_weights(model, qfn)

        if device != "cuda":
            model.to(device)

        ppl = evaluate_perplexity(model, test_tokens, device)

        results.append({
            "model": label,
            "model_hf": model_name,
            "format": qname,
            "perplexity": round(ppl, 4),
        })
        print(f"  >>> {qname}: perplexity = {ppl:.4f}")

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Print comparison
    baseline_ppl = results[0]["perplexity"]
    print(f"\n  {label} PTQ Summary:")
    for r in results:
        delta = r["perplexity"] - baseline_ppl
        pct = delta / baseline_ppl * 100 if baseline_ppl else 0
        tag = "" if r["format"] == "FP16 baseline" else f"  ({pct:+.2f}%)"
        print(f"    {r['format']:<18} {r['perplexity']:.4f}{tag}")

    return results


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description="Scaling PTQ experiment")
    parser.add_argument("--models", nargs="*", default=None,
                        help="HF model IDs. Use 'all' for full list, 'default' for standard set.")
    parser.add_argument("--sqnr-only", action="store_true",
                        help="Only run SQNR measurement, skip perplexity")
    parser.add_argument("--ppl-only", action="store_true",
                        help="Only run perplexity, skip SQNR")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: repo/results/scaling/)")
    args = parser.parse_args()

    # Resolve model list
    if args.models is None or args.models == ["default"] or args.models == []:
        models = MODELS_DEFAULT
    elif args.models == ["all"]:
        models = MODELS_ALL
    else:
        # User-specified models
        models = [(m, m.split("/")[-1], "causal") for m in args.models]

    # Device
    if torch.cuda.is_available():
        device = "cuda"
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Device: {device}")

    # Output dir
    if args.output_dir:
        out_dir = args.output_dir
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        out_dir = os.path.join(os.path.dirname(script_dir), "results", "scaling")
    os.makedirs(out_dir, exist_ok=True)

    all_sqnr = []
    all_ppl = []
    t0 = time.time()

    for model_name, label, model_type in models:
        try:
            if not args.ppl_only:
                sqnr_rows = measure_sqnr(model_name, label)
                all_sqnr.extend(sqnr_rows)

            if not args.sqnr_only:
                ppl_rows = run_ptq_perplexity(model_name, label, device)
                all_ppl.extend(ppl_rows)

        except Exception as e:
            print(f"\n  FAILED on {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.0f}s ({elapsed/3600:.1f} hr)")

    # Save results
    if all_sqnr:
        path = os.path.join(out_dir, "scaling_sqnr.json")
        with open(path, "w") as f:
            json.dump(all_sqnr, f, indent=2)
        print(f"SQNR results: {path}")

        # Summary
        qf8 = [r["qf8_sqnr"] for r in all_sqnr]
        fp8 = [r["mxfp8_sqnr"] for r in all_sqnr]
        int8 = [r["int8_sqnr"] for r in all_sqnr]
        print(f"\nSQNR Summary ({len(all_sqnr)} tensors):")
        print(f"  QF8:   {np.mean(qf8):.1f} ± {np.std(qf8):.2f} dB")
        print(f"  MXFP8: {np.mean(fp8):.1f} ± {np.std(fp8):.2f} dB")
        print(f"  INT8:  {np.mean(int8):.1f} ± {np.std(int8):.2f} dB")

    if all_ppl:
        path = os.path.join(out_dir, "scaling_ppl.json")
        with open(path, "w") as f:
            json.dump(all_ppl, f, indent=2)
        print(f"PPL results: {path}")

        # Summary table
        print(f"\nPerplexity Summary (WikiText-2):")
        print(f"  {'Model':<20} {'Baseline':>10} {'QF8':>10} {'MXFP8':>10} {'INT8':>10}")
        models_seen = list(OrderedDict.fromkeys(r["model"] for r in all_ppl))
        for m in models_seen:
            mr = {r["format"]: r["perplexity"] for r in all_ppl if r["model"] == m}
            bl = mr.get("FP16 baseline", 0)
            print(f"  {m:<20} {bl:>10.2f} {mr.get('QF8', 0):>10.2f} "
                  f"{mr.get('MXFP8 E4M3', 0):>10.2f} {mr.get('INT8', 0):>10.2f}")


if __name__ == "__main__":
    main()
