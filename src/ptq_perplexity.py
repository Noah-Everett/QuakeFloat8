#!/usr/bin/env python3
"""
ptq_perplexity.py — Post-Training Quantization perplexity eval.

Loads pretrained GPT-2 models, quantizes weights (W-only) to QF8 / MXFP8 / INT8,
and measures perplexity on WikiText-2. No training — purely inference eval.

Usage:
    python src/ptq_perplexity.py                          # GPT-2 Small only
    python src/ptq_perplexity.py gpt2 gpt2-medium gpt2-large  # All three
"""

import math
import time
import json
import sys
import copy
import torch
import torch.nn.functional as F
from collections import OrderedDict

# Import quantization round-trips from the SQNR benchmark
from sqnr_benchmark import qf8_roundtrip, mxfp8_roundtrip, int8_roundtrip


# ======================================================================
# Weight Quantization: Apply round-trip to all GEMM weights
# ======================================================================

def quantize_model_weights(model, quant_fn):
    """Apply quantization round-trip to all linear layer weights in-place.

    Quantizes the same layers as the SQNR benchmark: all 2D+ weight tensors
    except embeddings. Biases, LayerNorm, and embeddings stay in FP32.

    Returns diagnostics dict with per-layer clip/zero rates.
    """
    diagnostics = {}
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.ndim < 2:
                continue
            if "embed" in name.lower() or "wte" in name.lower() or "wpe" in name.lower():
                continue

            q_w, diag = quant_fn(param.data)
            param.data.copy_(q_w)
            diagnostics[name] = diag

    return diagnostics


# ======================================================================
# Perplexity Evaluation
# ======================================================================

def load_wikitext2():
    """Load WikiText-2 test set, return as a single token tensor."""
    from datasets import load_dataset
    from transformers import AutoTokenizer

    print("Loading WikiText-2 test set...")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

    # Join all text
    text = "\n".join(t for t in ds["text"] if t.strip())

    # Use GPT-2 tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokens = tokenizer.encode(text)
    return torch.tensor(tokens, dtype=torch.long)


@torch.no_grad()
def evaluate_perplexity(model, tokens, device, stride=512, max_length=1024):
    """Evaluate perplexity using sliding window.

    Args:
        model: HuggingFace causal LM
        tokens: 1D tensor of token IDs
        device: torch device
        stride: how far to slide the window each step
        max_length: context window size (GPT-2 = 1024)

    Returns:
        perplexity (float), total neg-log-likelihood, token count
    """
    model.eval()
    model.to(device)

    n = tokens.numel()
    nlls = []

    t0 = time.time()
    n_windows = 0

    for begin in range(0, n - 1, stride):
        end = min(begin + max_length, n)
        input_ids = tokens[begin:end].unsqueeze(0).to(device)
        target_ids = input_ids.clone()

        # Only compute loss on the stride portion (avoid double-counting)
        if begin > 0:
            target_ids[:, :-stride] = -100

        outputs = model(input_ids, labels=target_ids)
        nlls.append(outputs.loss.item() * (target_ids != -100).sum().item())
        n_windows += 1

        if n_windows % 50 == 0:
            elapsed = time.time() - t0
            print(f"    {n_windows} windows, {elapsed:.1f}s elapsed...")

    total_nll = sum(nlls)
    total_tokens = sum(
        (tokens[begin:min(begin + max_length, n)].numel() - (max_length - stride if begin > 0 else 0))
        for begin in range(0, n - 1, stride)
    )
    # More accurate: count tokens that contributed to loss
    # total_tokens ≈ n - 1 (each token predicted once via sliding window)
    # But with stride < max_length, some tokens are predicted multiple times
    # Use the standard formula: ppl = exp(total_nll / count)
    count = sum(
        min(stride, min(begin + max_length, n) - begin - 1) if begin > 0
        else min(max_length, n) - 1
        for begin in range(0, n - 1, stride)
    )
    ppl = math.exp(total_nll / count) if count > 0 else float("inf")

    elapsed = time.time() - t0
    print(f"    Done: {n_windows} windows, {count} tokens, {elapsed:.1f}s")

    return ppl, total_nll, count


# ======================================================================
# Main
# ======================================================================

def run_ptq_eval(model_name="gpt2", device="cpu"):
    """Run PTQ perplexity eval for one model across all quantization formats."""
    from transformers import AutoModelForCausalLM

    # Load test tokens once
    test_tokens = load_wikitext2()
    print(f"Test tokens: {test_tokens.numel():,}")

    quantizers = OrderedDict([
        ("FP32 (baseline)", None),
        ("QF8", qf8_roundtrip),
        ("MXFP8", mxfp8_roundtrip),
        ("INT8", int8_roundtrip),
    ])

    results = []

    for qname, qfn in quantizers.items():
        print(f"\n{'='*60}")
        print(f"  {model_name} — {qname}")
        print(f"{'='*60}")

        # Load fresh model each time
        print(f"  Loading {model_name}...")
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32)
        n_params = sum(p.numel() for p in model.parameters())

        # Apply quantization
        if qfn is not None:
            print(f"  Quantizing weights ({qname})...")
            diag = quantize_model_weights(model, qfn)
            avg_clip = sum(d["clip_high_rate"] for d in diag.values()) / len(diag)
            avg_zero = sum(d["zero_code_rate"] for d in diag.values()) / len(diag)
            print(f"  Avg clip rate: {avg_clip:.4%}, avg zero rate: {avg_zero:.4%}")
        else:
            diag = {}

        # Evaluate
        print(f"  Evaluating perplexity on WikiText-2...")
        ppl, nll, count = evaluate_perplexity(model, test_tokens, device)

        results.append({
            "model": model_name,
            "format": qname,
            "perplexity": round(ppl, 4),
            "nll": round(nll, 4),
            "tokens": count,
            "n_params": n_params,
        })
        print(f"  >>> Perplexity: {ppl:.4f}")

        # Free memory
        del model
        if device == "mps":
            torch.mps.empty_cache()

    return results


def main():
    models = ["gpt2"]
    if len(sys.argv) > 1:
        models = sys.argv[1:]

    # Device selection
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Device: {device}")

    all_results = []
    for model_name in models:
        results = run_ptq_eval(model_name, device)
        all_results.extend(results)

    # Print summary table
    print(f"\n{'='*70}")
    print("SUMMARY: PTQ Perplexity (WikiText-2)")
    print(f"{'='*70}")
    print(f"  {'Model':<15} {'Format':<18} {'Perplexity':>12} {'Δ ppl':>10} {'Δ %':>8}")
    print(f"  {'-'*15} {'-'*18} {'-'*12} {'-'*10} {'-'*8}")

    # Group by model
    models_seen = []
    for r in all_results:
        if r["model"] not in models_seen:
            models_seen.append(r["model"])

    for model_name in models_seen:
        model_results = [r for r in all_results if r["model"] == model_name]
        baseline_ppl = model_results[0]["perplexity"]  # FP32

        for r in model_results:
            delta = r["perplexity"] - baseline_ppl
            pct = (delta / baseline_ppl * 100) if baseline_ppl else 0
            marker = "" if r["format"] == "FP32 (baseline)" else f"{delta:+.4f}"
            pct_str = "" if r["format"] == "FP32 (baseline)" else f"{pct:+.2f}%"
            print(f"  {r['model']:<15} {r['format']:<18} {r['perplexity']:>12.4f} "
                  f"{marker:>10} {pct_str:>8}")
        print()

    # Save results
    import os
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "benchmarks")
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, "ptq_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
