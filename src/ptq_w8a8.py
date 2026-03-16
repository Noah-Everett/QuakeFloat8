#!/usr/bin/env python3
"""
ptq_w8a8.py — W8A8 Post-Training Quantization perplexity eval.

Quantizes both weights AND activations (inputs to all linear layers) to
QF8 / MXFP8 / INT8, then measures perplexity on WikiText-2.

Activations are quantized dynamically (block scale computed at inference time),
which is the standard approach for MX-style block-scaled formats.

Usage:
    python src/ptq_w8a8.py                                    # GPT-2 Small only
    python src/ptq_w8a8.py gpt2 gpt2-medium gpt2-large       # All three
"""

import math
import time
import json
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from sqnr_benchmark import qf8_roundtrip, mxfp8_roundtrip, int8_roundtrip


# ======================================================================
# W8A8 Quantization: Weights + Activation Hooks
# ======================================================================

def quantize_model_weights(model, quant_fn):
    """Apply quantization round-trip to all linear layer weights in-place."""
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


def install_activation_hooks(model, quant_fn):
    """Install forward pre-hooks that quantize inputs to all linear layers.

    Only quantizes inputs to nn.Linear modules (the GEMM path).
    LayerNorm, softmax, residual adds, and embeddings are untouched.
    """
    hooks = []

    def make_hook(name):
        def hook_fn(module, args):
            x = args[0]
            # Quantize activation on CPU/MPS — move to float32 for round-trip
            x_q, _ = quant_fn(x.float())
            return (x_q.to(x.dtype),) + args[1:]
        return hook_fn

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Skip embedding-related linears (lm_head often shares weights with wte)
            if "embed" in name.lower() or "wte" in name.lower() or "wpe" in name.lower():
                continue
            h = module.register_forward_pre_hook(make_hook(name))
            hooks.append(h)

    return hooks


def remove_hooks(hooks):
    for h in hooks:
        h.remove()


# ======================================================================
# Perplexity Evaluation (same as ptq_perplexity.py)
# ======================================================================

def load_wikitext2():
    """Load WikiText-2 test set, return as a single token tensor."""
    from datasets import load_dataset
    from transformers import AutoTokenizer

    print("Loading WikiText-2 test set...")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n".join(t for t in ds["text"] if t.strip())
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokens = tokenizer.encode(text)
    return torch.tensor(tokens, dtype=torch.long)


@torch.no_grad()
def evaluate_perplexity(model, tokens, device, stride=512, max_length=1024):
    """Evaluate perplexity using sliding window."""
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

        if begin > 0:
            target_ids[:, :-stride] = -100

        outputs = model(input_ids, labels=target_ids)
        nlls.append(outputs.loss.item() * (target_ids != -100).sum().item())
        n_windows += 1

        if n_windows % 50 == 0:
            elapsed = time.time() - t0
            print(f"    {n_windows} windows, {elapsed:.1f}s elapsed...")

    total_nll = sum(nlls)
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

def run_w8a8_eval(model_name="gpt2", device="cpu"):
    """Run W8A8 PTQ perplexity eval for one model."""
    from transformers import AutoModelForCausalLM

    test_tokens = load_wikitext2()
    print(f"Test tokens: {test_tokens.numel():,}")

    quantizers = OrderedDict([
        ("FP32 (baseline)", None),
        ("W-only QF8", ("w_only", qf8_roundtrip)),
        ("W-only MXFP8", ("w_only", mxfp8_roundtrip)),
        ("W-only INT8", ("w_only", int8_roundtrip)),
        ("W8A8 QF8", ("w8a8", qf8_roundtrip)),
        ("W8A8 MXFP8", ("w8a8", mxfp8_roundtrip)),
        ("W8A8 INT8", ("w8a8", int8_roundtrip)),
    ])

    results = []

    for qname, qspec in quantizers.items():
        print(f"\n{'='*60}")
        print(f"  {model_name} — {qname}")
        print(f"{'='*60}")

        print(f"  Loading {model_name}...")
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32)
        n_params = sum(p.numel() for p in model.parameters())

        hooks = []
        if qspec is not None:
            mode, qfn = qspec

            # Always quantize weights (except for FP32 baseline)
            print(f"  Quantizing weights...")
            quantize_model_weights(model, qfn)

            # For W8A8, also hook activations
            if mode == "w8a8":
                print(f"  Installing activation quantization hooks...")
                hooks = install_activation_hooks(model, qfn)
                n_hooks = len(hooks)
                print(f"  {n_hooks} linear layers hooked for activation quantization")

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

        remove_hooks(hooks)
        del model
        if device == "mps":
            torch.mps.empty_cache()

    return results


def main():
    models = ["gpt2"]
    if len(sys.argv) > 1:
        models = sys.argv[1:]

    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Device: {device}")

    all_results = []
    for model_name in models:
        results = run_w8a8_eval(model_name, device)
        all_results.extend(results)

    # Summary table
    print(f"\n{'='*70}")
    print("SUMMARY: W8A8 PTQ Perplexity (WikiText-2)")
    print(f"{'='*70}")
    print(f"  {'Model':<15} {'Format':<18} {'Perplexity':>12} {'Δ ppl':>10} {'Δ %':>8}")
    print(f"  {'-'*15} {'-'*18} {'-'*12} {'-'*10} {'-'*8}")

    models_seen = []
    for r in all_results:
        if r["model"] not in models_seen:
            models_seen.append(r["model"])

    for model_name in models_seen:
        model_results = [r for r in all_results if r["model"] == model_name]
        baseline_ppl = model_results[0]["perplexity"]

        for r in model_results:
            delta = r["perplexity"] - baseline_ppl
            pct = (delta / baseline_ppl * 100) if baseline_ppl else 0
            marker = "" if r["format"] == "FP32 (baseline)" else f"{delta:+.4f}"
            pct_str = "" if r["format"] == "FP32 (baseline)" else f"{pct:+.2f}%"
            print(f"  {r['model']:<15} {r['format']:<18} {r['perplexity']:>12.4f} "
                  f"{marker:>10} {pct_str:>8}")
        print()

    # Save results
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, "w8a8_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
