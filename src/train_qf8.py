#!/usr/bin/env python3
"""
train_qf8.py - Compare FP32, FP8-E4M3, and QF8 quantization on a tiny GPT-2.

Trains three identical character-level language models under different
quantization regimes using block-scaled round-trip quantization with
straight-through estimator (STE).  Reports loss curves and quantization penalty.

Usage:
    python3 train_qf8.py
"""

import os
import sys
import math
import time
import urllib.request
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# -- QF8 imports --
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quakefloat8 import (
    BLOCK_SIZE, BIAS, FRAC_BITS, MAX_CODE, MIN_POSITIVE_MAG,
    _CODE_TO_MAG, E8M0_MIN_EXP, E8M0_MAX_EXP,
)

# ======================================================================
# Configuration
# ======================================================================

D_MODEL   = 128
N_HEADS   = 4
N_LAYERS  = 2
D_FF      = 512
VOCAB     = 256          # byte-level characters
SEQ_LEN   = 128
BATCH     = 4
LR        = 3e-4
STEPS     = 500
LOG_EVERY = 25
DEVICE    = "cpu"

DATA_URL = ("https://raw.githubusercontent.com/karpathy/char-rnn/"
            "master/data/tinyshakespeare/input.txt")


# ======================================================================
# 1. FP8-E4M3 Lookup Table
# ======================================================================

def _build_fp8e4m3_lut():
    """Precompute all 128 positive FP8-E4M3 code-to-magnitude values."""
    bias = 7
    lut = np.zeros(128, dtype=np.float64)
    for c in range(128):
        exp = (c >> 3) & 0xF
        man = c & 0x7
        if exp == 15 and man == 7:
            lut[c] = 448.0           # NaN -> clamp to max
        elif exp == 0:
            lut[c] = man * 2.0 ** (1 - bias - 3)   # subnormal
        else:
            lut[c] = (1.0 + man / 8.0) * 2.0 ** (exp - bias)
    return lut


_FP8_MAG      = _build_fp8e4m3_lut()              # 128 entries, monotonic
_FP8_MAX      = 448.0
_FP8_LOG2_MAX = np.log2(_FP8_MAX)                  # ~8.807
_FP8_MID      = (_FP8_MAG[:-1] + _FP8_MAG[1:]) / 2.0  # 127 midpoints


# ======================================================================
# 2. Vectorized NumPy Round-Trip Functions
# ======================================================================

def _qf8_roundtrip_np(x):
    """Block-scaled QF8 round-trip (quantize then dequantize) along last axis.

    Fully vectorized: processes all blocks in a single numpy pass.
    """
    orig_shape = x.shape
    last = orig_shape[-1]
    x2 = x.reshape(-1, last).astype(np.float64)
    rows, cols = x2.shape

    # Pad last axis to multiple of BLOCK_SIZE
    pad = (-cols) % BLOCK_SIZE
    if pad:
        x2 = np.pad(x2, ((0, 0), (0, pad)))

    blocks = x2.reshape(-1, BLOCK_SIZE)              # (total_blocks, 32)

    # -- E8M0 shared scale per block --
    amax = np.max(np.abs(blocks), axis=1)             # (total_blocks,)
    nz = amax > 0
    exps = np.zeros(len(blocks), dtype=np.float64)
    exps[nz] = np.ceil(np.log2(amax[nz]) - (MAX_CODE - BIAS) / 16.0)
    exps = np.clip(exps, E8M0_MIN_EXP, E8M0_MAX_EXP)
    scales = np.exp2(exps)                            # (total_blocks,)

    # -- Encode --
    scaled = blocks / scales[:, None]
    signs = (scaled < 0).astype(np.uint8)
    abs_s = np.abs(scaled)
    log2_v = np.log2(np.maximum(abs_s, 1e-300))
    raw = np.round(BIAS + (1 << FRAC_BITS) * log2_v).astype(np.int32)
    codes = np.where(abs_s >= MIN_POSITIVE_MAG * 0.5,
                     np.clip(raw, 1, MAX_CODE), 0)

    # -- Decode --
    mags = _CODE_TO_MAG[codes] * scales[:, None]
    result = np.where(signs, -mags, mags)

    return result.reshape(rows, -1)[:, :cols].reshape(orig_shape)


def _fp8e4m3_roundtrip_np(x):
    """Block-scaled FP8-E4M3 round-trip (quantize then dequantize) along last axis.

    Uses precomputed LUT + searchsorted for vectorized nearest-code encoding.
    """
    orig_shape = x.shape
    last = orig_shape[-1]
    x2 = x.reshape(-1, last).astype(np.float64)
    rows, cols = x2.shape

    pad = (-cols) % BLOCK_SIZE
    if pad:
        x2 = np.pad(x2, ((0, 0), (0, pad)))

    blocks = x2.reshape(-1, BLOCK_SIZE)

    # -- E8M0 shared scale per block --
    amax = np.max(np.abs(blocks), axis=1)
    nz = amax > 0
    exps = np.zeros(len(blocks), dtype=np.float64)
    exps[nz] = np.ceil(np.log2(amax[nz]) - _FP8_LOG2_MAX)
    exps = np.clip(exps, E8M0_MIN_EXP, E8M0_MAX_EXP)
    scales = np.exp2(exps)

    # -- Encode via searchsorted (round-to-nearest) --
    scaled = blocks / scales[:, None]
    signs = np.sign(scaled)
    abs_s = np.abs(scaled)
    codes = np.searchsorted(_FP8_MID, abs_s.ravel()).reshape(abs_s.shape)
    codes = codes.astype(np.int32)
    codes = np.clip(codes, 0, 126)                   # exclude NaN code

    # -- Decode --
    mags = _FP8_MAG[codes] * scales[:, None]
    result = signs * mags

    return result.reshape(rows, -1)[:, :cols].reshape(orig_shape)


# ======================================================================
# 3. PyTorch STE Autograd Functions
# ======================================================================

class _QF8STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        arr = x.detach().float().contiguous().numpy().astype(np.float64)
        out = _qf8_roundtrip_np(arr)
        return torch.from_numpy(out.astype(np.float32)).to(x.device)

    @staticmethod
    def backward(ctx, g):
        return g   # straight-through


class _FP8STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        arr = x.detach().float().contiguous().numpy().astype(np.float64)
        out = _fp8e4m3_roundtrip_np(arr)
        return torch.from_numpy(out.astype(np.float32)).to(x.device)

    @staticmethod
    def backward(ctx, g):
        return g


def qf8_ste(x):
    return _QF8STE.apply(x)


def fp8_ste(x):
    return _FP8STE.apply(x)


# ======================================================================
# 4. Quantized Linear Layer
# ======================================================================

class QuantizedLinear(nn.Module):
    """nn.Linear with optional round-trip quantization (STE) on weights+activations."""

    def __init__(self, in_features, out_features, quant_fn=None, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.quant_fn = quant_fn

    def forward(self, x):
        if self.quant_fn is not None:
            w = self.quant_fn(self.linear.weight)
            x = self.quant_fn(x)
        else:
            w = self.linear.weight
        return F.linear(x, w, self.linear.bias)


# ======================================================================
# 5. Tiny GPT-2 Model
# ======================================================================

def _init_weights(module):
    """GPT-2 style initialization (normal with std=0.02)."""
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=0.02)


class CausalSelfAttention(nn.Module):
    def __init__(self, qfn=None):
        super().__init__()
        self.nh = N_HEADS
        self.dh = D_MODEL // N_HEADS
        self.qkv  = QuantizedLinear(D_MODEL, 3 * D_MODEL, qfn, bias=False)
        self.proj = QuantizedLinear(D_MODEL, D_MODEL, qfn, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.nh, self.dh)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = [t.transpose(1, 2) for t in (q, k, v)]   # (B, nh, T, dh)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.dh))
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool),
                          diagonal=1)
        att = att.masked_fill(mask, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = (att @ v).transpose(1, 2).reshape(B, T, C)
        return self.proj(y)


class TransformerBlock(nn.Module):
    def __init__(self, qfn=None):
        super().__init__()
        self.ln1  = nn.LayerNorm(D_MODEL)
        self.attn = CausalSelfAttention(qfn)
        self.ln2  = nn.LayerNorm(D_MODEL)
        self.ff1  = QuantizedLinear(D_MODEL, D_FF, qfn)
        self.ff2  = QuantizedLinear(D_FF, D_MODEL, qfn)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff2(F.gelu(self.ff1(self.ln2(x))))
        return x


class TinyGPT(nn.Module):
    def __init__(self, qfn=None):
        super().__init__()
        self.tok_emb = nn.Embedding(VOCAB, D_MODEL)
        self.pos_emb = nn.Embedding(SEQ_LEN, D_MODEL)
        self.blocks  = nn.ModuleList(
            [TransformerBlock(qfn) for _ in range(N_LAYERS)]
        )
        self.ln_f = nn.LayerNorm(D_MODEL)
        self.head = QuantizedLinear(D_MODEL, VOCAB, qfn, bias=False)
        self.apply(_init_weights)

    def forward(self, idx):
        B, T = idx.shape
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device))
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln_f(x))


# ======================================================================
# 6. Data Loading
# ======================================================================

def load_data():
    """Download tiny Shakespeare (or fall back to synthetic)."""
    cache = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "shakespeare.txt")
    if os.path.exists(cache):
        with open(cache) as f:
            text = f.read()
    else:
        try:
            print("Downloading tiny Shakespeare...")
            resp = urllib.request.urlopen(DATA_URL, timeout=30)
            text = resp.read().decode("utf-8")
            with open(cache, "w") as f:
                f.write(text)
            print(f"  Downloaded {len(text):,} characters")
        except Exception as e:
            print(f"  Download failed ({e}); generating synthetic data")
            rng = np.random.default_rng(42)
            # Structured synthetic: repeating printable ASCII blocks
            base = "".join(chr(c) for c in range(32, 127)) * 100
            text = base * 25

    data = np.frombuffer(text.encode("utf-8"), dtype=np.uint8).copy()
    n = int(0.9 * len(data))
    print(f"Data: {n:,} train / {len(data) - n:,} val characters")
    return data[:n], data[n:]


def get_batch(data):
    """Sample a random batch of (input, target) sequences."""
    ix = np.random.randint(0, len(data) - SEQ_LEN - 1, size=BATCH)
    x = torch.tensor(
        np.stack([data[i : i + SEQ_LEN] for i in ix]), dtype=torch.long
    )
    y = torch.tensor(
        np.stack([data[i + 1 : i + SEQ_LEN + 1] for i in ix]), dtype=torch.long
    )
    return x, y


# ======================================================================
# 7. Training Loop
# ======================================================================

def train_model(name, quant_fn, train_data, val_data):
    """Train a TinyGPT model and return metrics dict."""
    print(f"\n{'=' * 60}")
    print(f"  Training: {name}")
    print(f"{'=' * 60}")

    torch.manual_seed(42)
    np.random.seed(42)

    model = TinyGPT(qfn=quant_fn)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS)

    train_losses = []
    val_snapshots = []
    step_times = []

    for step in range(1, STEPS + 1):
        t0 = time.time()
        model.train()
        x, y = get_batch(train_data)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, VOCAB), y.view(-1))

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        dt = time.time() - t0
        train_losses.append(loss.item())
        step_times.append(dt)

        if step % LOG_EVERY == 0 or step == 1:
            model.eval()
            with torch.no_grad():
                vx, vy = get_batch(val_data)
                vloss = F.cross_entropy(
                    model(vx).view(-1, VOCAB), vy.view(-1)
                ).item()
            val_snapshots.append((step, vloss))
            ms = np.mean(step_times[-LOG_EVERY:]) * 1000
            print(f"  step {step:4d} | train {loss.item():.4f} | "
                  f"val {vloss:.4f} | {ms:.0f} ms/step")

    final_train = float(np.mean(train_losses[-50:]))
    final_val = val_snapshots[-1][1]
    avg_ms = float(np.mean(step_times)) * 1000
    print(f"  -- Final: train={final_train:.4f}  val={final_val:.4f}  "
          f"avg={avg_ms:.0f} ms/step")

    return dict(
        name=name,
        train_losses=train_losses,
        val_snapshots=val_snapshots,
        step_times=step_times,
        final_train=final_train,
        final_val=final_val,
        avg_ms=avg_ms,
        n_params=n_params,
    )


# ======================================================================
# 8. Save Results
# ======================================================================

def save_results(results, path):
    """Write comparison report as Markdown."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fp32 = results[0]

    lines = []
    lines.append("# QF8 Training Experiment Results\n")
    lines.append(f"**Date**: {time.strftime('%Y-%m-%d %H:%M UTC')}\n")

    lines.append("## Configuration\n")
    lines.append(f"- Model: TinyGPT-2 (d_model={D_MODEL}, heads={N_HEADS}, "
                 f"layers={N_LAYERS}, d_ff={D_FF})")
    lines.append(f"- Vocab: {VOCAB} (byte-level), seq_len={SEQ_LEN}")
    lines.append(f"- Training: {STEPS} steps, batch={BATCH}, lr={LR}, cosine decay")
    lines.append(f"- Quantization: block-scaled round-trip + STE, "
                 f"block_size={BLOCK_SIZE}")
    lines.append(f"- Device: {DEVICE}\n")

    # Summary table
    lines.append("## Summary\n")
    lines.append("| Model | Final Train | Final Val | Avg ms/step |")
    lines.append("|-------|------------|----------|-------------|")
    for r in results:
        lines.append(f"| {r['name']} | {r['final_train']:.4f} | "
                     f"{r['final_val']:.4f} | {r['avg_ms']:.0f} |")

    # Quantization penalty
    lines.append("\n## Quantization Penalty (vs FP32)\n")
    for r in results[1:]:
        delta = r["final_val"] - fp32["final_val"]
        pct = (delta / fp32["final_val"] * 100) if fp32["final_val"] else 0
        lines.append(f"- **{r['name']}**: delta_val = {delta:+.4f} ({pct:+.1f}%)")

    # Training loss curve (sampled)
    lines.append("\n## Training Loss (sampled every 25 steps)\n")
    hdr = "| Step | " + " | ".join(r["name"] for r in results) + " |"
    sep = "|------|" + "|".join("--------" for _ in results) + "|"
    lines.append(hdr)
    lines.append(sep)
    for s in range(0, STEPS, LOG_EVERY):
        row = f"| {s + 1:4d} "
        for r in results:
            if s < len(r["train_losses"]):
                row += f"| {r['train_losses'][s]:.4f} "
            else:
                row += "|   --   "
        row += "|"
        lines.append(row)

    # Validation snapshots
    lines.append("\n## Validation Loss Snapshots\n")
    lines.append(hdr)
    lines.append(sep)
    all_steps = sorted({s for r in results for s, _ in r["val_snapshots"]})
    for step in all_steps:
        row = f"| {step:4d} "
        for r in results:
            val = dict(r["val_snapshots"]).get(step)
            row += f"| {val:.4f} " if val is not None else "|   --   "
        row += "|"
        lines.append(row)

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nResults saved to {path}")


# ======================================================================
# Main
# ======================================================================

def _sanity_check():
    """Quick roundtrip sanity check before training."""
    rng = np.random.RandomState(123)
    x = rng.randn(8, 128).astype(np.float64) * 0.02

    for name, fn in [("QF8", _qf8_roundtrip_np), ("FP8", _fp8e4m3_roundtrip_np)]:
        out = fn(x)
        assert out.shape == x.shape, f"{name}: shape mismatch"
        sig = np.sum(x ** 2)
        noise = np.sum((x - out) ** 2)
        qsnr = 10 * np.log10(sig / noise) if noise > 0 else float("inf")
        print(f"  {name} roundtrip QSNR = {qsnr:.1f} dB")
        assert qsnr > 15, f"{name} QSNR too low: {qsnr:.1f} dB"


def main():
    t_start = time.time()
    print("=" * 60)
    print("  QF8 Training Experiment")
    print("=" * 60)
    print(f"PyTorch {torch.__version__} | NumPy {np.__version__} | {DEVICE}\n")

    print("Sanity check...")
    _sanity_check()
    print()

    train_data, val_data = load_data()

    configs = [
        ("FP32",     None),
        ("FP8-E4M3", fp8_ste),
        ("QF8",      qf8_ste),
    ]

    results = []
    for name, qfn in configs:
        r = train_model(name, qfn, train_data, val_data)
        results.append(r)

    # Save report
    notes_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "notes"
    )
    save_results(results, os.path.join(notes_dir, "training-results.md"))

    # Final summary
    elapsed = time.time() - t_start
    print(f"\n{'=' * 60}")
    print("  FINAL COMPARISON")
    print(f"{'=' * 60}")
    print(f"  {'Model':<12} {'Train':>8} {'Val':>8} {'ms/step':>10}")
    print(f"  {'---':<12} {'---':>8} {'---':>8} {'---':>10}")
    for r in results:
        print(f"  {r['name']:<12} {r['final_train']:8.4f} "
              f"{r['final_val']:8.4f} {r['avg_ms']:10.0f}")
    print(f"\n  Total time: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
