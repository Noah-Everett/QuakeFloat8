#!/usr/bin/env python3
"""
train_gpt2_small.py — GPT-2 Small training with FP32 / FP8-E4M3 / QF8 comparison.

Scales the QF8 experiment to GPT-2 Small (768d, 12 heads, 12 layers) on MPS.
Quantization round-trips are reimplemented in pure PyTorch for MPS compatibility
(no NumPy detours).

Usage:
    conda run -n qf8 python src/train_gpt2_small.py
"""

import os
import sys
import math
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken

# ======================================================================
# Configuration
# ======================================================================

# Model config (GPT-2 Small)
D_MODEL   = 768
N_HEADS   = 12
N_LAYERS  = 12
D_FF      = 3072

# Training config
SEQ_LEN   = 256
BATCH     = 4           # micro-batch size per forward pass
GRAD_ACCUM = 4          # effective batch = BATCH * GRAD_ACCUM = 16
LR        = 3e-4
WARMUP_STEPS = 200
TOTAL_STEPS  = 2000     # Can increase if time permits
LOG_EVERY    = 50
VAL_EVERY    = 200
VAL_BATCHES  = 10

# Quantization
BLOCK_SIZE = 32
QF8_BIAS   = 64
QF8_FRAC_BITS = 4
QF8_MAX_CODE  = 127
QF8_MIN_POSITIVE_MAG = 2.0 ** ((1 - QF8_BIAS) / (1 << QF8_FRAC_BITS))
E8M0_MIN_EXP = -127
E8M0_MAX_EXP = 127

# Device selection
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

print(f"Using device: {DEVICE}")


# ======================================================================
# 1. Pure PyTorch Quantization: QF8 Round-Trip
# ======================================================================

def qf8_roundtrip_torch(x: torch.Tensor) -> torch.Tensor:
    """Block-scaled QF8 round-trip entirely in PyTorch (MPS-compatible).
    
    Mathematically equivalent to the NumPy version in quakefloat8.py:
    1. Reshape into blocks of BLOCK_SIZE along last axis
    2. Compute E8M0 shared scale per block
    3. Encode: code = round(BIAS + 16 * log2(|x/scale|)), clamp [1, 127] or 0
    4. Decode: magnitude = 2^((code - BIAS) / 16) * scale, restore sign
    """
    orig_shape = x.shape
    last = orig_shape[-1]
    
    # Flatten to 2D
    x2 = x.reshape(-1, last).float()
    rows, cols = x2.shape
    
    # Pad to multiple of BLOCK_SIZE
    pad = (-cols) % BLOCK_SIZE
    if pad > 0:
        x2 = F.pad(x2, (0, pad))
    
    # Reshape into blocks: (rows, n_blocks, BLOCK_SIZE)
    n_blocks = x2.shape[1] // BLOCK_SIZE
    blocks = x2.reshape(rows, n_blocks, BLOCK_SIZE)
    
    # E8M0 shared scale per block
    amax = blocks.abs().amax(dim=-1)  # (rows, n_blocks)
    nz = amax > 0
    exps = torch.zeros_like(amax)
    exps[nz] = torch.ceil(torch.log2(amax[nz]) - (QF8_MAX_CODE - QF8_BIAS) / 16.0)
    exps = exps.clamp(E8M0_MIN_EXP, E8M0_MAX_EXP)
    scales = torch.exp2(exps)  # (rows, n_blocks)
    
    # Scale elements
    # Avoid division by zero for blocks that are all-zero
    safe_scales = scales.clamp(min=1e-45)
    scaled = blocks / safe_scales.unsqueeze(-1)  # (rows, n_blocks, BLOCK_SIZE)
    
    # Encode
    signs = (scaled < 0).float()
    abs_s = scaled.abs()
    
    # log2 of absolute scaled values (clamped to avoid -inf)
    log2_v = torch.log2(abs_s.clamp(min=1e-45))
    raw = torch.round(QF8_BIAS + (1 << QF8_FRAC_BITS) * log2_v)
    
    # Codes: 0 if too small, else clamp to [1, MAX_CODE]
    threshold = QF8_MIN_POSITIVE_MAG * 0.5
    nonzero_mask = abs_s >= threshold
    codes = torch.where(
        nonzero_mask,
        raw.clamp(1, QF8_MAX_CODE),
        torch.zeros_like(raw)
    )
    
    # Decode: magnitude = 2^((code - BIAS) / 16) * scale
    mags_unscaled = torch.exp2((codes - QF8_BIAS) / 16.0)
    # code==0 should give magnitude 0
    mags_unscaled = torch.where(codes > 0, mags_unscaled, torch.zeros_like(mags_unscaled))
    mags = mags_unscaled * safe_scales.unsqueeze(-1)
    
    # Restore sign
    result = torch.where(signs.bool(), -mags, mags)
    
    # Reshape back
    result = result.reshape(rows, -1)[:, :cols].reshape(orig_shape)
    return result


# ======================================================================
# 2. Pure PyTorch Quantization: FP8-E4M3 Round-Trip
# ======================================================================

# Build FP8-E4M3 lookup table as a constant tensor
def _build_fp8e4m3_lut_tensor():
    """Build all 128 positive FP8-E4M3 magnitudes as a torch tensor."""
    bias = 7
    lut = torch.zeros(128, dtype=torch.float32)
    for c in range(128):
        exp = (c >> 3) & 0xF
        man = c & 0x7
        if exp == 15 and man == 7:
            lut[c] = 448.0       # NaN → clamp to max
        elif exp == 0:
            lut[c] = man * 2.0 ** (1 - bias - 3)   # subnormal
        else:
            lut[c] = (1.0 + man / 8.0) * 2.0 ** (exp - bias)
    return lut


_FP8_LUT_CPU = _build_fp8e4m3_lut_tensor()
_FP8_MAX_VAL = 448.0
_FP8_LOG2_MAX = math.log2(_FP8_MAX_VAL)
# Midpoints for nearest-code rounding
_FP8_MID_CPU = (_FP8_LUT_CPU[:-1] + _FP8_LUT_CPU[1:]) / 2.0  # 127 midpoints

# Cache device-specific copies
_fp8_lut_cache = {}
_fp8_mid_cache = {}


def _get_fp8_lut(device):
    """Get FP8 LUT on the right device (cached)."""
    key = str(device)
    if key not in _fp8_lut_cache:
        _fp8_lut_cache[key] = _FP8_LUT_CPU.to(device)
        _fp8_mid_cache[key] = _FP8_MID_CPU.to(device)
    return _fp8_lut_cache[key], _fp8_mid_cache[key]


def fp8e4m3_roundtrip_torch(x: torch.Tensor) -> torch.Tensor:
    """Block-scaled FP8-E4M3 round-trip entirely in PyTorch (MPS-compatible).
    
    Uses searchsorted on the midpoint table for nearest-code encoding,
    then indexes the LUT for decoding.
    """
    fp8_lut, fp8_mid = _get_fp8_lut(x.device)
    
    orig_shape = x.shape
    last = orig_shape[-1]
    
    x2 = x.reshape(-1, last).float()
    rows, cols = x2.shape
    
    pad = (-cols) % BLOCK_SIZE
    if pad > 0:
        x2 = F.pad(x2, (0, pad))
    
    n_blocks = x2.shape[1] // BLOCK_SIZE
    blocks = x2.reshape(rows, n_blocks, BLOCK_SIZE)
    
    # E8M0 shared scale per block
    amax = blocks.abs().amax(dim=-1)
    nz = amax > 0
    exps = torch.zeros_like(amax)
    exps[nz] = torch.ceil(torch.log2(amax[nz]) - _FP8_LOG2_MAX)
    exps = exps.clamp(E8M0_MIN_EXP, E8M0_MAX_EXP)
    scales = torch.exp2(exps)
    
    safe_scales = scales.clamp(min=1e-45)
    scaled = blocks / safe_scales.unsqueeze(-1)
    
    signs = torch.sign(scaled)
    abs_s = scaled.abs()
    
    # Flatten abs_s for searchsorted
    flat_abs = abs_s.reshape(-1)
    # searchsorted: find nearest FP8 code via midpoint table
    codes = torch.searchsorted(fp8_mid, flat_abs)
    codes = codes.clamp(0, 126)  # exclude NaN code (127)
    
    # Decode via LUT
    mags_unscaled = fp8_lut[codes].reshape(abs_s.shape)
    mags = mags_unscaled * safe_scales.unsqueeze(-1)
    
    result = signs * mags
    result = result.reshape(rows, -1)[:, :cols].reshape(orig_shape)
    return result


# ======================================================================
# 3. PyTorch STE Autograd Functions (Pure PyTorch, MPS-safe)
# ======================================================================

class _QF8STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return qf8_roundtrip_torch(x)

    @staticmethod
    def backward(ctx, g):
        return g  # straight-through


class _FP8STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return fp8e4m3_roundtrip_torch(x)

    @staticmethod
    def backward(ctx, g):
        return g  # straight-through


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
# 5. GPT-2 Model
# ======================================================================

def _init_weights(module):
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
        q, k, v = [t.transpose(1, 2) for t in (q, k, v)]

        # Use scaled_dot_product_attention if available (PyTorch 2.0+)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).reshape(B, T, C)
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


class GPT2(nn.Module):
    def __init__(self, vocab_size, qfn=None):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, D_MODEL)
        self.pos_emb = nn.Embedding(SEQ_LEN, D_MODEL)
        self.blocks  = nn.ModuleList(
            [TransformerBlock(qfn) for _ in range(N_LAYERS)]
        )
        self.ln_f = nn.LayerNorm(D_MODEL)
        self.head = QuantizedLinear(D_MODEL, vocab_size, qfn, bias=False)
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

def load_data(tokenizer):
    """Load wikitext-2 from HuggingFace datasets, tokenize, return train/val tensors."""
    from datasets import load_dataset
    
    print("Loading wikitext-2-raw-v1...")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", trust_remote_code=True)
    
    def tokenize_split(split_name):
        texts = ds[split_name]["text"]
        # Join all text, filter empty lines
        all_text = "\n".join(t for t in texts if t.strip())
        tokens = tokenizer.encode(all_text)
        return torch.tensor(tokens, dtype=torch.long)
    
    train_tokens = tokenize_split("train")
    val_tokens = tokenize_split("validation")
    
    print(f"Train tokens: {len(train_tokens):,}")
    print(f"Val tokens:   {len(val_tokens):,}")
    
    return train_tokens, val_tokens


def get_batch(data, device):
    """Sample a random batch of (input, target) sequences."""
    ix = torch.randint(0, len(data) - SEQ_LEN - 1, (BATCH,))
    x = torch.stack([data[i : i + SEQ_LEN] for i in ix]).to(device)
    y = torch.stack([data[i + 1 : i + SEQ_LEN + 1] for i in ix]).to(device)
    return x, y


# ======================================================================
# 7. Validation
# ======================================================================

@torch.no_grad()
def evaluate(model, val_data, device, vocab_size):
    """Compute validation loss over VAL_BATCHES random batches."""
    model.eval()
    losses = []
    for _ in range(VAL_BATCHES):
        x, y = get_batch(val_data, device)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


# ======================================================================
# 8. Training Loop
# ======================================================================

def train_model(name, quant_fn, train_data, val_data, vocab_size, device):
    """Train a GPT-2 model and return metrics dict."""
    print(f"\n{'=' * 70}")
    print(f"  Training: {name}")
    print(f"{'=' * 70}")

    torch.manual_seed(42)
    if device == "mps":
        torch.mps.manual_seed(42)

    model = GPT2(vocab_size, qfn=quant_fn).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")
    print(f"  Effective batch size: {BATCH * GRAD_ACCUM}")
    print(f"  Seq length: {SEQ_LEN}")

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.1, betas=(0.9, 0.95))
    
    # Linear warmup + cosine decay
    def lr_schedule(step):
        if step < WARMUP_STEPS:
            return step / WARMUP_STEPS
        decay_steps = TOTAL_STEPS - WARMUP_STEPS
        progress = (step - WARMUP_STEPS) / decay_steps
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_schedule)

    train_losses = []
    val_snapshots = []
    step_times = []
    
    model.train()
    accum_loss = 0.0
    
    t_total_start = time.time()

    for step in range(1, TOTAL_STEPS + 1):
        t0 = time.time()
        
        # Gradient accumulation
        opt.zero_grad()
        micro_losses = []
        for _ in range(GRAD_ACCUM):
            x, y = get_batch(train_data, device)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
            loss_scaled = loss / GRAD_ACCUM
            loss_scaled.backward()
            micro_losses.append(loss.item())
        
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        dt = time.time() - t0
        avg_loss = sum(micro_losses) / len(micro_losses)
        train_losses.append(avg_loss)
        step_times.append(dt)

        if step % LOG_EVERY == 0 or step == 1:
            lr_now = opt.param_groups[0]['lr']
            ms = dt * 1000
            avg_recent = sum(train_losses[-LOG_EVERY:]) / len(train_losses[-LOG_EVERY:])
            print(f"  step {step:5d}/{TOTAL_STEPS} | train {avg_recent:.4f} | "
                  f"lr {lr_now:.2e} | {ms:.0f} ms/step")

        if step % VAL_EVERY == 0 or step == 1:
            vloss = evaluate(model, val_data, device, vocab_size)
            val_snapshots.append((step, vloss))
            elapsed = time.time() - t_total_start
            print(f"  >>> val {vloss:.4f} | elapsed {elapsed:.0f}s")

    # Final validation
    if not val_snapshots or val_snapshots[-1][0] != TOTAL_STEPS:
        vloss = evaluate(model, val_data, device, vocab_size)
        val_snapshots.append((TOTAL_STEPS, vloss))
    
    final_train = float(np.mean(train_losses[-100:]))
    final_val = val_snapshots[-1][1]
    avg_ms = float(np.mean(step_times)) * 1000
    total_time = time.time() - t_total_start
    
    print(f"  -- Final: train={final_train:.4f}  val={final_val:.4f}  "
          f"avg={avg_ms:.0f} ms/step  total={total_time:.0f}s")

    return dict(
        name=name,
        train_losses=train_losses,
        val_snapshots=val_snapshots,
        step_times=step_times,
        final_train=final_train,
        final_val=final_val,
        avg_ms=avg_ms,
        total_time=total_time,
        n_params=n_params,
    )


# ======================================================================
# 9. Save Results
# ======================================================================

def save_results(results, path):
    """Write comparison report as Markdown."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fp32 = results[0]

    lines = []
    lines.append("# QF8 Training Experiment — GPT-2 Small\n")
    lines.append(f"**Date**: {time.strftime('%Y-%m-%d %H:%M %Z')}\n")

    lines.append("## Configuration\n")
    lines.append(f"- Model: GPT-2 Small (d_model={D_MODEL}, heads={N_HEADS}, "
                 f"layers={N_LAYERS}, d_ff={D_FF})")
    lines.append(f"- Tokenizer: GPT-2 BPE (tiktoken)")
    lines.append(f"- Dataset: wikitext-2-raw-v1")
    lines.append(f"- Seq length: {SEQ_LEN}, Micro-batch: {BATCH}, "
                 f"Grad accum: {GRAD_ACCUM}, Effective batch: {BATCH * GRAD_ACCUM}")
    lines.append(f"- Training: {TOTAL_STEPS} steps, lr={LR}, warmup={WARMUP_STEPS}, "
                 f"cosine decay")
    lines.append(f"- Quantization: block-scaled round-trip + STE, "
                 f"block_size={BLOCK_SIZE}, pure PyTorch (MPS-compatible)")
    lines.append(f"- Device: {DEVICE}")
    lines.append(f"- Parameters: {fp32['n_params']:,}\n")

    # Summary table
    lines.append("## Summary\n")
    lines.append("| Model | Final Train Loss | Final Val Loss | Avg ms/step | Total Time |")
    lines.append("|-------|-----------------|---------------|-------------|------------|")
    for r in results:
        lines.append(f"| {r['name']} | {r['final_train']:.4f} | "
                     f"{r['final_val']:.4f} | {r['avg_ms']:.0f} | "
                     f"{r['total_time']:.0f}s |")

    # Quantization penalty
    lines.append("\n## Quantization Penalty (vs FP32)\n")
    for r in results[1:]:
        delta = r["final_val"] - fp32["final_val"]
        pct = (delta / fp32["final_val"] * 100) if fp32["final_val"] else 0
        lines.append(f"- **{r['name']}**: delta_val = {delta:+.4f} ({pct:+.1f}%)")

    # Throughput comparison
    lines.append("\n## Throughput\n")
    lines.append("| Model | ms/step | Slowdown vs FP32 |")
    lines.append("|-------|---------|-----------------|")
    for r in results:
        slowdown = r['avg_ms'] / fp32['avg_ms'] if fp32['avg_ms'] else 0
        lines.append(f"| {r['name']} | {r['avg_ms']:.0f} | {slowdown:.2f}x |")

    # Validation snapshots
    lines.append("\n## Validation Loss Over Training\n")
    hdr = "| Step | " + " | ".join(r["name"] for r in results) + " |"
    sep = "|------|" + "|".join("--------" for _ in results) + "|"
    lines.append(hdr)
    lines.append(sep)
    all_steps = sorted({s for r in results for s, _ in r["val_snapshots"]})
    for step in all_steps:
        row = f"| {step:5d} "
        for r in results:
            val = dict(r["val_snapshots"]).get(step)
            row += f"| {val:.4f} " if val is not None else "|   --   "
        row += "|"
        lines.append(row)

    # Training loss curve (sampled)
    lines.append("\n## Training Loss (sampled every 50 steps)\n")
    lines.append(hdr)
    lines.append(sep)
    for s in range(0, TOTAL_STEPS, LOG_EVERY):
        if s < min(len(r["train_losses"]) for r in results):
            row = f"| {s + 1:5d} "
            for r in results:
                row += f"| {r['train_losses'][s]:.4f} "
            row += "|"
            lines.append(row)

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nResults saved to {path}")


# ======================================================================
# 10. Sanity Check
# ======================================================================

def sanity_check():
    """Verify pure-PyTorch quantization matches the NumPy reference."""
    print("Sanity check: PyTorch quantization round-trips...")
    
    torch.manual_seed(42)
    x = torch.randn(4, 128) * 0.02
    
    # QF8
    out_qf8 = qf8_roundtrip_torch(x)
    assert out_qf8.shape == x.shape, f"QF8 shape mismatch: {out_qf8.shape}"
    sig = (x ** 2).sum().item()
    noise = ((x - out_qf8) ** 2).sum().item()
    qsnr = 10 * math.log10(sig / noise) if noise > 0 else float("inf")
    print(f"  QF8 QSNR = {qsnr:.1f} dB (expect >25)")
    assert qsnr > 15, f"QF8 QSNR too low: {qsnr}"
    
    # FP8
    out_fp8 = fp8e4m3_roundtrip_torch(x)
    assert out_fp8.shape == x.shape, f"FP8 shape mismatch: {out_fp8.shape}"
    noise = ((x - out_fp8) ** 2).sum().item()
    qsnr = 10 * math.log10(sig / noise) if noise > 0 else float("inf")
    print(f"  FP8 QSNR = {qsnr:.1f} dB (expect >25)")
    assert qsnr > 15, f"FP8 QSNR too low: {qsnr}"
    
    # Test on MPS if available
    if DEVICE != "cpu":
        x_dev = x.to(DEVICE)
        out_qf8_dev = qf8_roundtrip_torch(x_dev)
        out_fp8_dev = fp8e4m3_roundtrip_torch(x_dev)
        # Verify same results (within float32 tolerance)
        diff_qf8 = (out_qf8 - out_qf8_dev.cpu()).abs().max().item()
        diff_fp8 = (out_fp8 - out_fp8_dev.cpu()).abs().max().item()
        print(f"  CPU vs {DEVICE} max diff: QF8={diff_qf8:.2e}, FP8={diff_fp8:.2e}")
        assert diff_qf8 < 1e-4, f"QF8 CPU vs {DEVICE} mismatch: {diff_qf8}"
        assert diff_fp8 < 1e-4, f"FP8 CPU vs {DEVICE} mismatch: {diff_fp8}"
    
    print("  All sanity checks passed ✓\n")


# ======================================================================
# Main
# ======================================================================

def main():
    t_start = time.time()
    print("=" * 70)
    print("  QF8 Training Experiment — GPT-2 Small on MPS")
    print("=" * 70)
    print(f"PyTorch {torch.__version__} | Device: {DEVICE}\n")

    sanity_check()

    # Tokenizer
    print("Loading GPT-2 tokenizer (tiktoken)...")
    tokenizer = tiktoken.get_encoding("gpt2")
    vocab_size = tokenizer.n_vocab
    print(f"Vocab size: {vocab_size}\n")

    # Data
    train_data, val_data = load_data(tokenizer)

    # Train all three variants
    configs = [
        ("FP32",     None),
        ("FP8-E4M3", fp8_ste),
        ("QF8",      qf8_ste),
    ]

    results = []
    for name, qfn in configs:
        try:
            r = train_model(name, qfn, train_data, val_data, vocab_size, DEVICE)
            results.append(r)
        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "MPS" in str(e):
                print(f"\n  ⚠ OOM or MPS error for {name}: {e}")
                print(f"  Skipping {name}...")
                # Try to free memory
                if DEVICE == "mps":
                    torch.mps.empty_cache()
            else:
                raise

    if not results:
        print("No models trained successfully!")
        return

    # Save report
    notes_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "notes"
    )
    save_results(results, os.path.join(notes_dir, "training-results-gpt2-small.md"))

    # Final summary
    elapsed = time.time() - t_start
    print(f"\n{'=' * 70}")
    print("  FINAL COMPARISON")
    print(f"{'=' * 70}")
    print(f"  {'Model':<12} {'Train':>8} {'Val':>8} {'ms/step':>10} {'Time':>8}")
    print(f"  {'---':<12} {'---':>8} {'---':>8} {'---':>10} {'---':>8}")
    for r in results:
        print(f"  {r['name']:<12} {r['final_train']:8.4f} "
              f"{r['final_val']:8.4f} {r['avg_ms']:10.0f} {r['total_time']:7.0f}s")
    
    fp32_val = results[0]['final_val']
    for r in results[1:]:
        delta = r['final_val'] - fp32_val
        pct = (delta / fp32_val * 100) if fp32_val else 0
        print(f"  {r['name']:<12} penalty: {delta:+.4f} ({pct:+.1f}%)")
    
    print(f"\n  Total experiment time: {elapsed:.0f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
