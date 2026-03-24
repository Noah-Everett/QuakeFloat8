#!/usr/bin/env python3
"""
train_gpt2_small.py — GPT-2 Small training with FP32 / FP8-E4M3 / QF8 comparison.

Scales the QF8 experiment to GPT-2 Small (768d, 12 heads, 12 layers).
Quantization round-trips are reimplemented in pure PyTorch for MPS compatibility
(no NumPy detours).

Usage:
    # Local quick test (WikiText-2, 2K steps)
    python src/train_gpt2_small.py --dataset wikitext2 --steps 2000

    # Full run (OpenWebText, 20K steps — needs ~128GB RAM)
    python src/train_gpt2_small.py --dataset openwebtext --steps 20000
"""

import argparse
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
# CLI
# ======================================================================

def parse_args():
    p = argparse.ArgumentParser(description="GPT-2 Small QAT: FP32 vs FP8 vs QF8")
    p.add_argument("--dataset", choices=["wikitext2", "openwebtext"], default="openwebtext",
                   help="Training dataset (default: openwebtext)")
    p.add_argument("--steps", type=int, default=20000, help="Total training steps (default: 20000)")
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    p.add_argument("--batch", type=int, default=4, help="Micro-batch size (default: 4)")
    p.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps (default: 4)")
    p.add_argument("--seq-len", type=int, default=256, help="Sequence length (default: 256)")
    return p.parse_args()

# ======================================================================
# Configuration
# ======================================================================

# Model config (GPT-2 Small)
D_MODEL   = 768
N_HEADS   = 12
N_LAYERS  = 12
D_FF      = 3072

# Training config (defaults, overridden by CLI)
SEQ_LEN   = 256        # also used for positional embedding size
LR        = 3e-4
LOG_EVERY    = 50
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

_FP8_BIAS = 7
_FP8_MAX_VAL = 448.0
_FP8_LOG2_MAX = math.log2(_FP8_MAX_VAL)
_FP8_MIN_SUBNORMAL = 2.0 ** (1 - _FP8_BIAS - 3)  # smallest positive: 2^(-9) = 1/512

# Build FP8-E4M3 decode LUT (128 entries, indexed by 7-bit code)
def _build_fp8e4m3_lut():
    lut = torch.zeros(128, dtype=torch.float32)
    for c in range(128):
        exp = (c >> 3) & 0xF
        man = c & 0x7
        if exp == 15 and man == 7:
            lut[c] = _FP8_MAX_VAL  # NaN → clamp to max
        elif exp == 0:
            lut[c] = man * 2.0 ** (1 - _FP8_BIAS - 3)
        else:
            lut[c] = (1.0 + man / 8.0) * 2.0 ** (exp - _FP8_BIAS)
    return lut

_FP8_LUT = _build_fp8e4m3_lut()
_fp8_lut_cache = {}


def fp8e4m3_roundtrip_torch(x: torch.Tensor) -> torch.Tensor:
    """Block-scaled FP8-E4M3 round-trip entirely in PyTorch (MPS-compatible).

    Encodes via arithmetic (floor/round on log2), decodes via small LUT index.
    No searchsorted needed.
    """
    # Cache LUT on device
    key = str(x.device)
    if key not in _fp8_lut_cache:
        _fp8_lut_cache[key] = _FP8_LUT.to(x.device)
    fp8_lut = _fp8_lut_cache[key]

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
    block_exps = torch.zeros_like(amax)
    block_exps[nz] = torch.ceil(torch.log2(amax[nz]) - _FP8_LOG2_MAX)
    block_exps = block_exps.clamp(E8M0_MIN_EXP, E8M0_MAX_EXP)
    scales = torch.exp2(block_exps)

    safe_scales = scales.clamp(min=1e-45)
    scaled = blocks / safe_scales.unsqueeze(-1)

    signs = scaled.sign()
    abs_s = scaled.abs().clamp(max=_FP8_MAX_VAL)

    # --- Encode to 7-bit code = (biased_exp << 3) | mantissa ---
    log2_abs = torch.log2(abs_s.clamp(min=_FP8_MIN_SUBNORMAL * 0.5))
    biased_exp = (torch.floor(log2_abs) + _FP8_BIAS).clamp(0, 15).long()

    # Normal: man = round((abs_s / 2^(biased_exp - bias) - 1) * 8)
    # Subnormal (biased_exp=0): man = round(abs_s / 2^(1-bias) * 8)
    is_subnorm = biased_exp < 1

    exp_unbiased = (biased_exp - _FP8_BIAS).float()
    significand = abs_s / torch.exp2(exp_unbiased) - 1.0
    man = torch.round(significand * 8.0).long()

    # Handle mantissa overflow: man=8 means bump exponent, man=0
    overflow = man >= 8
    man = torch.where(overflow, torch.zeros_like(man), man)
    biased_exp = torch.where(overflow, biased_exp + 1, biased_exp)

    # Subnormal path
    man_sub = torch.round(abs_s * (2.0 ** (_FP8_BIAS + 3 - 1))).clamp(0, 7).long()
    man = torch.where(is_subnorm, man_sub, man)
    biased_exp = torch.where(is_subnorm, torch.zeros_like(biased_exp), biased_exp)

    # Clamp to valid range: max code = 126 (exp=15, man=6 = 448), code 127 is NaN
    code = (biased_exp << 3) | man.clamp(0, 7)
    code = code.clamp(0, 126)

    # Zero out tiny inputs
    code = torch.where(abs_s < _FP8_MIN_SUBNORMAL * 0.5,
                       torch.zeros_like(code), code)

    # --- Decode via LUT ---
    mags_unscaled = fp8_lut[code.long()]
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

def load_data(tokenizer, dataset_name):
    """Load dataset, tokenize, return train/val tensors."""
    from datasets import load_dataset

    if dataset_name == "wikitext2":
        print("Loading wikitext-2-raw-v1...")
        ds = load_dataset("wikitext", "wikitext-2-raw-v1")

        def tokenize_split(split_name):
            texts = ds[split_name]["text"]
            all_text = "\n".join(t for t in texts if t.strip())
            tokens = tokenizer.encode(all_text)
            return torch.tensor(tokens, dtype=torch.long)

        train_tokens = tokenize_split("train")
        val_tokens = tokenize_split("validation")

    elif dataset_name == "openwebtext":
        print("Loading OpenWebText (this may take a while on first run)...")

        # Check for cached tokenized tensors
        cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                 ".cache")
        train_cache = os.path.join(cache_dir, "owt_train.bin")
        val_cache = os.path.join(cache_dir, "owt_val.bin")

        if os.path.exists(train_cache) and os.path.exists(val_cache):
            print("  Loading cached tokenized tensors...")
            train_tokens = torch.from_numpy(
                np.memmap(train_cache, dtype=np.uint16, mode="r")).long()
            val_tokens = torch.from_numpy(
                np.memmap(val_cache, dtype=np.uint16, mode="r")).long()
        else:
            ds = load_dataset("openwebtext", split="train")
            n_total = len(ds)
            n_val = min(5000, n_total // 100)  # ~1% for val, capped at 5K docs
            n_train = n_total - n_val

            print(f"  Total docs: {n_total:,} | Train: {n_train:,} | Val: {n_val:,}")

            def tokenize_docs(docs, label=""):
                # Use numpy arrays to avoid Python list memory blow-up
                chunks = []
                n = len(docs)
                t0 = time.time()
                for i, doc in enumerate(docs):
                    text = doc["text"]
                    if text.strip():
                        chunks.append(np.array(tokenizer.encode(text), dtype=np.uint16))
                    if (i + 1) % 100000 == 0 or (i + 1) == n:
                        elapsed = time.time() - t0
                        rate = (i + 1) / elapsed
                        print(f"    {label} {i+1:,}/{n:,} docs "
                              f"({rate:.0f} docs/s, {elapsed:.0f}s)")
                return np.concatenate(chunks)

            print("  Tokenizing train split...")
            train_np = tokenize_docs(ds.select(range(n_train)), "train")
            print("  Tokenizing val split...")
            val_np = tokenize_docs(ds.select(range(n_train, n_total)), "val")

            # Cache to disk for future runs
            os.makedirs(cache_dir, exist_ok=True)
            print(f"  Caching to {cache_dir}...")
            fp = np.memmap(train_cache, dtype=np.uint16, mode="w+",
                           shape=train_np.shape)
            fp[:] = train_np
            fp.flush()
            fp = np.memmap(val_cache, dtype=np.uint16, mode="w+",
                           shape=val_np.shape)
            fp[:] = val_np
            fp.flush()

            # Convert uint16 -> int64 via torch (avoids numpy double-alloc)
            train_tokens = torch.from_numpy(train_np).long()
            del train_np
            val_tokens = torch.from_numpy(val_np).long()
            del val_np

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    print(f"Train tokens: {len(train_tokens):,}")
    print(f"Val tokens:   {len(val_tokens):,}")

    return train_tokens, val_tokens


def get_batch(data, device, seq_len, batch_size):
    """Sample a random batch of (input, target) sequences."""
    ix = torch.randint(0, len(data) - seq_len - 1, (batch_size,))
    x = torch.stack([data[i : i + seq_len] for i in ix]).to(device)
    y = torch.stack([data[i + 1 : i + seq_len + 1] for i in ix]).to(device)
    return x, y


# ======================================================================
# 7. Validation
# ======================================================================

@torch.no_grad()
def evaluate(model, val_data, device, vocab_size, seq_len, batch_size):
    """Compute validation loss over VAL_BATCHES random batches."""
    model.eval()
    losses = []
    for _ in range(VAL_BATCHES):
        x, y = get_batch(val_data, device, seq_len, batch_size)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


# ======================================================================
# 8. Training Loop
# ======================================================================

def train_model(name, quant_fn, train_data, val_data, vocab_size, device, args):
    """Train a GPT-2 model and return metrics dict."""
    total_steps = args.steps
    batch_size = args.batch
    grad_accum = args.grad_accum
    seq_len = args.seq_len
    warmup_steps = min(200, total_steps // 10)
    val_every = max(200, total_steps // 50)  # ~50 val checkpoints

    print(f"\n{'=' * 70}")
    print(f"  Training: {name}")
    print(f"{'=' * 70}")

    torch.manual_seed(args.seed)
    if device == "mps":
        torch.mps.manual_seed(args.seed)
    elif device == "cuda":
        torch.cuda.manual_seed(args.seed)

    model = GPT2(vocab_size, qfn=quant_fn).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")
    print(f"  Effective batch size: {batch_size * grad_accum}")
    print(f"  Seq length: {seq_len}")

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.1, betas=(0.9, 0.95))

    # Linear warmup + cosine decay
    def lr_schedule(step):
        if step < warmup_steps:
            return step / warmup_steps
        decay_steps = total_steps - warmup_steps
        progress = (step - warmup_steps) / decay_steps
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_schedule)

    train_losses = []
    val_snapshots = []
    step_times = []

    model.train()

    t_total_start = time.time()

    for step in range(1, total_steps + 1):
        t0 = time.time()

        # Gradient accumulation
        opt.zero_grad()
        micro_losses = []
        for _ in range(grad_accum):
            x, y = get_batch(train_data, device, seq_len, batch_size)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
            loss_scaled = loss / grad_accum
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
            print(f"  step {step:5d}/{total_steps} | train {avg_recent:.4f} | "
                  f"lr {lr_now:.2e} | {ms:.0f} ms/step")

        if step % val_every == 0 or step == 1:
            vloss = evaluate(model, val_data, device, vocab_size, seq_len, batch_size)
            val_snapshots.append((step, vloss))
            elapsed = time.time() - t_total_start
            print(f"  >>> val {vloss:.4f} | elapsed {elapsed:.0f}s")

    # Final validation
    if not val_snapshots or val_snapshots[-1][0] != total_steps:
        vloss = evaluate(model, val_data, device, vocab_size, seq_len, batch_size)
        val_snapshots.append((total_steps, vloss))

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

def save_results(results, path, args):
    """Write raw results as JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    out = {
        "config": {
            "model": f"GPT-2 Small (d_model={D_MODEL}, heads={N_HEADS}, layers={N_LAYERS})",
            "dataset": args.dataset,
            "steps": args.steps,
            "seq_len": args.seq_len,
            "batch": args.batch,
            "grad_accum": args.grad_accum,
            "effective_batch": args.batch * args.grad_accum,
            "lr": LR,
            "warmup": min(200, args.steps // 10),
            "block_size": BLOCK_SIZE,
            "seed": args.seed,
            "device": DEVICE,
            "n_params": results[0]["n_params"],
            "date": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        },
        "runs": [
            {
                "name": r["name"],
                "final_train": r["final_train"],
                "final_val": r["final_val"],
                "avg_ms_per_step": r["avg_ms"],
                "total_time_s": r["total_time"],
                "val_snapshots": r["val_snapshots"],
                "train_losses": r["train_losses"],
                "step_times": r["step_times"],
            }
            for r in results
        ],
    }

    with open(path, "w") as f:
        json.dump(out, f, indent=2)
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
    args = parse_args()

    # Update global so GPT2 positional embeddings match
    global SEQ_LEN
    SEQ_LEN = args.seq_len

    t_start = time.time()
    print("=" * 70)
    print(f"  QF8 Training Experiment — GPT-2 Small")
    print(f"  Dataset: {args.dataset} | Steps: {args.steps} | Seed: {args.seed}")
    print("=" * 70)
    print(f"PyTorch {torch.__version__} | Device: {DEVICE}\n")

    sanity_check()

    # Tokenizer
    print("Loading GPT-2 tokenizer (tiktoken)...")
    tokenizer = tiktoken.get_encoding("gpt2")
    vocab_size = tokenizer.n_vocab
    print(f"Vocab size: {vocab_size}\n")

    # Data
    train_data, val_data = load_data(tokenizer, args.dataset)

    # Train all three variants
    configs = [
        ("FP32",     None),
        ("FP8-E4M3", fp8_ste),
        ("QF8",      qf8_ste),
    ]

    results = []
    for name, qfn in configs:
        try:
            r = train_model(name, qfn, train_data, val_data, vocab_size, DEVICE, args)
            results.append(r)
        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "MPS" in str(e):
                print(f"\n  OOM or device error for {name}: {e}")
                print(f"  Skipping {name}...")
                if DEVICE == "mps":
                    torch.mps.empty_cache()
                elif DEVICE == "cuda":
                    torch.cuda.empty_cache()
            else:
                raise

    if not results:
        print("No models trained successfully!")
        return

    # Save report
    results_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "training"
    )
    save_results(results, os.path.join(results_dir, "gpt2_small_qat.json"), args)

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
