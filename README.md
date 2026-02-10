# QuakeFloat8 (QF8)

**A log-domain 8-bit number format that makes ML multiplication as cheap as addition.**

[![Lean 4 Verified](https://img.shields.io/badge/Lean%204-formally%20verified-blue)](lean/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-green)](src/)

## The Idea

The legendary [Quake III fast inverse square root](https://en.wikipedia.org/wiki/Fast_inverse_square_root) works because IEEE float bit patterns secretly approximate log₂(x). Integer ops on float bits = approximate log-domain arithmetic.

**QF8 makes this intentional.** Instead of an accidental approximation, we design an 8-bit format where:

- **Multiply = integer addition** (no multiplier circuit needed)
- **2× more precision** than FP8 (16 levels per octave vs 8)
- **7.3× smaller multipliers** than FP32
- **Formally verified** optimality guarantees in Lean 4

## Why It Matters

Modern ML is bottlenecked by matrix multiplication. Multipliers are expensive — they dominate chip area and power. QF8 replaces multiplication with addition in the log domain, potentially enabling cheaper, faster ML inference hardware.

## Results

| Metric | FP8 E4M3 | QF8 | Improvement |
|--------|----------|-----|-------------|
| Signal-to-noise (weights) | 31.5 dB | 38.1 dB | **+6.6 dB** |
| Signal-to-noise (matmul) | 28.5 dB | 35.1 dB | **+6.6 dB** |
| Training loss (GPT-2) | 2.5478 | 2.5445 | Matches FP32 |

The +6.6 dB advantage means **4.5× less quantization noise** at the same bit width.

## Formal Verification

Key mathematical properties are proven in [Lean 4](https://lean-lang.org/), not just tested:

| Property | Status |
|----------|--------|
| Log-uniform quantization is minimax optimal | ✅ Proven |
| Product error decomposition | ✅ Proven |
| Exact mean squared relative error bound | ✅ Proven |
| Separation between log and uniform quantizers | ✅ Proven |

See [`lean/RESULTS.md`](lean/RESULTS.md) for proof details.

## Quick Start

```python
from src.quakefloat8 import QF8Quantizer

q = QF8Quantizer(block_size=32)
weights_qf8 = q.quantize(weights)
output = q.matmul(activations, weights_qf8)
```

## Format Specification

```
Per block (32 elements):
  Shared scale: 8-bit E8M0 (power of 2)

Per element (8 bits):
  [sign][log₂|x| in u3.4 fixed-point]
  
  Multiply: XOR signs, add 7-bit codes
  Value: (-1)^sign × scale × 2^((code-64)/16)
```

## Project Structure

```
├── src/           # Python implementation
├── lean/          # Lean 4 formal proofs  
└── paper/         # LaTeX paper + PDF
```

## Paper

The full technical paper with proofs and hardware analysis is in [`paper/quakefloat8.pdf`](paper/quakefloat8.pdf).

## Related Work

- [Johnson 2018](https://arxiv.org/abs/1811.01721) — Rethinking Floating Point for Deep Learning
- [MX Formats](https://arxiv.org/abs/2310.10537) — Microscaling Data Formats  
- [FP8 Formats](https://arxiv.org/abs/2209.05433) — FP8 Formats for Deep Learning

## License

MIT
