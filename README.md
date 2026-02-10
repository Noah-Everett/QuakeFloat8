# QuakeFloat8 (QF8)

**A log-domain 8-bit number format for ML, inspired by the Quake III fast inverse square root.**

[![Lean](https://img.shields.io/badge/Lean%204-verified-blue)](lean/)
[![Python](https://img.shields.io/badge/Python-3.10+-green)](src/)

## Core Idea

The Quake III fast inverse sqrt works because IEEE 754 float bit patterns accidentally approximate log₂(x) — so integer operations on the bits give you approximate floating-point operations for free.

**QF8 makes the log property *intentional* and *optimized for ML*:**

- **Format**: 1 sign bit + 7-bit u3.4 fixed-point log₂|x|, with E8M0 block scaling (k=32)
- **Multiplication = 7-bit integer addition** (XOR signs, add codes — no mantissa multiplier)
- **+6.6 dB QSNR** over FP8-E4M3 at the same effective bits/element (8.25)
- **O(1) decode** via bit manipulation (Bit-Manipulation Decodable quantizer)

## Results Summary

| Metric | FP8-E4M3 (block) | QF8 | Advantage |
|--------|-------------------|-----|-----------|
| Mean QSNR (GPT-2 weights) | 31.5 dB | 38.1 dB | **+6.6 dB** |
| Matmul QSNR (128×256×128) | 28.5 dB | 35.1 dB | **+6.6 dB** |
| Training val loss (TinyGPT-2) | 2.5478 | 2.5445 | Matches FP32 |

## Lean 4 Verification

During formal verification, we identified errors in theoretical claims from prior literature and derived and verified correct versions. All proofs checked in Lean 4 (v4.27.0):

| Theorem | Status | Notes |
|---------|--------|-------|
| **2.3: Minimax NMSE Optimality** | ✅ Verified | Log-uniform quantization is optimal |
| **3.1: Product Error Decomposition** | ✅ Corrected | Prior work had sign error in last term; we proved correct (minus) version |
| **4.1: Exact MSRE = ε²/12** | ✅ Corrected | Prior claim (ε²/2) was wrong by 6×; we derived and proved exact bound |
| **5.3: Log vs Uniform Separation** | ✅ Corrected | Prior claim used MSE; we proved correct separation under MSRE |
| **5.3(3): Log vs Uniform (MSE)** | ✅ Corrected | Prior claim was false; we proved uniform beats log by 10-50× on MSE for fixed distributions |

See [`lean/RESULTS.md`](lean/RESULTS.md) for full verification details.

## Structure

```
├── src/                    # Python implementation
│   ├── quakefloat8.py      # QF8 encode/decode/matmul
│   ├── benchmark.py        # Benchmark suite
│   └── train_qf8.py        # Training experiments
├── lean/                   # Lean 4 formal verification
│   ├── RESULTS.md          # Verification summary
│   ├── MinimaxNMSE.lean    # Theorem 2.3
│   ├── ProductErrorV2.lean # Theorem 3.1 (corrected)
│   ├── ExactMSRE.lean      # Theorem 4.1
│   └── ...
├── notes/                  # Research notes
├── docs/                   # Documentation & writeups
└── paper/                  # Paper drafts
```

## Key Contributions

1. **Format Design**: Log-uniform quantization with block scaling, optimized for transformer workloads
2. **Implementation**: Full Python implementation with vectorized ops and Kulisch-style accumulation
3. **Formal Verification**: Lean 4 proofs identifying errors in prior theoretical claims
4. **Benchmarks**: Comprehensive evaluation showing +6.6 dB advantage over MXFP8

## Usage

```python
from src.quakefloat8 import QF8Quantizer

# Create quantizer with block size 32
q = QF8Quantizer(block_size=32)

# Quantize weights
weights_qf8 = q.quantize(weights)

# Matrix multiply (uses Kulisch accumulation internally)
output = q.matmul(activations_qf8, weights_qf8)
```

## Related Work

- [Johnson 2018](https://arxiv.org/abs/1811.01721) — "Rethinking Floating Point for Deep Learning"
- [MX Formats](https://arxiv.org/abs/2310.10537) — Microscaling Data Formats for Deep Learning
- [FP8 Formats](https://arxiv.org/abs/2209.05433) — FP8 Formats for Deep Learning

## Status

Active research project. See [`docs/WRITEUP.md`](docs/WRITEUP.md) for full status and roadmap.

## License

MIT License. See [LICENSE](LICENSE).
