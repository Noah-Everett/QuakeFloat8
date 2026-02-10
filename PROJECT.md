# Quake Float: Custom Number Representations for ML

**Type:** Research (idea stage — raw)
**Collaborators:** Noah Everett (solo)
**Status:** Raw idea, needs literature review

## Core Idea

Inspired by the Quake 3 fast inverse square root: the hack works because IEEE 754 float bit patterns approximate log₂(x), so integer operations on the bits give you approximate floating-point operations essentially for free.

**Question:** Can you design a number format where the bit-level structure is specifically optimized for operations that ML uses most (matmul, exp, softmax, normalization), even if it's worse at operations ML rarely needs?

## Why This Is Interesting
- The IEEE 754 format was designed for general-purpose computing, not ML
- ML workloads use a very specific subset of operations overwhelmingly
- Google already proved this matters with bfloat16 (sacrifice mantissa precision for range → better for ML)
- But bfloat16 is still fundamentally IEEE-shaped. What if you started from scratch?

## Related Work to Investigate
- **Posit number system** (John Gustafson) — tapered precision, claimed advantages over IEEE floats
- **bfloat16** — Google's truncated float for TPUs
- **FP8 formats** (E4M3, E5M2) — NVIDIA/ARM/Intel standard for inference
- **Logarithmic number systems** — represent numbers as fixed-point logarithms
- **Block floating point** — shared exponent across groups of values
- **Quantization literature** — INT8, INT4, binary neural networks
- **Fast inverse sqrt** itself — the bit-level trick and its mathematical basis
- **Custom ALU design** literature — what's the hardware cost of new formats?

## Key Questions
1. What operations does a typical ML training loop actually spend time on? (Profile it)
2. What properties of number representation make those operations fast?
3. Is there a formal optimization problem: "find the encoding that minimizes total compute for this workload"?
4. How much precision can you sacrifice in rarely-used operations before training breaks?
5. What's the hardware story? (A format is useless if no chip implements it)

## Potential Angles
- **Theoretical**: formalize the optimization problem, prove bounds on how much speedup is possible
- **Empirical**: implement a software emulator for custom formats, benchmark on real models
- **Systems**: design a format and simulate its hardware cost

## Notes
- Noah hasn't investigated this much yet
- This sits at the intersection of CS systems, numerical analysis, and ML
- Could be very practical/impactful if a real improvement is found
- The "genius function" angle (Quake-style bit tricks) is the hook, but the real substance is number representation theory
