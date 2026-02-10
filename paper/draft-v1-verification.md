# QF8 Paper Draft v1 — Verification Report

**Reviewed:** 2026-02-07  
**Document:** `draft-v1.md`  
**Method:** Mathematical verification via Mathematica (wolfram.py), citation checking via arXiv/IEEE Xplore

---

## Executive Summary

| Category | Result |
|----------|--------|
| Core mathematical claims | ✅ **VERIFIED** (4 of 4 key theorems) |
| SQNR calculations | ⚠️ QF8 correct, FP8 methodology unclear |
| Citations | ⚠️ 1 error found (van Baalen venue wrong) |
| Hardware claims | ❓ Cannot verify without synthesis tools |
| Internal consistency | ❌ 1 error found (7/240 vs 7/2880) |

**Action required:** Fix 2 errors before publication.

---

## 1. Mathematical Formula Verification

### 1.1 Minimax MSRE Optimality: ε²/12 ✅ VERIFIED

**Claim (Theorem 2.1):** Log-uniform quantization achieves MSRE* = ε²/12

**Verification:**
```mathematica
exactMSRE[eps_] := 2 - 2*Sqrt[Exp[eps]]*eps/(Exp[eps] - 1)
```

| ε | Exact MSRE | ε²/12 | Ratio |
|---|-----------|-------|-------|
| 0.01 | 8.333×10⁻⁶ | 8.333×10⁻⁶ | 0.99999 |
| 0.05 | 2.083×10⁻⁴ | 2.083×10⁻⁴ | 0.99993 |
| 0.10 | 8.331×10⁻⁴ | 8.333×10⁻⁴ | 0.99971 |
| 0.20 | 3.329×10⁻³ | 3.333×10⁻³ | 0.99883 |

**Result:** Excellent agreement (<0.12% deviation even at ε=0.2). Claim verified.

### 1.2 Exact Per-Cell MSRE Formula ✅ VERIFIED

**Claim (Theorem 4.1):** MSRE_k = 2 - 2√r·ln(r)/(r-1)

**Verification:** Mathematica integration of (1 - √r/u)² over [1, r]:
```mathematica
directMSRE = (1/(r-1)) * Integrate[(1 - Sqrt[r]/u)^2, {u, 1, r}]
```
Result: Simplifies to exactly `2 - 2*Sqrt[r]*Log[r]/(r-1)`

**Difference from claimed formula:** 0 (exact match)

### 1.3 Product Error Decomposition ✅ VERIFIED

**Claim (Theorem 3.1, Corollary):** Under centroid condition:
- NMSE_prod = nX + nY - nX·nY
- Equivalently: 1 - NMSE_prod = (1-nX)(1-nY)

**Verification:**
```mathematica
generalFormula[nX_, nY_, aX_, aY_] := nX + nY + nX*nY + 2*aX*aY - 2*aX*nY - 2*nX*aY
centroidResult = Simplify[generalFormula[nX, nY, nX, nY]]
```
Result: `nX + nY - nX*nY` ✓

Algebraic identity `1 - (1-nX)(1-nY) = nX + nY - nX*nY` confirmed.

**Note:** Paper correctly identifies sign error in prior work (plus should be minus in cross-term).

### 1.4 Taylor Expansion Coefficients ⚠️ INCONSISTENCY FOUND

**Claim (Section 4.1):** MSRE = ε²/12 - 7ε⁴/2880 + O(ε⁶)

**Verification:**
```mathematica
Series[exactMSRE[eps], {eps, 0, 6}]
```
Result: `ε²/12 - 7ε⁴/2880 + 31ε⁶/483840 + O(ε⁷)`

**Section 4.1 is CORRECT.** The coefficient is indeed -7/2880.

**However, Section 6.2 claims:** "Higher-order coefficient 7/240 confirmed"

❌ **ERROR:** 7/240 ≠ 7/2880 (off by factor of 12)

**Required fix:** Change Section 6.2 from "7/240" to "7/2880"

---

## 2. SQNR Calculations

### 2.1 QF8 SQNR ✅ VERIFIED

**Claim:** 38.1 dB for N=256, R=16

**Calculation:**
```mathematica
ε = 16*Log[2]/256 = 0.0433217
NMSE = ε²/12 = 1.564×10⁻⁴
SQNR = 10*Log10[1/NMSE] = 38.06 dB
```

**Result:** 38.06 dB matches claimed 38.1 dB ✓

### 2.2 FP8 E4M3 SQNR ⚠️ DISCREPANCY

**Claim:** 31.5 dB

**Log-uniform model calculation:**
- FP8 E4M3 has 8 levels per octave (3 mantissa bits)
- ε = ln(2)/8 = 0.0866
- NMSE = ε²/12 = 6.26×10⁻⁴
- SQNR = 32.04 dB

**Discrepancy:** Paper claims 31.5 dB, model gives 32.04 dB

**Assessment:** The 0.5 dB difference is plausible because FP8 E4M3 mantissa quantization is not exactly log-uniform. The paper's claim may come from empirical measurement on Gaussian inputs. **Not a clear error, but methodology should be clarified.**

### 2.3 Advantage Calculation

**Claimed:** +6.6 dB (38.1 - 31.5)
**Theoretical:** +6.02 dB using same log-uniform model for both

**Note:** The advantage is real; the exact dB depends on how FP8 SQNR is measured.

---

## 3. Citation Verification

### 3.1 Verified Against arXiv ✅

| # | Citation | arXiv ID | Status |
|---|----------|----------|--------|
| 1 | Cambier 2020 "Shifted and Squeezed 8-bit..." | 2001.05674 | ✅ Correct |
| 3 | Dettmers 2023 "QLoRA" | 2305.14314 | ✅ Correct |
| 7 | Johnson 2018 "Rethinking Floating Point" | 1811.01721 | ✅ Correct |
| 11 | Micikevicius 2022 "FP8 Formats" | 2209.05433 | ✅ Correct |
| 12 | Miyashita 2016 "CNNs using Logarithmic..." | 1603.01025 | ✅ Correct |
| 13 | Noune 2022 "8-bit Numerical Formats" | 2206.02915 | ✅ Correct |
| 14 | Rouhani 2023 "Microscaling Data Formats" | 2310.10537 | ✅ Correct |
| 15 | Rouhani 2023 "Shared Microexponents" | 2302.08007 | ✅ Correct |
| 17 | Sze 2017 "Efficient Processing of DNNs" | 1703.09039 | ✅ Correct |

### 3.2 Verified Against Other Sources ✅

| Citation | Source | Status |
|----------|--------|--------|
| Horowitz 2014 "Computing's Energy Problem" | IEEE ISSCC | ✅ Confirmed (IEEE Xplore 6757323) |

### 3.3 Citation Error Found ❌

**Citation 18:** "van Baalen, M. et al. (2023). FP8 Quantization: The Power of the Exponent. NeurIPS."

**Actual paper:**
- arXiv:2208.09225
- First author: Kuzmin (not van Baalen)
- Not published at NeurIPS 2023 (it's an arXiv preprint from 2022)

**Required fix:** Change to:
> Kuzmin, A., Van Baalen, M., et al. (2022). "FP8 Quantization: The Power of the Exponent." arXiv:2208.09225.

### 3.4 Not Verified (No arXiv, Appear Legitimate)

These citations were not independently verified but appear correct by title/venue:
- Coleman 2008 (IEEE Trans Computers)
- Gustafson & Yonemoto 2017 (Supercomputing Frontiers)
- Kingsbury & Rayner 1971 (Electronics Letters)
- Kulisch 1971 (Numerische Mathematik)
- Swartzlander & Alexopoulos 1975 (IEEE Trans Computers)
- Jouppi 2017 (ISCA)
- Drumond 2018 (NeurIPS)

---

## 4. Hardware Claims

### 4.1 Gate Count Estimates

**Paper claims:**
- QF8 multiply: ~66 gates
- FP8 E4M3 multiply: ~231 gates
- 7-bit CLA adder: ~50 gates
- INT8 8×8 multiplier: ~400 gates

**Textbook estimates (NAND2-equivalent):**
- 7-bit CLA adder: ~35 gates (5N for N-bit CLA)
- 8×8 Wallace tree multiplier: ~96-192 gates

**Assessment:** The paper's gate counts are higher than textbook formulas suggest. This could be due to:
1. Actual synthesis results (more accurate than textbook)
2. NAND2-equivalent vs literal gate count differences
3. Including additional logic (exception handling, rounding)

<!-- UNVERIFIED: Hardware gate counts cannot be verified without synthesis tools/PDK access -->

### 4.2 Energy and Area Claims

Claims of 7.3× area savings vs FP32 and 1.7× vs FP8 are plausible given the adder-vs-multiplier advantage, but cannot be verified without silicon data.

<!-- UNVERIFIED: Requires synthesis or silicon measurement to confirm -->

---

## 5. Internal Consistency Checks

### 5.1 Levels per Octave ✅

- QF8 u3.4 format: 4 fractional bits → 2⁴ = 16 levels/octave ✓
- FP8 E4M3: 3 mantissa bits → 2³ = 8 levels/octave ✓
- Ratio: 2× as claimed ✓

### 5.2 Numerical Values in Tables ✅

Paper's Table (Section 4.2):
| ε | Exact MSRE | ε²/12 | ε²/2 | Exact/(ε²/12) |
|---|---|---|---|---|
| 0.01 | 8.333×10⁻⁶ | 8.333×10⁻⁶ | 5.000×10⁻⁵ | 1.000 |

Verified: These match our Mathematica calculations.

### 5.3 Taylor Coefficient Inconsistency ❌

As noted above, Section 4.1 and Section 6.2 contradict each other:
- Section 4.1: "-7ε⁴/2880" (CORRECT)
- Section 6.2: "7/240 confirmed" (WRONG)

---

## 6. Summary of Required Changes

### Errors (Must Fix)

1. **Section 6.2, Lean Verification:** Change "Higher-order coefficient 7/240 confirmed" to "Higher-order coefficient 7/2880 confirmed"

2. **Reference 18:** Change:
   > van Baalen, M. et al. (2023). "FP8 Quantization: The Power of the Exponent." NeurIPS.
   
   To:
   > Kuzmin, A., Van Baalen, M., Ren, Y., Nagel, M., Peters, J., Blankevoort, T. (2022). "FP8 Quantization: The Power of the Exponent." arXiv:2208.09225.

### Clarifications (Recommended)

1. **Section 2.6 (FP8 SQNR):** Add a note explaining how 31.5 dB was computed (empirical on Gaussian? different model?), since log-uniform model gives 32.04 dB.

2. **Hardware section:** Add <!-- UNVERIFIED: gate counts from synthesis, not independently verified --> or provide synthesis methodology details.

---

## 7. What Was Verified

| Category | Items Checked | Method |
|----------|---------------|--------|
| Core theorems | 4 (Minimax, Per-cell MSRE, Product error, Taylor) | Mathematica |
| SQNR calculations | 2 (QF8, FP8) | Mathematica |
| arXiv citations | 9 | web_fetch |
| IEEE citations | 1 (Horowitz ISSCC) | IEEE Xplore |
| NeurIPS citation | 1 (van Baalen - found error) | arXiv search |
| Internal consistency | 3 checks | Manual + Mathematica |

---

## 8. Items That Cannot Be Verified

1. **Hardware gate counts** — Requires synthesis tools and PDK
2. **Experimental results** (TinyGPT-2 training, SQNR measurements) — Would require re-running experiments
3. **Lean 4 verification claims** — Would require running Lean proofs (the 7/240 error suggests these may have issues)
4. **Non-arXiv historical citations** (Kingsbury 1971, Kulisch 1971, etc.) — Appear legitimate by title/venue

---

*Verification performed by: Lumin (AI assistant)*  
*Tools: wolfram.py (Mathematica Cloud), web_fetch (arXiv, IEEE Xplore)*
