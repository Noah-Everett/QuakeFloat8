/-
  QuakeFloat8 — Theorem 5.3(3) Counterexample Verification

  CLAIM (Paper): Log-uniform quantization beats uniform for matmul error.
  COUNTER-CLAIM (Skeptic): For the paper's own parameters, uniform wins on MSE.

  We compare MSE and NMSE of uniform vs log-uniform quantization numerically.
-/

def showFloat (f : Float) : String := toString f
def floatMax (a b : Float) : Float := if a > b then a else b
def floatMin (a b : Float) : Float := if a < b then a else b

-- Uniform quantizer: nearest level from {a, a+Δ, ..., b}
def uniformQuantize (a b : Float) (N : Nat) (x : Float) : Float :=
  let delta := (b - a) / Float.ofNat (N - 1)
  let idx := ((x - a) / delta).round
  let idx' := floatMax 0.0 (floatMin idx (Float.ofNat (N - 1)))
  a + idx' * delta

-- Log-uniform quantizer: nearest level from {a, a·r, a·r², ...}
def logQuantize (a b : Float) (N : Nat) (x : Float) : Float :=
  let logA := a.log
  let logR := (b / a).log / Float.ofNat (N - 1)
  let logX := x.log
  let idx := ((logX - logA) / logR).round
  let idx' := floatMax 0.0 (floatMin idx (Float.ofNat (N - 1)))
  (logA + idx' * logR).exp

-- Generate uniform samples in [a, b]
def uniformSamples (a b : Float) (n : Nat) : List Float :=
  (List.range n).map fun i =>
    let t := Float.ofNat i / Float.ofNat (n - 1)
    a + t * (b - a)

-- Generate top-heavy samples (quadratic: more mass near b)
def topHeavySamples (a b : Float) (n : Nat) : List Float :=
  (List.range n).map fun i =>
    let t := Float.ofNat i / Float.ofNat (n - 1)
    a + t * t * (b - a)

-- Compute MSE for both quantizers on a list of samples
def computeMSE (a b : Float) (N : Nat) (samples : List Float) : Float × Float :=
  let n := Float.ofNat samples.length
  let result := samples.foldl (fun (acc : Float × Float) x =>
    let xU := uniformQuantize a b N x
    let xL := logQuantize a b N x
    let errU := (x - xU) * (x - xU)
    let errL := (x - xL) * (x - xL)
    (acc.1 + errU, acc.2 + errL)) (0.0, 0.0)
  (result.1 / n, result.2 / n)

-- Compute NMSE for both quantizers
def computeNMSE (a b : Float) (N : Nat) (samples : List Float) : Float × Float :=
  let result := samples.foldl (fun (acc : Float × Float × Float) x =>
    let xU := uniformQuantize a b N x
    let xL := logQuantize a b N x
    let errU := (x - xU) * (x - xU)
    let errL := (x - xL) * (x - xL)
    (acc.1 + errU, acc.2.1 + errL, acc.2.2 + x * x)) (0.0, 0.0, 0.0)
  (result.1 / result.2.2, result.2.1 / result.2.2)

-- ============================================================
-- Main counterexample verification
-- ============================================================

#eval do
  IO.println "============================================"
  IO.println "  Theorem 5.3(3) Counter-Verification"
  IO.println "============================================"
  IO.println ""

  -- Test 1: Uniform dist, small range, 4-bit
  IO.println "--- Test 1: Uniform [0.1, 100], N=16 (4-bit) ---"
  let s1 := uniformSamples 0.1 100.0 1000
  let (mU1, mL1) := computeMSE 0.1 100.0 16 s1
  IO.println s!"  Uniform MSE: {showFloat mU1}"
  IO.println s!"  Log MSE:     {showFloat mL1}"
  IO.println s!"  Ratio (log/uniform): {showFloat (mL1/mU1)}"
  IO.println s!"  Winner: {if mU1 < mL1 then "UNIFORM ← log loses!" else "LOG"}"
  IO.println ""

  -- Test 2: Uniform dist, wide range, 4-bit
  IO.println "--- Test 2: Uniform [0.01, 100], N=16 (4-bit) ---"
  let s2 := uniformSamples 0.01 100.0 1000
  let (mU2, mL2) := computeMSE 0.01 100.0 16 s2
  IO.println s!"  Uniform MSE: {showFloat mU2}"
  IO.println s!"  Log MSE:     {showFloat mL2}"
  IO.println s!"  Ratio (log/uniform): {showFloat (mL2/mU2)}"
  IO.println s!"  Winner: {if mU2 < mL2 then "UNIFORM ← log loses!" else "LOG"}"
  IO.println ""

  -- Test 3: Top-heavy dist (neural net like), 4-bit
  IO.println "--- Test 3: Top-heavy [0.1, 100], N=16 (4-bit) ---"
  let s3 := topHeavySamples 0.1 100.0 1000
  let (mU3, mL3) := computeMSE 0.1 100.0 16 s3
  IO.println s!"  Uniform MSE: {showFloat mU3}"
  IO.println s!"  Log MSE:     {showFloat mL3}"
  IO.println s!"  Ratio (log/uniform): {showFloat (mL3/mU3)}"
  IO.println s!"  Winner: {if mU3 < mL3 then "UNIFORM ← log loses!" else "LOG"}"
  IO.println ""

  -- Test 4: E4M3-like, 8-bit, uniform dist
  IO.println "--- Test 4: E4M3-like [0.001953, 448], N=256 (8-bit) ---"
  let s4 := uniformSamples 0.001953125 448.0 5000
  let (mU4, mL4) := computeMSE 0.001953125 448.0 256 s4
  IO.println s!"  Uniform MSE: {showFloat mU4}"
  IO.println s!"  Log MSE:     {showFloat mL4}"
  IO.println s!"  Ratio (log/uniform): {showFloat (mL4/mU4)}"
  IO.println s!"  Winner: {if mU4 < mL4 then "UNIFORM ← log loses!" else "LOG"}"
  IO.println ""

  -- Test 5: E4M3-like, 8-bit, top-heavy
  IO.println "--- Test 5: Top-heavy E4M3-like [0.001953, 448], N=256 ---"
  let s5 := topHeavySamples 0.001953125 448.0 5000
  let (mU5, mL5) := computeMSE 0.001953125 448.0 256 s5
  IO.println s!"  Uniform MSE: {showFloat mU5}"
  IO.println s!"  Log MSE:     {showFloat mL5}"
  IO.println s!"  Ratio (log/uniform): {showFloat (mL5/mU5)}"
  IO.println s!"  Winner: {if mU5 < mL5 then "UNIFORM ← log loses!" else "LOG"}"

-- ============================================================
-- NMSE comparison (this is what the paper's theorem is about)
-- ============================================================

#eval do
  IO.println ""
  IO.println "============================================"
  IO.println "  NMSE Comparison (normalized by E[x²])"
  IO.println "============================================"
  IO.println ""

  IO.println "--- NMSE: Uniform [0.1, 100], N=16 ---"
  let s1 := uniformSamples 0.1 100.0 1000
  let (nU1, nL1) := computeNMSE 0.1 100.0 16 s1
  IO.println s!"  Uniform NMSE: {showFloat nU1}"
  IO.println s!"  Log NMSE:     {showFloat nL1}"
  IO.println s!"  Winner: {if nU1 < nL1 then "UNIFORM" else "LOG"}"
  IO.println ""

  IO.println "--- NMSE: Uniform [0.001953, 448], N=256 (E4M3) ---"
  let s2 := uniformSamples 0.001953125 448.0 5000
  let (nU2, nL2) := computeNMSE 0.001953125 448.0 256 s2
  IO.println s!"  Uniform NMSE: {showFloat nU2}"
  IO.println s!"  Log NMSE:     {showFloat nL2}"
  IO.println s!"  Winner: {if nU2 < nL2 then "UNIFORM" else "LOG"}"
  IO.println ""

  IO.println "--- NMSE: Top-heavy [0.001953, 448], N=256 (E4M3) ---"
  let s3 := topHeavySamples 0.001953125 448.0 5000
  let (nU3, nL3) := computeNMSE 0.001953125 448.0 256 s3
  IO.println s!"  Uniform NMSE: {showFloat nU3}"
  IO.println s!"  Log NMSE:     {showFloat nL3}"
  IO.println s!"  Winner: {if nU3 < nL3 then "UNIFORM" else "LOG"}"

-- ============================================================
-- Key insight about when each quantizer wins
-- ============================================================

/-
  ANALYSIS:

  1. Log-uniform quantization minimizes worst-case RELATIVE error (NMSE).
     This means: for any input distribution, the relative error is bounded.
     Small values get fine-grained quantization, large values get coarse.

  2. Uniform quantization minimizes worst-case ABSOLUTE error.
     Every value gets the same step size Δ = (b-a)/(N-1).

  3. For MSE (absolute squared error), uniform wins when:
     - The distribution has significant mass at large values
     - The dynamic range b/a is moderate
     - Because log-quantization wastes many levels on tiny values near a

  4. For NMSE (relative squared error), log wins in the worst case by construction.
     But for a FIXED distribution, uniform can still win on NMSE if the
     distribution is concentrated.

  5. Theorem 5.3(3) overreaches by claiming log always beats uniform under
     the paper's specific parameters. The minimax guarantee (Theorem 2.3)
     is about WORST-CASE over ALL distributions, not a specific one.
-/
