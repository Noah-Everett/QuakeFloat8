/-
  QuakeFloat8 — Numerical Corrections Verification
  Formalization in core Lean 4 (no Mathlib)

  Corrections:
  1. E4M3 SQNR = 31.5 dB (not 25.1 dB)
  2. Bit allocation = (9.75, 14.07, 0.18) for (weights, activations, gradients)
  3. Quadratic approximation max error = 0.32% (not 0.1%)
-/

-- ============================================================
-- Part 1: E4M3 SQNR Calculation
-- ============================================================

-- E4M3 format: 1 sign, 4 exponent, 3 mantissa bits
-- Bias = 7, so representable values: (-1)^s · 2^(e-7) · (1 + m/8)
-- where e ∈ {1,...,15} (normal), m ∈ {0,...,7}
-- Special: e=0 → subnormal: (-1)^s · 2^(-6) · (m/8)
-- e=15, m=7 → NaN; all others are finite

-- Generate all positive E4M3 representable values
def e4m3Values : List Float := Id.run do
  let mut vals : List Float := []
  -- Subnormals: e=0, m=1..7 → 2^(-6) · m/8 = m · 2^(-9)
  for m in List.range 8 do
    if m > 0 then
      vals := vals ++ [Float.ofNat m * Float.pow 2.0 (-9.0)]
  -- Normals: e=1..14, m=0..7 → 2^(e-7) · (1 + m/8)
  for e in List.range 15 do
    if e > 0 then
      for m in List.range 8 do
        let exp := Float.ofNat e - 7.0
        let mant := 1.0 + Float.ofNat m / 8.0
        vals := vals ++ [Float.pow 2.0 exp * mant]
  -- e=15, m=0..6 (m=7 is NaN)
  for m in List.range 7 do
    let mant := 1.0 + Float.ofNat m / 8.0
    vals := vals ++ [Float.pow 2.0 8.0 * mant]
  return vals

-- E4M3 quantizer: find nearest representable value
def e4m3Quantize (vals : List Float) (x : Float) : Float := Id.run do
  let ax := Float.abs x
  let mut best : Float := 0.0
  let mut bestDist : Float := 1.0e30
  for v in vals do
    let d := Float.abs (ax - v)
    if d < bestDist then
      bestDist := d
      best := v
  return if x < 0.0 then -best else best

-- Compute SQNR for Gaussian N(0,1)
-- Use quasi-Monte Carlo with inverse normal CDF

def invNormalCDF (p : Float) : Float :=
  if p <= 0.0 || p >= 1.0 then 0.0
  else
    let t := if p < 0.5 then p else 1.0 - p
    let u := Float.sqrt (-2.0 * Float.log t)
    let c0 := 2.515517; let c1 := 0.802853; let c2 := 0.010328
    let d1 := 1.432788; let d2 := 0.189269; let d3 := 0.001308
    let result := u - (c0 + c1 * u + c2 * u * u) / (1.0 + d1 * u + d2 * u * u + d3 * u * u * u)
    if p < 0.5 then -result else result

#eval! do
  IO.println "============================================"
  IO.println "  E4M3 SQNR on N(0,1)"
  IO.println "============================================"

  let vals := e4m3Values
  IO.println s!"  Number of positive E4M3 values: {vals.length}"
  IO.println s!"  Min: {vals.head!}"
  IO.println s!"  Max: {vals.getLast!}"

  -- Generate Gaussian samples
  let nSamples := 20000
  let mut mseSum : Float := 0.0
  let mut sigPow : Float := 0.0
  let mut count : Nat := 0

  for i in List.range nSamples do
    let u := (Float.ofNat i + 0.5) / Float.ofNat nSamples
    let x := invNormalCDF u  -- N(0,1) sample
    let qx := e4m3Quantize vals x
    mseSum := mseSum + (x - qx) * (x - qx)
    sigPow := sigPow + x * x
    count := count + 1

  let mse := mseSum / Float.ofNat count
  let sigPower := sigPow / Float.ofNat count
  let nmse := mse / sigPower
  let sqnr := -10.0 * Float.log nmse / Float.log 10.0

  IO.println s!"  MSE = {mse}"
  IO.println s!"  Signal power = {sigPower}"
  IO.println s!"  NMSE = {nmse}"
  IO.println s!"  SQNR = {sqnr} dB"
  IO.println ""
  IO.println s!"  Expected: ~31.5 dB (corrected), not 25.1 dB (old)"
  IO.println s!"  Match: {if sqnr > 30.0 && sqnr < 33.0 then "✓" else "✗"}"

-- Per-binade breakdown
#eval! do
  IO.println ""
  IO.println "============================================"
  IO.println "  E4M3 per-binade MSE breakdown"
  IO.println "============================================"

  let vals := e4m3Values
  let nSamples := 20000

  -- Binades of interest: [2^k, 2^(k+1)) for k = -4,...,2
  let binades := [(-4, -3), (-3, -2), (-2, -1), (-1, 0), (0, 1), (1, 2), (2, 3)]

  IO.println "Binade     | Step Δ    | Prob mass | MSE contrib (×2) | % total"
  IO.println "-----------|-----------|-----------|------------------|--------"

  let mut totalMSE : Float := 0.0
  let mut binadeMSEs : List (String × Float × Float) := []

  for (klo, khi) in binades do
    let lo := Float.pow 2.0 (Float.ofInt klo)
    let hi := Float.pow 2.0 (Float.ofInt khi)
    let step := hi / 8.0  -- E4M3 has 8 levels per binade

    -- Count Gaussian samples in this binade and compute MSE
    let mut mseBinade : Float := 0.0
    let mut probMass : Float := 0.0

    for i in List.range nSamples do
      let u := (Float.ofNat i + 0.5) / Float.ofNat nSamples
      let x := invNormalCDF u
      let ax := Float.abs x
      if ax >= lo && ax < hi then
        let qx := e4m3Quantize vals x
        mseBinade := mseBinade + (x - qx) * (x - qx)
        probMass := probMass + 1.0

    let mseContrib := mseBinade / Float.ofNat nSamples * 2.0  -- ×2 for both tails
    totalMSE := totalMSE + mseContrib / 2.0
    let probFrac := probMass / Float.ofNat nSamples * 2.0

    binadeMSEs := binadeMSEs ++ [(s!"[2^{klo}, 2^{khi})", mseContrib, probFrac)]

  for (name, mse, prob) in binadeMSEs do
    let pct := mse / (totalMSE * 2.0) * 100.0
    IO.println s!"  {name}  | -         | {prob}  | {mse}  | {pct}%"

  IO.println s!"  Total MSE (one-sided) = {totalMSE}"
  IO.println s!"  SQNR = {-10.0 * Float.log (totalMSE * 2.0) / Float.log 10.0} dB"

-- ============================================================
-- Part 2: Bit Allocation Formula
-- ============================================================

#eval! do
  IO.println ""
  IO.println "============================================"
  IO.println "  Bit Allocation: reverse water-filling"
  IO.println "============================================"
  IO.println ""
  IO.println "  Parameters:"
  IO.println "    c_w = 1, c_a = 1, c_g = 10⁻⁶"
  IO.println "    σ²_w = 2.5×10⁻³, σ²_a = 1.0, σ²_g = 4.3×10⁻³"
  IO.println "    B_total = 24"
  IO.println ""

  let cw := 1.0
  let ca := 1.0
  let cg := 1.0e-6
  let sw2 := 2.5e-3
  let sa2 := 1.0
  let sg2 := 4.3e-3
  let bTotal := 24.0

  let csw := cw * sw2
  let csa := ca * sa2
  let csg := cg * sg2

  IO.println s!"  c_w·σ²_w = {csw}"
  IO.println s!"  c_a·σ²_a = {csa}"
  IO.println s!"  c_g·σ²_g = {csg}"

  -- Geometric mean
  let g := Float.exp ((Float.log csw + Float.log csa + Float.log csg) / 3.0)
  IO.println s!"  G = (∏ c_t·σ²_t)^(1/3) = {g}"

  -- Allocations: B_t = B_total/3 + (1/2)·log₂(c_t·σ²_t / G)
  let log2 := Float.log 2.0
  let bw := bTotal / 3.0 + 0.5 * Float.log (csw / g) / log2
  let ba := bTotal / 3.0 + 0.5 * Float.log (csa / g) / log2
  let bg := bTotal / 3.0 + 0.5 * Float.log (csg / g) / log2

  IO.println s!"  B_w = {bw}"
  IO.println s!"  B_a = {ba}"
  IO.println s!"  B_g = {bg}"
  IO.println s!"  Sum = {bw + ba + bg}"
  IO.println ""
  IO.println s!"  Expected: (9.75, 14.07, 0.18)"
  IO.println s!"  Match B_w ≈ 9.75: {if Float.abs (bw - 9.75) < 0.1 then "✓" else "✗"}"
  IO.println s!"  Match B_a ≈ 14.07: {if Float.abs (ba - 14.07) < 0.1 then "✓" else "✗"}"
  IO.println s!"  Match B_g ≈ 0.18: {if Float.abs (bg - 0.18) < 0.1 then "✓" else "✗"}"

-- ============================================================
-- Part 3: Quadratic Approximation Error
-- ============================================================

-- Approximation: 2^f ≈ 1 + a·f + (1-a)·f² for f ∈ [0,1)
-- Document coefficients: a = 0.6564

#eval! do
  IO.println ""
  IO.println "============================================"
  IO.println "  Quadratic approximation: 2^f ≈ 1 + af + (1-a)f²"
  IO.println "============================================"

  -- Test document coefficients
  let a_doc := 0.6564
  let b_doc := 1.0 - a_doc  -- = 0.3436

  -- Test minimax optimal coefficients
  let a_opt := 0.6602
  let b_opt := 1.0 - a_opt  -- = 0.3398

  IO.println ""
  IO.println "  Document coefficients: a=0.6564, b=0.3436"

  let mut maxErrDoc : Float := 0.0
  let mut maxFDoc : Float := 0.0
  let mut maxErrOpt : Float := 0.0
  let mut maxFOpt : Float := 0.0

  let nSteps := 10000
  for i in List.range nSteps do
    let f := Float.ofNat i / Float.ofNat nSteps
    let exact := Float.pow 2.0 f
    let approxDoc := 1.0 + a_doc * f + b_doc * f * f
    let approxOpt := 1.0 + a_opt * f + b_opt * f * f
    let relErrDoc := Float.abs (exact - approxDoc) / exact
    let relErrOpt := Float.abs (exact - approxOpt) / exact
    if relErrDoc > maxErrDoc then
      maxErrDoc := relErrDoc
      maxFDoc := f
    if relErrOpt > maxErrOpt then
      maxErrOpt := relErrOpt
      maxFOpt := f

  IO.println s!"    Max relative error: {maxErrDoc * 100.0}% at f={maxFDoc}"
  IO.println s!"    Expected: 0.32%"
  IO.println s!"    Match: {if Float.abs (maxErrDoc * 100.0 - 0.32) < 0.05 then "✓" else "✗"}"
  IO.println ""

  IO.println "  Minimax coefficients: a=0.6602, b=0.3398"
  IO.println s!"    Max relative error: {maxErrOpt * 100.0}% at f={maxFOpt}"
  IO.println s!"    Expected: 0.27%"
  IO.println s!"    Match: {if Float.abs (maxErrOpt * 100.0 - 0.27) < 0.05 then "✓" else "✗"}"

  IO.println ""
  IO.println "  Old claim of < 0.1% is FALSE — even optimal quadratic gives 0.27%"
  IO.println "  Need cubic or piecewise to reach < 0.1%"

-- Verify endpoint conditions
#eval! do
  IO.println ""
  IO.println "============================================"
  IO.println "  Endpoint conditions"
  IO.println "============================================"

  let a := 0.6564
  let b := 1.0 - a
  IO.println s!"  p(0) = 1 + 0 + 0 = {1.0 + a * 0.0 + b * 0.0 * 0.0} (should be 1)"
  IO.println s!"  p(1) = 1 + a + b = {1.0 + a + b} (should be 2)"
  IO.println s!"  2^0 = {Float.pow 2.0 0.0}"
  IO.println s!"  2^1 = {Float.pow 2.0 1.0}"

  -- Error profile at key points
  IO.println ""
  IO.println "  Error profile:"
  for f in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] do
    let exact := Float.pow 2.0 f
    let approx := 1.0 + a * f + b * f * f
    let relErr := (exact - approx) / exact * 100.0
    IO.println s!"    f={f}: exact={exact}, approx={approx}, rel_err={relErr}%"

-- ============================================================
-- Part 4: QF8 vs E4M3 SQNR comparison
-- ============================================================

#eval! do
  IO.println ""
  IO.println "============================================"
  IO.println "  QF8 vs E4M3 SQNR comparison"
  IO.println "============================================"

  -- QF8: log-uniform, N=256, R=16 octaves
  let eps := 16.0 * Float.log 2.0 / 256.0
  let qf8_nmse := eps * eps / 12.0
  let qf8_sqnr := -10.0 * Float.log qf8_nmse / Float.log 10.0

  IO.println s!"  QF8: NMSE = {qf8_nmse}, SQNR = {qf8_sqnr} dB"
  IO.println s!"  E4M3: SQNR ≈ 31.5 dB (computed above)"
  IO.println s!"  Gap: {qf8_sqnr - 31.5} dB"
  IO.println s!"  Noise power ratio: {Float.pow 10.0 ((qf8_sqnr - 31.5) / 10.0)}×"

-- ============================================================
-- Summary
-- ============================================================

/-
  NUMERICAL CORRECTIONS VERIFIED:
  ===============================

  1. E4M3 SQNR:
     - Old: 25.1 dB (wrong — omitted sub-unity binades)
     - Corrected: ~31.5 dB ✓
     - Our computation: ~31.6 dB ✓

  2. Bit allocation (B_total = 24):
     - Old: (7.6, 11.5, 4.9) [not reproduced — different parameters?]
     - Corrected: (9.75, 14.07, 0.18) ✓
     - Our computation matches to < 0.1 bits ✓

  3. Quadratic approximation max error:
     - Old: < 0.1% (false)
     - Corrected: 0.32% with a=0.6564 ✓
     - Minimax optimal: 0.27% with a=0.6602 ✓
     - Our computation matches both ✓

  4. QF8 vs E4M3 gap: ~6.6 dB ✓
-/
