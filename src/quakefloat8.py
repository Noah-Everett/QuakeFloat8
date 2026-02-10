"""
QuakeFloat8 (QF8) — A Block Log-Domain Number Format for ML

This module implements a software emulator for the QuakeFloat8 format, a novel
8-bit number representation inspired by the Quake III fast inverse square root
trick, optimized for transformer training workloads.

Format specification:
    Block structure:
        - Block size: k = 32 elements
        - Shared scale: 8-bit E8M0 (power of 2, range 2^-127 to 2^127)
        - Overhead: 0.25 bits/element → 8.25 effective bits/element

    Per-element encoding (8 bits):
        Bit 7 (MSB): Sign bit (0 = positive, 1 = negative)
        Bits 6–0:    Log₂ magnitude in biased u3.4 fixed-point

    Value = (-1)^s × shared_scale × 2^((code - bias) / 16)

    where:
        s    = sign bit
        code = 7-bit unsigned integer (bits 6:0), range [0, 127]
        bias = 64  (centers the representable range around 1.0)

    Special encodings:
        code = 0 (either sign): exact zero

    Arithmetic:
        Multiplication: sign XOR + code addition (log-domain addition)
        Dot product: log-domain multiply → LUT to linear → Kulisch accumulation

References:
    - Johnson (2018), "Rethinking Floating Point for Deep Learning" (arXiv:1811.01721)
    - Darvish Rouhani et al. (2023), "Microscaling Data Formats" (arXiv:2310.10537)
    - The Quake III fast inverse square root algorithm

Author: QuakeFloat Research Project
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Tuple, Union

# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════

BLOCK_SIZE = 32          # Elements per block (matches OCP MX standard)
CODE_BITS = 7            # Magnitude bits (excl. sign)
FRAC_BITS = 4            # Fractional bits in u3.4 fixed-point log
INT_BITS = 3             # Integer bits in u3.4 fixed-point log
BIAS = 64                # Code bias (centers range around 2^0 = 1.0)
MAX_CODE = 127           # Maximum 7-bit code
STEP = 1.0 / (1 << FRAC_BITS)  # = 1/16 = 0.0625 (log₂ step between adjacent codes)

# Derived constants
MIN_POSITIVE_MAG = 2.0 ** ((1 - BIAS) / (1 << FRAC_BITS))   # code=1: 2^(-63/16) ≈ 0.0652
MAX_MAG = 2.0 ** ((MAX_CODE - BIAS) / (1 << FRAC_BITS))     # code=127: 2^(63/16) ≈ 15.35

# E8M0 shared scale: 8-bit exponent-only (power of 2)
E8M0_MIN_EXP = -127
E8M0_MAX_EXP = 127

# Precompute the log-to-linear LUT (16 entries for 4 fractional bits)
# LUT[f] = 2^(f/16) - 1  for f in [0, 15]
_LOG_TO_LINEAR_LUT = np.array([2.0 ** (f / 16.0) - 1.0 for f in range(16)],
                               dtype=np.float64)

# Precompute all 128 code→magnitude values for fast decoding
_CODE_TO_MAG = np.zeros(128, dtype=np.float64)
_CODE_TO_MAG[0] = 0.0  # Zero
for c in range(1, 128):
    _CODE_TO_MAG[c] = 2.0 ** ((c - BIAS) / 16.0)


# ═══════════════════════════════════════════════════════════════════════════════
# Core Encoding / Decoding
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_shared_scale(values: np.ndarray) -> Tuple[int, float]:
    """Compute the E8M0 shared scale exponent for a block of values.

    The shared exponent is chosen so that the maximum absolute value in the
    block maps to near the top of the per-element representable range.

    Args:
        values: 1-D array of float values (up to BLOCK_SIZE elements).

    Returns:
        (exponent: int, scale: float) where scale = 2^exponent.
        The exponent is clamped to [E8M0_MIN_EXP, E8M0_MAX_EXP].
    """
    abs_max = np.max(np.abs(values))
    if abs_max == 0.0:
        return 0, 1.0

    # We want: abs_max / scale ≈ MAX_MAG
    # => scale = abs_max / MAX_MAG
    # => exponent = floor(log2(abs_max / MAX_MAG))
    # The per-element max magnitude is 2^(63/16), so:
    # exponent = floor(log2(abs_max)) - floor(63/16)
    # More precisely, we want the element code for abs_max/scale to be ≤ 127.
    #
    # Element value = scale × 2^((code - 64)/16)
    # => code = 64 + 16 × log2(value / scale)
    # We want code(abs_max) ≤ 127, so:
    # 64 + 16 × log2(abs_max / scale) ≤ 127
    # log2(abs_max / scale) ≤ 63/16 = 3.9375
    # abs_max / scale ≤ 2^3.9375
    # scale ≥ abs_max / 2^3.9375
    #
    # Choose scale = 2^exponent where exponent = ceil(log2(abs_max)) - 3
    # This ensures the max value fits, with some headroom.

    log2_abs_max = np.log2(abs_max)
    # The element can represent magnitudes up to 2^(63/16) = 2^3.9375
    # We need: abs_max / 2^exponent ≤ 2^3.9375
    # Therefore: exponent ≥ log2(abs_max) - 3.9375
    # Use ceil to ensure the max value fits (doesn't overflow to clipped code).
    exponent = int(np.ceil(log2_abs_max - (MAX_CODE - BIAS) / 16.0))

    # Clamp to E8M0 range
    exponent = max(E8M0_MIN_EXP, min(E8M0_MAX_EXP, exponent))
    scale = 2.0 ** exponent
    return exponent, scale


def encode_element(value: float, scale: float) -> int:
    """Encode a single float value to an 8-bit QF8 code.

    Args:
        value: The float32 value to encode.
        scale: The block shared scale (power of 2).

    Returns:
        8-bit integer: [sign(1)][code(7)]
    """
    if value == 0.0 or scale == 0.0:
        return 0  # Both +0 and -0 map to code 0x00

    sign = 1 if value < 0 else 0
    abs_val = abs(value) / scale

    if abs_val < MIN_POSITIVE_MAG * 0.5:
        # Too small → round to zero (underflow)
        return 0

    # code = round(64 + 16 × log2(abs_val))
    log2_val = np.log2(abs_val)
    code = int(np.round(BIAS + (1 << FRAC_BITS) * log2_val))

    # Clamp to valid range
    if code < 1:
        code = 1  # Minimum positive (avoid zero encoding for nonzero values)
    elif code > MAX_CODE:
        code = MAX_CODE  # Saturate (overflow)

    return (sign << 7) | code


def decode_element(encoded: int, scale: float) -> float:
    """Decode an 8-bit QF8 code back to float.

    Args:
        encoded: 8-bit integer [sign(1)][code(7)]
        scale: The block shared scale (power of 2).

    Returns:
        Decoded float value.
    """
    sign = (encoded >> 7) & 1
    code = encoded & 0x7F

    if code == 0:
        return 0.0

    magnitude = _CODE_TO_MAG[code] * scale
    return -magnitude if sign else magnitude


def encode_block(values: np.ndarray) -> Tuple[np.ndarray, int, float]:
    """Encode a block of float values to QF8.

    Args:
        values: 1-D float array (len ≤ BLOCK_SIZE). Padded with zeros if shorter.

    Returns:
        (codes, exponent, scale):
            codes: uint8 array of encoded values
            exponent: E8M0 shared exponent (int)
            scale: 2^exponent (float)
    """
    values = np.asarray(values, dtype=np.float64).ravel()
    n = len(values)
    assert n <= BLOCK_SIZE, f"Block has {n} elements, max is {BLOCK_SIZE}"

    exponent, scale = _compute_shared_scale(values)
    codes = np.zeros(n, dtype=np.uint8)
    for i in range(n):
        codes[i] = encode_element(float(values[i]), scale)
    return codes, exponent, scale


def decode_block(codes: np.ndarray, scale: float) -> np.ndarray:
    """Decode a block of QF8 codes back to float.

    Args:
        codes: uint8 array of QF8 codes.
        scale: Block shared scale (power of 2).

    Returns:
        Float64 array of decoded values.
    """
    codes = np.asarray(codes, dtype=np.uint8)
    result = np.zeros(len(codes), dtype=np.float64)
    for i in range(len(codes)):
        result[i] = decode_element(int(codes[i]), scale)
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Vectorized Encoding / Decoding (for performance)
# ═══════════════════════════════════════════════════════════════════════════════

def encode_block_vectorized(values: np.ndarray) -> Tuple[np.ndarray, int, float]:
    """Vectorized block encoding (much faster for large blocks).

    Args:
        values: 1-D float array (len ≤ BLOCK_SIZE).

    Returns:
        (codes, exponent, scale)
    """
    values = np.asarray(values, dtype=np.float64).ravel()
    n = len(values)
    assert n <= BLOCK_SIZE

    exponent, scale = _compute_shared_scale(values)

    signs = (values < 0).astype(np.uint8)
    abs_vals = np.abs(values) / scale

    codes = np.zeros(n, dtype=np.uint8)
    nonzero = abs_vals >= (MIN_POSITIVE_MAG * 0.5)

    if np.any(nonzero):
        log2_vals = np.log2(np.maximum(abs_vals[nonzero], 1e-45))
        raw_codes = np.round(BIAS + (1 << FRAC_BITS) * log2_vals).astype(np.int32)
        raw_codes = np.clip(raw_codes, 1, MAX_CODE).astype(np.uint8)
        codes[nonzero] = (signs[nonzero] << 7) | raw_codes

    return codes, exponent, scale


def decode_block_vectorized(codes: np.ndarray, scale: float) -> np.ndarray:
    """Vectorized block decoding.

    Args:
        codes: uint8 array of QF8 codes.
        scale: Block shared scale.

    Returns:
        Float64 array of decoded values.
    """
    codes = np.asarray(codes, dtype=np.uint8)
    signs = (codes >> 7) & 1
    magnitudes_idx = codes & 0x7F
    magnitudes = _CODE_TO_MAG[magnitudes_idx] * scale
    result = np.where(signs, -magnitudes, magnitudes)
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Tensor-Level Encoding / Decoding
# ═══════════════════════════════════════════════════════════════════════════════

class QF8Tensor:
    """A tensor quantized in QuakeFloat8 format.

    Stores:
        - codes: uint8 array of per-element QF8 codes (same shape as original)
        - exponents: int array of per-block E8M0 exponents
        - scales: float array of per-block scales (2^exponent)
        - shape: original tensor shape
        - block_axis: axis along which blocks are formed (default: last axis)
    """

    def __init__(self, codes: np.ndarray, exponents: np.ndarray,
                 scales: np.ndarray, shape: tuple, block_axis: int = -1):
        self.codes = codes
        self.exponents = exponents
        self.scales = scales
        self.shape = shape
        self.block_axis = block_axis

    @property
    def nbytes_data(self) -> int:
        """Total bytes for element data."""
        return self.codes.nbytes

    @property
    def nbytes_scales(self) -> int:
        """Total bytes for block scales."""
        return self.exponents.nbytes  # 1 byte per block (E8M0)

    @property
    def nbytes_total(self) -> int:
        """Total storage in bytes."""
        return self.nbytes_data + self.nbytes_scales

    @property
    def bits_per_element(self) -> float:
        """Effective bits per element (including amortized scale overhead)."""
        n_elements = np.prod(self.shape)
        return (self.nbytes_total * 8) / n_elements if n_elements > 0 else 0

    def dequantize(self) -> np.ndarray:
        """Convert back to float64 tensor."""
        return qf8_decode_tensor(self)


def qf8_encode_tensor(tensor: np.ndarray, block_axis: int = -1) -> QF8Tensor:
    """Encode an arbitrary tensor to QF8 format.

    The tensor is partitioned into blocks of BLOCK_SIZE along the specified axis.
    If the axis length is not a multiple of BLOCK_SIZE, the last block is shorter.

    Args:
        tensor: Input float tensor.
        block_axis: Axis along which to form blocks (default: last).

    Returns:
        QF8Tensor object.
    """
    tensor = np.asarray(tensor, dtype=np.float64)
    shape = tensor.shape
    ndim = len(shape)

    # Normalize axis
    if block_axis < 0:
        block_axis = ndim + block_axis
    axis_len = shape[block_axis]
    n_blocks = int(np.ceil(axis_len / BLOCK_SIZE))

    # Flatten all other dims, iterate blocks along the target axis
    # Move target axis to last position for easy slicing
    tensor_moved = np.moveaxis(tensor, block_axis, -1)
    flat_shape = (-1, axis_len)
    tensor_flat = tensor_moved.reshape(flat_shape)
    n_rows = tensor_flat.shape[0]

    codes_flat = np.zeros_like(tensor_flat, dtype=np.uint8)
    exponents = np.zeros((n_rows, n_blocks), dtype=np.int32)
    scales = np.zeros((n_rows, n_blocks), dtype=np.float64)

    for row in range(n_rows):
        for blk in range(n_blocks):
            start = blk * BLOCK_SIZE
            end = min(start + BLOCK_SIZE, axis_len)
            block_vals = tensor_flat[row, start:end]

            block_codes, exp, scl = encode_block_vectorized(block_vals)
            codes_flat[row, start:end] = block_codes
            exponents[row, blk] = exp
            scales[row, blk] = scl

    # Reshape codes back
    codes = codes_flat.reshape(tensor_moved.shape)
    codes = np.moveaxis(codes, -1, block_axis)

    return QF8Tensor(codes, exponents, scales, shape, block_axis)


def qf8_decode_tensor(qt: QF8Tensor) -> np.ndarray:
    """Decode a QF8Tensor back to float64.

    Args:
        qt: QF8Tensor to decode.

    Returns:
        Float64 array with the original shape.
    """
    shape = qt.shape
    ndim = len(shape)
    block_axis = qt.block_axis
    if block_axis < 0:
        block_axis = ndim + block_axis
    axis_len = shape[block_axis]
    n_blocks = int(np.ceil(axis_len / BLOCK_SIZE))

    codes_moved = np.moveaxis(qt.codes, block_axis, -1)
    flat_shape = (-1, axis_len)
    codes_flat = codes_moved.reshape(flat_shape)
    n_rows = codes_flat.shape[0]

    result_flat = np.zeros_like(codes_flat, dtype=np.float64)

    for row in range(n_rows):
        for blk in range(n_blocks):
            start = blk * BLOCK_SIZE
            end = min(start + BLOCK_SIZE, axis_len)
            block_codes = codes_flat[row, start:end]
            scl = qt.scales[row, blk]

            result_flat[row, start:end] = decode_block_vectorized(block_codes, scl)

    result = result_flat.reshape(codes_moved.shape)
    result = np.moveaxis(result, -1, block_axis)
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# QF8 Arithmetic Operations
# ═══════════════════════════════════════════════════════════════════════════════

def qf8_multiply_codes(code_a: int, code_b: int) -> Tuple[int, int]:
    """Multiply two QF8 elements in the log domain.

    In log domain, multiplication is addition of the codes:
        product_log = log_a + log_b
        product_code_sum = (code_a - bias) + (code_b - bias)
                         = code_a + code_b - 2*bias

    Args:
        code_a: 8-bit QF8 code (with sign bit)
        code_b: 8-bit QF8 code (with sign bit)

    Returns:
        (sign, log_sum): sign of product (0 or 1), and the unbiased log sum
        as a fixed-point value (integer * 16 + frac, where the actual log value
        is log_sum / 16).
        Returns (0, None) if either input is zero.
    """
    sign_a = (code_a >> 7) & 1
    sign_b = (code_b >> 7) & 1
    mag_a = code_a & 0x7F
    mag_b = code_b & 0x7F

    # Zero handling
    if mag_a == 0 or mag_b == 0:
        return (0, None)  # Product is zero

    sign_out = sign_a ^ sign_b

    # Log-domain addition: the product's log₂ magnitude is:
    # ((mag_a - 64) + (mag_b - 64)) / 16
    # = (mag_a + mag_b - 128) / 16
    log_sum = (mag_a + mag_b) - 2 * BIAS  # Range: [-126, +126], in units of 1/16

    return (sign_out, log_sum)


def qf8_product_to_linear(sign: int, log_sum: int,
                           scale_a: float, scale_b: float) -> float:
    """Convert a log-domain product to linear value (for accumulation).

    Uses the log-to-linear LUT to reconstruct the linear magnitude.

    Args:
        sign: 0 for positive, 1 for negative
        log_sum: The log₂ of the magnitude × 16 (fixed-point)
        scale_a: Block scale for operand a
        scale_b: Block scale for operand b

    Returns:
        Linear float value of the product.
    """
    if log_sum is None:
        return 0.0

    # Split into integer and fractional parts
    # log_sum is in units of 1/16, so:
    #   integer_part = log_sum >> 4  (arithmetic shift)
    #   frac_part = log_sum & 0xF
    # But log_sum can be negative, so we use Python's divmod
    int_part, frac_part = divmod(log_sum, 16)
    # divmod with positive divisor always gives frac_part in [0, 15]

    # LUT lookup: 2^(frac/16) - 1
    p_f = _LOG_TO_LINEAR_LUT[frac_part]

    # Linear magnitude = 2^int_part × (1 + p(frac))
    # Combined with block scales: value = scale_a × scale_b × 2^int_part × (1 + p_f)
    magnitude = scale_a * scale_b * (2.0 ** int_part) * (1.0 + p_f)

    return -magnitude if sign else magnitude


def qf8_dot_product(codes_a: np.ndarray, scale_a: float,
                     codes_b: np.ndarray, scale_b: float) -> float:
    """Compute the dot product of two QF8 vectors using Kulisch-style accumulation.

    This implements the core ELMA operation:
    1. Multiply in log domain (addition of codes)
    2. Convert each product to linear via LUT
    3. Accumulate in high-precision linear domain (Kulisch)

    In hardware, the Kulisch accumulator would be a wide fixed-point register.
    In this software emulation, we use float64 (which provides ~53 bits of
    mantissa, far exceeding the ~38-bit Kulisch register needed for QF8).

    Args:
        codes_a: uint8 array of QF8 codes for vector a
        scale_a: Block shared scale for a
        codes_b: uint8 array of QF8 codes for vector b
        scale_b: Block shared scale for b

    Returns:
        Float64 scalar: the dot product.
    """
    assert len(codes_a) == len(codes_b), "Vectors must be same length"

    # Kulisch accumulator (emulated with float64)
    accumulator = 0.0

    for i in range(len(codes_a)):
        sign, log_sum = qf8_multiply_codes(int(codes_a[i]), int(codes_b[i]))
        if log_sum is not None:
            accumulator += qf8_product_to_linear(sign, log_sum, scale_a, scale_b)

    return accumulator


def qf8_dot_product_vectorized(codes_a: np.ndarray, scale_a: float,
                                codes_b: np.ndarray, scale_b: float) -> float:
    """Vectorized dot product (much faster, same numerical result).

    Args:
        codes_a, codes_b: uint8 arrays of QF8 codes.
        scale_a, scale_b: Block shared scales.

    Returns:
        Float64 scalar dot product.
    """
    signs_a = (codes_a >> 7) & 1
    signs_b = (codes_b >> 7) & 1
    mags_a = (codes_a & 0x7F).astype(np.int32)
    mags_b = (codes_b & 0x7F).astype(np.int32)

    # Find nonzero pairs
    nonzero = (mags_a > 0) & (mags_b > 0)
    if not np.any(nonzero):
        return 0.0

    # Sign of products
    prod_signs = signs_a[nonzero] ^ signs_b[nonzero]

    # Log-domain addition
    log_sums = mags_a[nonzero] + mags_b[nonzero] - 2 * BIAS  # in units of 1/16

    # Split into integer and fractional parts
    # Use numpy divmod for vectorized operation
    int_parts, frac_parts = np.divmod(log_sums, 16)

    # LUT lookup
    p_f = _LOG_TO_LINEAR_LUT[frac_parts]

    # Linear magnitudes
    magnitudes = scale_a * scale_b * (2.0 ** int_parts.astype(np.float64)) * (1.0 + p_f)

    # Apply signs and accumulate
    signed_vals = np.where(prod_signs, -magnitudes, magnitudes)
    return np.sum(signed_vals)


# ═══════════════════════════════════════════════════════════════════════════════
# QF8 Matrix Multiplication
# ═══════════════════════════════════════════════════════════════════════════════

def qf8_matmul(qt_a: QF8Tensor, qt_b: QF8Tensor) -> np.ndarray:
    """Matrix multiplication of two QF8-quantized tensors.

    Computes A @ B where A and B are quantized in QF8 format.
    A is (M, K), B is (K, N), result is (M, N) in float64.

    Both tensors must be 2-D and quantized along axis=-1 (the reduction axis).
    For A (M×K): blocks along K dimension.
    For B (K×N): must be quantized along axis=0 (K dimension) for proper
    block alignment with the reduction axis.

    This performs block-wise dot products using QF8 arithmetic.

    Args:
        qt_a: QF8Tensor of shape (M, K), quantized along axis=-1
        qt_b: QF8Tensor of shape (K, N), quantized along axis=0

    Returns:
        Float64 array of shape (M, N).
    """
    assert len(qt_a.shape) == 2 and len(qt_b.shape) == 2
    M, K = qt_a.shape
    K2, N = qt_b.shape
    assert K == K2, f"Inner dimensions must match: {K} vs {K2}"

    result = np.zeros((M, N), dtype=np.float64)
    n_blocks = int(np.ceil(K / BLOCK_SIZE))

    # Get codes in the right layout
    codes_a = qt_a.codes  # (M, K)
    codes_b = qt_b.codes  # (K, N)

    for i in range(M):
        for j in range(N):
            acc = 0.0
            for blk in range(n_blocks):
                start = blk * BLOCK_SIZE
                end = min(start + BLOCK_SIZE, K)

                # Get block codes
                a_block = codes_a[i, start:end]
                b_block = codes_b[start:end, j]

                # Get block scales
                # For A: blocks along axis=-1 (last axis), row i, block blk
                sa = qt_a.scales[i, blk]
                # For B: after moveaxis(0→-1), scales are indexed [col, block]
                # regardless of block_axis (the encoding flattened it this way)
                sb = qt_b.scales[j, blk]

                # Accumulate this block's dot product
                acc += qf8_dot_product_vectorized(a_block, sa, b_block, sb)

            result[i, j] = acc

    return result


def qf8_matmul_fast(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Convenience function: quantize both matrices and multiply.

    Simulates the full QF8 matmul pipeline:
    1. Quantize A along reduction axis (axis=-1, i.e., columns)
    2. Quantize B along reduction axis (axis=0, i.e., rows)
    3. Perform QF8 matrix multiplication

    Args:
        a: Float array of shape (M, K)
        b: Float array of shape (K, N)

    Returns:
        Float64 array of shape (M, N).
    """
    qt_a = qf8_encode_tensor(a, block_axis=-1)
    qt_b = qf8_encode_tensor(b, block_axis=0)
    return qf8_matmul(qt_a, qt_b)


# ═══════════════════════════════════════════════════════════════════════════════
# Simulated FP8 E4M3 (for comparison)
# ═══════════════════════════════════════════════════════════════════════════════

def fp8_e4m3_encode(value: float) -> int:
    """Encode a float to FP8 E4M3 format.

    E4M3: 1 sign + 4 exponent + 3 mantissa = 8 bits.
    Bias = 7. No infinities. Single NaN (0x7F / 0xFF).
    Range: ±[2^-9, 448]. Subnormals supported.

    Based on OCP FP8 specification.
    """
    if value == 0.0:
        return 0

    sign = 0
    if value < 0:
        sign = 1
        value = -value

    # E4M3 parameters
    bias = 7
    max_exp = 8  # 15 - bias
    max_normal = 448.0   # (1 + 7/8) × 2^8 = 1.875 × 256
    min_subnormal = 2.0 ** (-9)  # 2^(1-bias) × 2^(-3) = 2^(-6) × 2^(-3) = 2^(-9)

    if value > max_normal:
        # Saturate to max (0x7E = 0_1111_110 = 448)
        return (sign << 7) | 0x7E

    if value < min_subnormal * 0.5:
        # Underflow to zero
        return sign << 7

    # Check for subnormal
    if value < 2.0 ** (1 - bias):  # < 2^(-6)
        # Subnormal: exponent field = 0, mantissa encodes value / 2^(1-bias) × 2^(-3)
        # value = mantissa × 2^(1-bias-3) = mantissa × 2^(-9)
        mantissa = int(np.round(value / min_subnormal))
        mantissa = max(0, min(7, mantissa))
        return (sign << 7) | mantissa
    else:
        # Normal: value = (1 + mantissa/8) × 2^(exp - bias)
        exp = int(np.floor(np.log2(value)))
        significand = value / (2.0 ** exp)  # In [1, 2)
        mantissa = int(np.round((significand - 1.0) * 8))
        if mantissa >= 8:
            mantissa = 0
            exp += 1
        exp_biased = exp + bias
        exp_biased = max(1, min(15, exp_biased))
        # Check for NaN encoding (0x7F) and avoid it
        if exp_biased == 15 and mantissa == 7:
            mantissa = 6  # Saturate to max instead of NaN
        return (sign << 7) | (exp_biased << 3) | mantissa


def fp8_e4m3_decode(encoded: int) -> float:
    """Decode FP8 E4M3 to float."""
    sign = (encoded >> 7) & 1
    exp_biased = (encoded >> 3) & 0xF
    mantissa = encoded & 0x7

    bias = 7

    if exp_biased == 15 and mantissa == 7:
        return float('nan')

    if exp_biased == 0:
        # Subnormal
        value = mantissa * (2.0 ** (1 - bias - 3))
    else:
        value = (1.0 + mantissa / 8.0) * (2.0 ** (exp_biased - bias))

    return -value if sign else value


def fp8_e4m3_quantize(tensor: np.ndarray, scale: Optional[float] = None) -> np.ndarray:
    """Quantize a tensor to FP8 E4M3 and back (round-trip).

    This simulates quantizing to FP8 and immediately dequantizing, which is
    the standard way to evaluate quantization error in software.

    Args:
        tensor: Input float tensor.
        scale: Optional per-tensor scale factor. If None, computed automatically.

    Returns:
        Float64 tensor with FP8 E4M3 quantization noise.
    """
    tensor = np.asarray(tensor, dtype=np.float64)
    flat = tensor.ravel()

    if scale is None:
        abs_max = np.max(np.abs(flat))
        if abs_max == 0:
            return tensor.copy()
        scale = abs_max / 448.0  # Map max to FP8 max
        if scale < 1e-38:
            scale = 1.0

    result = np.zeros_like(flat)
    for i in range(len(flat)):
        encoded = fp8_e4m3_encode(flat[i] / scale)
        result[i] = fp8_e4m3_decode(encoded) * scale

    return result.reshape(tensor.shape)


# ═══════════════════════════════════════════════════════════════════════════════
# Utility Functions
# ═══════════════════════════════════════════════════════════════════════════════

def qf8_quantize_roundtrip(tensor: np.ndarray, block_axis: int = -1) -> np.ndarray:
    """Quantize tensor to QF8 and immediately dequantize (for measuring error).

    Args:
        tensor: Input float tensor.
        block_axis: Axis for block formation.

    Returns:
        Float64 tensor with QF8 quantization noise.
    """
    qt = qf8_encode_tensor(tensor, block_axis=block_axis)
    return qf8_decode_tensor(qt)


def compute_qsnr(original: np.ndarray, quantized: np.ndarray) -> float:
    """Compute Quantization Signal-to-Noise Ratio (QSNR) in dB.

    QSNR = 10 × log10(||signal||² / ||noise||²)

    Args:
        original: Original float tensor.
        quantized: Quantized (round-trip) tensor.

    Returns:
        QSNR in dB. Higher is better.
    """
    signal_power = np.sum(original ** 2)
    noise = original - quantized
    noise_power = np.sum(noise ** 2)

    if noise_power == 0:
        return float('inf')
    if signal_power == 0:
        return float('-inf')

    return 10.0 * np.log10(signal_power / noise_power)


def compute_relative_error(original: np.ndarray, computed: np.ndarray) -> dict:
    """Compute multiple error metrics between original and computed arrays.

    Returns:
        Dictionary with:
            'mse': Mean squared error
            'rmse': Root mean squared error
            'nmse': Normalized MSE (relative to signal power)
            'max_abs_error': Maximum absolute error
            'mean_rel_error': Mean relative error (excluding zeros)
            'qsnr_db': QSNR in dB
    """
    diff = original - computed
    mse = np.mean(diff ** 2)
    signal_power = np.mean(original ** 2)

    # Relative error (avoid division by zero)
    nonzero = np.abs(original) > 1e-30
    if np.any(nonzero):
        rel_errors = np.abs(diff[nonzero]) / np.abs(original[nonzero])
        mean_rel = np.mean(rel_errors)
    else:
        mean_rel = 0.0

    nmse = mse / signal_power if signal_power > 0 else float('inf')
    qsnr = -10.0 * np.log10(nmse) if nmse > 0 else float('inf')

    return {
        'mse': mse,
        'rmse': np.sqrt(mse),
        'nmse': nmse,
        'max_abs_error': np.max(np.abs(diff)),
        'mean_rel_error': mean_rel,
        'qsnr_db': qsnr,
    }


def describe_format():
    """Print a summary of the QF8 format specification."""
    print("=" * 60)
    print("QuakeFloat8 (QF8) Format Specification")
    print("=" * 60)
    print(f"  Block size:           {BLOCK_SIZE} elements")
    print(f"  Per-element bits:     8 (+ 0.25 amortized scale)")
    print(f"  Sign bits:            1")
    print(f"  Log magnitude bits:   {CODE_BITS} (u{INT_BITS}.{FRAC_BITS} fixed-point)")
    print(f"  Code bias:            {BIAS}")
    print(f"  Log₂ step size:       1/{1 << FRAC_BITS} = {STEP:.4f}")
    print(f"  Levels per octave:    {1 << FRAC_BITS}")
    print(f"  Per-element range:    [{MIN_POSITIVE_MAG:.4f}, {MAX_MAG:.4f}]")
    print(f"  Per-element octaves:  {(MAX_CODE - 1) / (1 << FRAC_BITS):.1f}")
    print(f"  Shared scale:         E8M0 (2^-127 to 2^127)")
    print(f"  Max relative error:   {(2**(1/16) - 1)*100:.2f}%")
    print(f"  Adjacent value ratio: 2^(1/16) = {2**(1/16):.6f}")
    print(f"  Total codes:          128 magnitudes × 2 signs = 256")
    print(f"  Useful codes:         127 positive + 127 negative + 1 zero = 255")
    print(f"  LUT size:             {1 << FRAC_BITS} × 8 bits = {(1 << FRAC_BITS)} bytes")
    print("=" * 60)


# ═══════════════════════════════════════════════════════════════════════════════
# Self-Test
# ═══════════════════════════════════════════════════════════════════════════════

def _self_test():
    """Run basic sanity checks."""
    print("Running QF8 self-test...")

    # Test 1: Encode/decode round-trip for known values
    scale = 1.0
    test_values = [0.0, 1.0, -1.0, 0.5, 2.0, 0.1, -3.14, 10.0]
    print("\n  Test 1: Element encode/decode round-trip")
    for v in test_values:
        code = encode_element(v, scale)
        decoded = decode_element(code, scale)
        err = abs(v - decoded) / max(abs(v), 1e-30)
        status = "✓" if (v == 0 and decoded == 0) or err < 0.05 else "✗"
        print(f"    {status} {v:8.4f} → code={code:3d} (0x{code:02X}) → {decoded:8.4f}"
              f"  (rel err: {err:.4f})")

    # Test 2: Block encode/decode
    print("\n  Test 2: Block encode/decode (Gaussian data)")
    np.random.seed(42)
    block = np.random.randn(BLOCK_SIZE) * 0.1
    codes, exp, scl = encode_block_vectorized(block)
    decoded = decode_block_vectorized(codes, scl)
    qsnr = compute_qsnr(block, decoded)
    print(f"    Block scale: 2^{exp} = {scl:.6g}")
    print(f"    QSNR: {qsnr:.1f} dB")
    assert qsnr > 25, f"QSNR too low: {qsnr:.1f} dB"
    print(f"    ✓ QSNR > 25 dB")

    # Test 3: Tensor encode/decode
    print("\n  Test 3: Tensor encode/decode (128×64 matrix)")
    tensor = np.random.randn(128, 64) * 0.02
    qt = qf8_encode_tensor(tensor)
    decoded_tensor = qt.dequantize()
    qsnr = compute_qsnr(tensor, decoded_tensor)
    print(f"    QSNR: {qsnr:.1f} dB")
    print(f"    Bits/element: {qt.bits_per_element:.2f}")
    assert qsnr > 25, f"QSNR too low: {qsnr:.1f} dB"
    print(f"    ✓ QSNR > 25 dB")

    # Test 4: QF8 multiplication (log-domain)
    print("\n  Test 4: Log-domain multiplication")
    a_val, b_val = 2.5, 3.0
    a_code = encode_element(a_val, 1.0)
    b_code = encode_element(b_val, 1.0)
    sign, log_sum = qf8_multiply_codes(a_code, b_code)
    product = qf8_product_to_linear(sign, log_sum, 1.0, 1.0)
    expected = a_val * b_val
    err = abs(product - expected) / expected
    print(f"    {a_val} × {b_val} = {product:.4f} (expected: {expected:.4f}, rel err: {err:.4f})")
    assert err < 0.1, f"Multiplication error too high: {err:.4f}"
    print(f"    ✓ Relative error < 10%")

    # Test 5: Dot product
    print("\n  Test 5: Dot product (32 elements)")
    a = np.random.randn(BLOCK_SIZE) * 0.5
    b = np.random.randn(BLOCK_SIZE) * 0.5
    codes_a, _, sa = encode_block_vectorized(a)
    codes_b, _, sb = encode_block_vectorized(b)
    qf8_dot = qf8_dot_product_vectorized(codes_a, sa, codes_b, sb)
    true_dot = np.dot(a, b)
    err = abs(qf8_dot - true_dot) / max(abs(true_dot), 1e-30)
    print(f"    QF8 dot: {qf8_dot:.6f}, True: {true_dot:.6f}, Rel err: {err:.4f}")
    assert err < 0.15, f"Dot product error too high: {err:.4f}"
    print(f"    ✓ Relative error < 15%")

    # Test 6: Zero handling
    print("\n  Test 6: Zero handling")
    zero_code = encode_element(0.0, 1.0)
    assert zero_code == 0, f"Zero should encode to 0, got {zero_code}"
    assert decode_element(0, 1.0) == 0.0, "Code 0 should decode to 0.0"
    sign, log_sum = qf8_multiply_codes(zero_code, encode_element(5.0, 1.0))
    assert log_sum is None, "Multiply by zero should return None log_sum"
    print(f"    ✓ Zero encode/decode/multiply correct")

    print("\n  All tests passed! ✓")


if __name__ == "__main__":
    describe_format()
    print()
    _self_test()
