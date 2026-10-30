"""Bit-exact fixed-point reference functions shared with the RTL.

Every function here uses Python ints only (no floats at inference time), and
matches the corresponding RTL unit bit for bit. The ML side can import this
module for the quantized model so both sides run the same arithmetic.
"""

import math

# ---------------- requantize (contract: Arithmetic rules) ----------------

def requant(acc: int, shift: int, bits: int = 16) -> int:
    """Round half up, arithmetic shift right, saturate. Matches rtl/requant.sv."""
    if shift > 0:
        acc = (acc + (1 << (shift - 1))) >> shift   # Python >> on int is arithmetic
    elif shift < 0:
        acc = acc << (-shift)
    lo, hi = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    return max(lo, min(hi, acc))


# ---------------- RMSNorm (contract: Nonlinear units) ----------------
#
# h_i = x_i / sqrt(mean(x^2) + eps). Because the norm is scale-invariant, the
# input's fraction bits cancel. The mean's divide by D = 16 is folded into the
# result instead of done as a (lossy) floor shift:
#   h = X / sqrt(ss/16 + EPS_RAW) = 4 X / sqrt(M),   M = ss + 16 * EPS_RAW
# 1/sqrt(M) is computed as:
#   1. e   = the even bit position at or just below M's leading one,
#            so F = M / 2^e lies in [1, 4)
#   2. idx = floor(F * 2^FB), an index in [2^FB, 2^LUT_BITS)
#   3. G   = RSQRT_LUT[idx] ~ 2^15 / sqrt(F)
#   4. 1/sqrt(M) = G * 2^(-15 - e/2)
# so h_raw = 4 X G 2^(-15 - e/2) 2^out_frac = requant(X * G, 13 + e/2 - out_frac).

LUT_BITS = 10                 # 1024 entries: one BRAM18
FB = LUT_BITS - 2             # fraction bits of F in the index
D_MODEL = 16
EPS_RAW = 10                  # round(1e-5 * 2^20) for Q6.10 inputs
RMS_SHIFT_BASE = 13           # 15 (LUT scale) - log2(sqrt(D_MODEL))


def rsqrt_lut() -> list[int]:
    """1024 entries; entry i = round(2^15 / sqrt(midpoint of its F interval)).

    Entries below 2^FB are never addressed (F >= 1) and are 0.
    """
    lut = [0] * (1 << FB)
    for i in range(1 << FB, 1 << LUT_BITS):
        f_mid = (i + 0.5) / (1 << FB)
        lut.append(math.floor(2**15 / math.sqrt(f_mid) + 0.5))
    return lut


def rmsnorm_fx(x: list[int], out_frac: int, lut: list[int]) -> list[int]:
    """Bit-exact RMSNorm. x: D_MODEL raw ints (any Q-format). Matches rtl/rmsnorm.sv."""
    assert len(x) == D_MODEL
    ss = sum(v * v for v in x)
    m = ss + D_MODEL * EPS_RAW
    lead = m.bit_length() - 1
    e = lead - (lead & 1)                      # even, so sqrt(2^e) = 2^(e/2)
    idx = m >> (e - FB) if e >= FB else m << (FB - e)
    g = lut[idx]
    shift = RMS_SHIFT_BASE + e // 2 - out_frac
    return [requant(v * g, shift) for v in x]


def write_mem(path, values: list[int], bits: int = 16) -> None:
    """One two's-complement hex word per line (contract: test-vector format)."""
    digits = (bits + 3) // 4
    mask = (1 << bits) - 1
    path.parent.mkdir(exist_ok=True)
    path.write_text("".join(f"{v & mask:0{digits}x}\n" for v in values))
