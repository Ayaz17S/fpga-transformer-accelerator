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


# ---------------- softmax (contract: Nonlinear units) ----------------
#
# scores are Q6.10. With m = max score and the difference rounded to the
# nearest table step of 1/64:
#   e_j = EXP_LUT[min(1023, (m - s_j + 8) >> 4)]   exp(-k/64) in Q2.14
#   S   = sum e_j                                   in [1.0, 16.0] since e_max = 1.0
#   R   = floor(2^30 / S)                           exact integer division
#   p_j = requant(e_j * R, 16)                      Q2.14

EXP_LUT_BITS = 10             # 1024 entries: one BRAM18, covers differences 0 .. 16
EXP_IDX_SHIFT = 4             # Q6.10 difference -> steps of 1/64
SM_RECIP_BITS = 30
SM_PROB_SHIFT = 16


def exp_lut() -> list[int]:
    """1024 entries; entry k = round(exp(-k/64) * 2^14). Entry 0 is exactly 1.0."""
    steps = 1 << (10 - EXP_IDX_SHIFT)
    return [math.floor(math.exp(-k / steps) * 2**14 + 0.5) for k in range(1 << EXP_LUT_BITS)]


def softmax_fx(scores: list[int], lut: list[int]) -> list[int]:
    """Bit-exact softmax over 1..16 scores. Matches rtl/softmax.sv."""
    m = max(scores)
    top = (1 << EXP_LUT_BITS) - 1
    half = 1 << (EXP_IDX_SHIFT - 1)            # round the index to nearest
    e = [lut[min(top, (m - s + half) >> EXP_IDX_SHIFT)] for s in scores]
    recip = (1 << SM_RECIP_BITS) // sum(e)
    return [requant(x * recip, SM_PROB_SHIFT) for x in e]


# ---------------- attention for one token (contract: Model reference 3.3-3.4) ----------------
#
# q: 16 values; ks, vs: the KV cache rows for positions 0..t (t+1 rows of 16).
# Head h uses dims 4h..4h+3. All tensors Q6.10, probabilities Q2.14.
#   score_j = requant(q_h . k_h[j], 11)   Q12.20 -> Q6.10, the extra bit is the x0.5
#   p       = softmax(score_0..t)
#   out_d   = requant(sum_j p_j v[j][d], 14)   Q8.24 -> Q6.10

N_HEAD = 4
HEAD_DIM = 4
ATT_SCORE_SHIFT = 11
ATT_OUT_SHIFT = 14


def attention_fx(q: list[int], ks: list[list[int]], vs: list[list[int]],
                 lut: list[int]) -> list[int]:
    """Bit-exact causal attention for one token. Matches rtl/attention.sv."""
    out = [0] * (N_HEAD * HEAD_DIM)
    for h in range(N_HEAD):
        dims = range(h * HEAD_DIM, (h + 1) * HEAD_DIM)
        scores = [requant(sum(q[d] * k[d] for d in dims), ATT_SCORE_SHIFT) for k in ks]
        p = softmax_fx(scores, lut)
        for d in dims:
            out[d] = requant(sum(pj * v[d] for pj, v in zip(p, vs)), ATT_OUT_SHIFT)
    return out


def write_mem(path, values: list[int], bits: int = 16) -> None:
    """One two's-complement hex word per line (contract: test-vector format)."""
    digits = (bits + 3) // 4
    mask = (1 << bits) - 1
    path.parent.mkdir(exist_ok=True)
    path.write_text("".join(f"{v & mask:0{digits}x}\n" for v in values))
