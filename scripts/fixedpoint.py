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


def read_mem(path, bits: int = 16) -> list[int]:
    """Inverse of write_mem: hex words back to signed ints."""
    out = []
    for line in path.read_text().split():
        v = int(line, 16)
        out.append(v - (1 << bits) if v >= 1 << (bits - 1) else v)
    return out


# ---------------- whole model: one token step (contract: Model reference) ----------------

VOCAB, CTX, D_FF = 27, 16, 64

# weight ROM layout, in order - must match rtl/model_pkg.sv
LAYOUT = [
    ("wte",     VOCAB * D_MODEL),
    ("wpe",     CTX * D_MODEL),
    ("wq",      D_MODEL * D_MODEL),
    ("wk",      D_MODEL * D_MODEL),
    ("wv",      D_MODEL * D_MODEL),
    ("wo",      D_MODEL * D_MODEL),
    ("fc1",     D_FF * D_MODEL),
    ("fc2",     D_MODEL * D_FF),
    ("lm_head", VOCAB * D_MODEL),
]
BASE = {}
_addr = 0
for _name, _count in LAYOUT:
    BASE[_name] = _addr
    _addr += _count
N_WEIGHTS = _addr  # 4192

EMB_SHIFT = 4          # wte + wpe: Q2.14 -> Q6.10
SHIFT_FROM_NORM = 16   # Q2.14 weights x Q4.12 normed input  -> Q6.10
SHIFT_FROM_ACT = 14    # Q2.14 weights x Q6.10 activation    -> Q6.10
NORM0_FRAC, NORM_FRAC = 10, 12


def sat16(v: int) -> int:
    return max(-32768, min(32767, v))


def matvec_fx(w: list[int], name: str, n_out: int, n_in: int,
              x: list[int], shift: int) -> list[int]:
    """y = requant(W x). Matches rtl/matvec.sv."""
    base = BASE[name]
    return [requant(sum(w[base + o * n_in + i] * x[i] for i in range(n_in)), shift)
            for o in range(n_out)]


class GptFx:
    """Bit-exact model. Keeps the KV cache between step() calls. Matches rtl/gpt_core.sv."""

    def __init__(self, weights: list[int], rsqrt: list[int], exp: list[int]):
        assert len(weights) == N_WEIGHTS
        self.w, self.rsqrt, self.exp = weights, rsqrt, exp
        self.k_cache = [None] * CTX
        self.v_cache = [None] * CTX

    def step(self, token: int, pos: int) -> dict:
        w = self.w
        emb = [requant(w[BASE["wte"] + token * D_MODEL + d] + w[BASE["wpe"] + pos * D_MODEL + d],
                       EMB_SHIFT) for d in range(D_MODEL)]
        x = rmsnorm_fx(emb, NORM0_FRAC, self.rsqrt)

        # attention block
        h = rmsnorm_fx(x, NORM_FRAC, self.rsqrt)
        q = matvec_fx(w, "wq", D_MODEL, D_MODEL, h, SHIFT_FROM_NORM)
        self.k_cache[pos] = matvec_fx(w, "wk", D_MODEL, D_MODEL, h, SHIFT_FROM_NORM)
        self.v_cache[pos] = matvec_fx(w, "wv", D_MODEL, D_MODEL, h, SHIFT_FROM_NORM)
        attn = attention_fx(q, self.k_cache[:pos + 1], self.v_cache[:pos + 1], self.exp)
        o = matvec_fx(w, "wo", D_MODEL, D_MODEL, attn, SHIFT_FROM_ACT)
        x = [sat16(a + b) for a, b in zip(x, o)]

        # MLP block
        h = rmsnorm_fx(x, NORM_FRAC, self.rsqrt)
        f = [max(0, v) for v in matvec_fx(w, "fc1", D_FF, D_MODEL, h, SHIFT_FROM_NORM)]
        g = matvec_fx(w, "fc2", D_MODEL, D_FF, f, SHIFT_FROM_ACT)
        x = [sat16(a + b) for a, b in zip(x, g)]

        logits = matvec_fx(w, "lm_head", VOCAB, D_MODEL, x, SHIFT_FROM_ACT)
        nxt = logits.index(max(logits))  # ties -> lowest index
        return {"x": x, "logits": logits, "next": nxt}


BOS = VOCAB - 1  # token 26 marks the start and the end of a name


def generate_fx(model: GptFx, start_token: int = BOS) -> list[int]:
    """Greedy autoregressive generation. Matches rtl/generator.sv.

    Feeds start_token at position 0, then each predicted token back in at the
    next position. Stops when BOS is predicted or the context (16) is full.
    Returns the generated tokens (BOS not included).
    """
    out, tok = [], start_token
    for pos in range(CTX):
        tok = model.step(tok, pos)["next"]
        if tok == BOS:
            break
        out.append(tok)
    return out
