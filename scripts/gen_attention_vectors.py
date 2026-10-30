"""Generate softmax and attention test vectors from the bit-exact Python reference.

Writes (all in data/):
    softmax_n.mem      N vector lengths (1..16)
    softmax_s.mem      N x 16 scores, Q6.10 (unused slots 0)
    softmax_p.mem      N x 16 expected probabilities, Q2.14
    attn_q.mem         SEQS x 16 positions x 16 values, Q6.10
    attn_k.mem, attn_v.mem   same layout: the k / v row produced at each position
    attn_out.mem       SEQS x 16 positions x 16 expected outputs, Q6.10
Also prints how far the fixed-point results are from float.

Usage (from the project root):
    python scripts/gen_attention_vectors.py
"""

import math
import pathlib
import random

from fixedpoint import (HEAD_DIM, N_HEAD, attention_fx, exp_lut, softmax_fx,
                        write_mem)

DATA = pathlib.Path(__file__).resolve().parent.parent / "data"
CTX, D_MODEL = 16, 16
SEQS = 8


def clamp16(v: int) -> int:
    return max(-32768, min(32767, v))


def q610(rng: random.Random, sigma: float) -> int:
    return clamp16(round(rng.gauss(0, sigma) * 1024))


def float_softmax(scores: list[int]) -> list[float]:
    xs = [s / 1024 for s in scores]
    m = max(xs)
    ex = [math.exp(x - m) for x in xs]
    total = sum(ex)
    return [v / total for v in ex]


def softmax_cases(rng: random.Random) -> list[list[int]]:
    cases = [
        [0],                                   # n = 1 -> p = 1.0 exactly
        [0] * 16,                              # uniform -> 1/16
        [5120] + [0] * 15,                     # one clear winner (5.0)
        [32767, -32768],                       # largest spread: exp clamps to entry 255
        [-1024 * k for k in range(16)],        # 0, -1, ... -15
        [3 * k for k in range(16)],            # differences below one LUT step
    ]
    for _ in range(200):
        n = rng.randint(1, 16)
        sigma = rng.choice([0.5, 1.0, 2.0, 4.0, 8.0])
        cases.append([q610(rng, sigma) for _ in range(n)])
    return cases


def main() -> None:
    rng = random.Random(5)
    lut = exp_lut()

    # ---------------- softmax ----------------
    ns, ss, ps = [], [], []
    worst_p = 0.0
    for scores in softmax_cases(rng):
        p = softmax_fx(scores, lut)
        pad = [0] * (16 - len(scores))
        ns.append(len(scores))
        ss += scores + pad
        ps += p + pad
        for pf, pq in zip(float_softmax(scores), p):
            worst_p = max(worst_p, abs(pq / 2**14 - pf))
    write_mem(DATA / "softmax_n.mem", ns)
    write_mem(DATA / "softmax_s.mem", ss)
    write_mem(DATA / "softmax_p.mem", ps)
    print(f"softmax: {len(ns)} vectors, worst |p - float p| = {worst_p:.5f}")

    # ---------------- attention over whole sequences ----------------
    qs, kall, vall, outs = [], [], [], []
    worst_out = 0.0
    for seq in range(SEQS):
        sigma = [0.5, 1.0, 2.0, 4.0][seq % 4]
        ks, vs = [], []
        for t in range(CTX):
            q = [q610(rng, sigma) for _ in range(D_MODEL)]
            k = [q610(rng, sigma) for _ in range(D_MODEL)]
            v = [q610(rng, sigma) for _ in range(D_MODEL)]
            ks.append(k)
            vs.append(v)
            out = attention_fx(q, ks, vs, lut)
            qs += q
            kall += k
            vall += v
            outs += out

            # float reference from the same integer q, k, v
            for h in range(N_HEAD):
                dims = range(h * HEAD_DIM, (h + 1) * HEAD_DIM)
                sc = [sum(q[d] * kk[d] for d in dims) / 2**20 * 0.5 for kk in ks]
                m = max(sc)
                ex = [math.exp(x - m) for x in sc]
                pf = [x / sum(ex) for x in ex]
                for d in dims:
                    of = sum(pj * vv[d] / 1024 for pj, vv in zip(pf, vs))
                    worst_out = max(worst_out, abs(out[d] / 1024 - of))
    write_mem(DATA / "attn_q.mem", qs)
    write_mem(DATA / "attn_k.mem", kall)
    write_mem(DATA / "attn_v.mem", vall)
    write_mem(DATA / "attn_out.mem", outs)
    print(f"attention: {SEQS} sequences x {CTX} positions, "
          f"worst |out - float out| = {worst_out:.4f} ({worst_out * 1024:.0f} LSBs)")


if __name__ == "__main__":
    main()
