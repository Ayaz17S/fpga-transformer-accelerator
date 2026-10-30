"""Generate RMSNorm test vectors from the bit-exact Python reference.

Writes (all in data/):
    rmsnorm_x.mem     N vectors x 16 inputs, Q6.10
    rmsnorm_frac.mem  N output fraction-bit counts (12 -> Q4.12, 10 -> Q6.10)
    rmsnorm_h.mem     N vectors x 16 expected outputs
Also prints how far the fixed-point result is from exact float RMSNorm.

Usage (from the project root):
    python scripts/gen_rmsnorm_vectors.py
"""

import math
import pathlib
import random

from fixedpoint import D_MODEL, EPS_RAW, rmsnorm_fx, rsqrt_lut, write_mem

DATA = pathlib.Path(__file__).resolve().parent.parent / "data"
IN_FRAC = 10


def clamp16(v: int) -> int:
    return max(-32768, min(32767, v))


def make_cases(rng: random.Random) -> list[list[int]]:
    cases = [
        [0] * D_MODEL,                                    # only eps
        [1024] * D_MODEL,                                 # all 1.0
        [-1024] * D_MODEL,
        [32767] + [0] * (D_MODEL - 1),                    # one-hot max -> h = 4
        [0] * (D_MODEL - 1) + [-32768],                   # one-hot min -> h = -4
        [32767] * D_MODEL,                                # largest sum of squares
        [rng.choice([-1, 1]) for _ in range(D_MODEL)],    # tiny
    ]
    # sweep the magnitude so every even exponent e is exercised
    for k in range(16):
        for _ in range(4):
            cases.append([clamp16(rng.randint(-(1 << k), 1 << k)) for _ in range(D_MODEL)])
    # typical activations: gaussian with sigma 0.25 .. 8
    for _ in range(120):
        sigma = rng.choice([0.25, 0.5, 1.0, 2.0, 4.0, 8.0])
        cases.append([clamp16(round(rng.gauss(0, sigma) * (1 << IN_FRAC))) for _ in range(D_MODEL)])
    return cases


def main() -> None:
    rng = random.Random(4)
    lut = rsqrt_lut()
    cases = make_cases(rng)

    xs, fracs, hs = [], [], []
    worst_lsb = 0.0
    for n, x in enumerate(cases):
        out_frac = 12 if n % 2 == 0 else 10
        h = rmsnorm_fx(x, out_frac, lut)
        xs += x
        fracs.append(out_frac)
        hs += h

        # accuracy against exact float RMSNorm, in output LSBs
        xf = [v / (1 << IN_FRAC) for v in x]
        rms = math.sqrt(sum(v * v for v in xf) / D_MODEL + EPS_RAW / (1 << 2 * IN_FRAC))
        for hv, xv in zip(h, xf):
            worst_lsb = max(worst_lsb, abs(hv - xv / rms * (1 << out_frac)))

    write_mem(DATA / "rmsnorm_x.mem", xs)
    write_mem(DATA / "rmsnorm_frac.mem", fracs)
    write_mem(DATA / "rmsnorm_h.mem", hs)
    print(f"wrote {len(cases)} vectors to {DATA}")
    print(f"worst error vs float RMSNorm: {worst_lsb:.1f} output LSBs")


if __name__ == "__main__":
    main()
