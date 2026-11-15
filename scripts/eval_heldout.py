"""Measure the bit-exact fixed-point model (GptFx = the RTL) against float64
on the reference's held-out set.

Held-out set = docs[num_steps:] of the checkpoint's `docs_order`, minus names
seen in training (docs[:num_steps]), exactly as the reference's
analyze_checkpoint.py defines it. Teacher forcing: each name is fed as
BOS + characters, predicting the next character and the closing BOS.

Reports: top-1 agreement fixed vs float64, disagreements broken down by the
float64 top-2 logit margin (a disagreement is only acceptable when the margin
is below the logit error), max/mean |logit error|, and token-mean NLL of both.

Usage (from the project root):
    python scripts/eval_heldout.py                 # whole held-out set (slow)
    python scripts/eval_heldout.py --stride 20     # every 20th name
"""

import argparse
import json
import math
import multiprocessing as mp
import pathlib

from fixedpoint import BOS, CTX, GptFx, exp_lut, read_mem, rsqrt_lut
from import_checkpoint import CKPT, UCHARS, GptFloat

ROOT = pathlib.Path(__file__).resolve().parent.parent
LOGIT_FRAC = 10   # logits are Q6.10


def nll(logits: list[float], target: int) -> float:
    m = max(logits)
    return math.log(sum(math.exp(v - m) for v in logits)) - (logits[target] - m)


def work(names: list[str]):
    words = read_mem(ROOT / "data" / "weights_all.mem")
    sd = json.loads(CKPT.read_bytes())["state_dict"]
    rs, ex = rsqrt_lut(), exp_lut()
    n = agree = 0
    max_err = err_sum = 0.0
    nll_fx = nll_fl = 0.0
    bad = []          # (margin, err) for each disagreement
    worst_agree_margin_violation = 0
    for name in names:
        toks = ([BOS] + [UCHARS.index(c) for c in name] + [BOS])[:CTX + 1]
        fx, fl = GptFx(words, rs, ex), GptFloat(sd)
        for pos in range(len(toks) - 1):
            lf = [v / 2**LOGIT_FRAC for v in fx.step(toks[pos], pos)["logits"]]
            lg = fl.step(toks[pos], pos)
            e = max(abs(a - b) for a, b in zip(lf, lg))
            max_err, err_sum = max(max_err, e), err_sum + e
            top = sorted(lg, reverse=True)
            margin = top[0] - top[1]
            if lf.index(max(lf)) == lg.index(max(lg)):
                agree += 1
            else:
                bad.append((margin, e))
                if margin > 2 * e:
                    worst_agree_margin_violation += 1
            nll_fx += nll(lf, toks[pos + 1])
            nll_fl += nll(lg, toks[pos + 1])
            n += 1
    return n, agree, max_err, err_sum, nll_fx, nll_fl, bad, worst_agree_margin_violation


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=1, help="evaluate every Nth held-out name")
    ap.add_argument("--jobs", type=int, default=mp.cpu_count())
    args = ap.parse_args()

    ck = json.loads(CKPT.read_bytes())
    docs, ns = ck["docs_order"], ck["meta"]["num_steps"]
    trained = set(docs[:ns])
    heldout = [d for d in docs[ns:] if d not in trained]
    names = heldout[::args.stride]
    print(f"held-out set: {len(heldout)} names; evaluating {len(names)} (stride {args.stride})")

    chunks = [names[i::args.jobs * 4] for i in range(args.jobs * 4)]
    with mp.Pool(args.jobs) as pool:
        parts = pool.map(work, chunks)

    n = sum(p[0] for p in parts)
    agree = sum(p[1] for p in parts)
    max_err = max(p[2] for p in parts)
    mean_err = sum(p[3] for p in parts) / n
    nfx, nfl = sum(p[4] for p in parts) / n, sum(p[5] for p in parts) / n
    bad = [b for p in parts for b in p[6]]
    viol = sum(p[7] for p in parts)
    print(f"tokens evaluated        : {n}")
    print(f"top-1 agreement         : {agree}/{n} = {100 * agree / n:.3f}%")
    print(f"max |logit error|       : {max_err:.4f}   mean max-per-token error: {mean_err:.4f}")
    print(f"token-mean NLL          : fixed {nfx:.6f}   float64 {nfl:.6f}   delta {nfx - nfl:+.6f}")
    if bad:
        print(f"disagreements           : {len(bad)}; largest float64 top-2 margin among them "
              f"{max(b[0] for b in bad):.4f}; with margin > 2x logit error: {viol}")


if __name__ == "__main__":
    main()
