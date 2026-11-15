"""Quantize the trained microGPT checkpoint into the weight ROM image.

Reads data/checkpoint/microgpt_14fb038_seed42.json (the golden reference's
trained state_dict), rounds every weight to Q2.14 (round half away from zero,
saturating) in the fixedpoint.LAYOUT order, and writes data/weights_all.mem.

It then checks the result against a float64 forward pass of the same
checkpoint (pure Python, mirrors upstream microgpt.py): greedy generation from
BOS and from a few prefixes must give the same names, and over the test names
(if --names is given) the fixed-point argmax is compared with the float64
argmax token by token.

Usage (from the project root):
    python scripts/import_checkpoint.py
    python scripts/import_checkpoint.py --names path/to/names.txt --limit 500
"""

import argparse
import hashlib
import json
import math
import pathlib

from fixedpoint import (BOS, CTX, D_MODEL, LAYOUT, N_WEIGHTS, VOCAB, GptFx,
                        exp_lut, generate_fx, rsqrt_lut, write_mem)

ROOT = pathlib.Path(__file__).resolve().parent.parent
CKPT = ROOT / "data" / "checkpoint" / "microgpt_14fb038_seed42.json"
UCHARS = "abcdefghijklmnopqrstuvwxyz"

# fixedpoint.LAYOUT name -> state_dict key
KEYS = {"wte": "wte", "wpe": "wpe", "wq": "layer0.attn_wq", "wk": "layer0.attn_wk",
        "wv": "layer0.attn_wv", "wo": "layer0.attn_wo", "fc1": "layer0.mlp_fc1",
        "fc2": "layer0.mlp_fc2", "lm_head": "lm_head"}


def quantize(x: float) -> int:
    """Q2.14, round half away from zero, saturate to 16 bits."""
    q = int(math.floor(abs(x) * 2**14 + 0.5)) * (1 if x >= 0 else -1)
    return max(-32768, min(32767, q))


def flatten(sd: dict) -> list[float]:
    flat = []
    for name, count in LAYOUT:
        rows = sd[KEYS[name]]
        vals = [v for row in rows for v in row]
        assert len(vals) == count, (name, len(vals), count)
        flat += vals
    return flat


# ---------------- float64 reference (mirrors upstream microgpt.py) ----------------

def _rms(x):
    s = sum(v * v for v in x) / len(x)
    r = (s + 1e-5) ** -0.5
    return [v * r for v in x]


def _lin(x, w):
    return [sum(a * b for a, b in zip(row, x)) for row in w]


class GptFloat:
    def __init__(self, sd: dict, n_head: int = 4):
        self.sd, self.nh = sd, n_head
        self.keys, self.vals = [], []

    def step(self, token: int, pos: int) -> list[float]:
        sd, hd = self.sd, D_MODEL // self.nh
        x = [a + b for a, b in zip(sd["wte"][token], sd["wpe"][pos])]
        x = _rms(x)
        res = x
        x = _rms(x)
        q = _lin(x, sd["layer0.attn_wq"])
        self.keys.append(_lin(x, sd["layer0.attn_wk"]))
        self.vals.append(_lin(x, sd["layer0.attn_wv"]))
        out = []
        for h in range(self.nh):
            sl = slice(h * hd, (h + 1) * hd)
            s = [sum(a * b for a, b in zip(q[sl], k[sl])) / hd ** 0.5 for k in self.keys]
            m = max(s)
            e = [math.exp(v - m) for v in s]
            t = sum(e)
            p = [v / t for v in e]
            out += [sum(p[i] * self.vals[i][sl][j] for i in range(len(p))) for j in range(hd)]
        x = [a + b for a, b in zip(_lin(out, sd["layer0.attn_wo"]), res)]
        res = x
        x = _rms(x)
        x = [max(0.0, v) for v in _lin(x, sd["layer0.mlp_fc1"])]
        x = [a + b for a, b in zip(_lin(x, sd["layer0.mlp_fc2"]), res)]
        return _lin(x, sd["lm_head"])


def generate_float(sd: dict, start: int = BOS) -> list[int]:
    m, out, tok = GptFloat(sd), [], start
    for pos in range(CTX):
        lg = m.step(tok, pos)
        tok = lg.index(max(lg))
        if tok == BOS:
            break
        out.append(tok)
    return out


def text(tokens: list[int]) -> str:
    return "".join(UCHARS[t] for t in tokens)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", help="names.txt to measure argmax agreement on")
    ap.add_argument("--limit", type=int, default=300, help="max names to evaluate")
    args = ap.parse_args()

    raw = CKPT.read_bytes()
    sd = json.loads(raw)["state_dict"]
    flat = flatten(sd)
    words = [quantize(v) for v in flat]
    assert len(words) == N_WEIGHTS
    sat = sum(abs(w) >= 32767 for w in words)
    err = max(abs(w / 2**14 - v) for w, v in zip(words, flat))
    write_mem(ROOT / "data" / "weights_all.mem", words)
    print(f"checkpoint sha256 {hashlib.sha256(raw).hexdigest()[:16]}...")
    print(f"wrote {len(words)} weights; saturated {sat}; max quantization error {err:.2e} "
          f"(half LSB = {0.5 / 2**14:.2e})")

    # greedy generation: fixed point vs float64
    fx = lambda start: generate_fx(GptFx(words, rsqrt_lut(), exp_lut()), start)
    print("greedy generation (fixed | float64):")
    for start in [BOS, 0, 1, 10]:
        a, b = text(fx(start)), text(generate_float(sd, start))
        print(f"  start {start:2d}: {a:<16s} | {b:<16s} {'ok' if a == b else 'DIFFERENT'}")

    if args.names:
        names = [n.strip() for n in pathlib.Path(args.names).read_text().split("\n") if n.strip()]
        names = names[:args.limit]
        agree = total = 0
        for name in names:
            toks = [BOS] + [UCHARS.index(c) for c in name] + [BOS]
            toks = toks[:CTX]
            f, g = GptFx(words, rsqrt_lut(), exp_lut()), GptFloat(sd)
            for pos in range(len(toks) - 1):
                lg = g.step(toks[pos], pos)
                agree += f.step(toks[pos], pos)["next"] == lg.index(max(lg))
                total += 1
        print(f"argmax agreement with float64 on {len(names)} names: "
              f"{agree}/{total} = {100 * agree / total:.2f}%")


if __name__ == "__main__":
    main()
