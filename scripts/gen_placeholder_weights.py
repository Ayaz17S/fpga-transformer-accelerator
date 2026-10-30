"""Write a placeholder weight ROM image until the trained weights arrive.

Values are seeded Gaussian random numbers in Q2.14, scaled like a freshly
initialized network: embeddings with sigma 0.5, each matrix with sigma
1/sqrt(fan_in). Full-range random weights would saturate almost every
activation, which makes for a weak end-to-end test. The layout comes from
fixedpoint.LAYOUT, which must match rtl/model_pkg.sv.

Usage (from the project root):
    python scripts/gen_placeholder_weights.py
"""

import math
import pathlib
import random

from fixedpoint import D_MODEL, D_FF, LAYOUT, N_WEIGHTS, write_mem

FAN_IN = {"wq": D_MODEL, "wk": D_MODEL, "wv": D_MODEL, "wo": D_MODEL,
          "fc1": D_MODEL, "fc2": D_FF, "lm_head": D_MODEL}


def main() -> None:
    rng = random.Random(1)
    words = []
    for name, count in LAYOUT:
        sigma = 0.5 if name in ("wte", "wpe") else 1 / math.sqrt(FAN_IN[name])
        for _ in range(count):
            words.append(max(-32768, min(32767, round(rng.gauss(0, sigma) * 2**14))))

    assert len(words) == N_WEIGHTS, len(words)
    out = pathlib.Path(__file__).resolve().parent.parent / "data" / "weights_all.mem"
    write_mem(out, words)
    print(f"wrote {len(words)} words to {out}")


if __name__ == "__main__":
    main()
