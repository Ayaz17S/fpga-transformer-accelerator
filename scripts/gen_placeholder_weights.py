"""Write a placeholder weight ROM image until the trained weights arrive.

Values are random 16-bit two's complement words (seeded, so the file is the
same every run). The layout must match the memory map in rtl/model_pkg.sv.

Usage (from the project root):
    python scripts/gen_placeholder_weights.py
"""

import pathlib
import random

D_MODEL, CTX, VOCAB, D_FF = 16, 16, 27, 64

# (name, words) in ROM order - keep in sync with rtl/model_pkg.sv
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
N_WEIGHTS = 4192


def main() -> None:
    rng = random.Random(1)
    words = []
    for _name, count in LAYOUT:
        for _ in range(count):
            value = rng.randint(-32768, 32767)
            words.append(f"{value & 0xFFFF:04x}")  # two's complement, 4 hex digits

    assert len(words) == N_WEIGHTS, len(words)

    out = pathlib.Path(__file__).resolve().parent.parent / "data" / "weights_all.mem"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(words) + "\n")
    print(f"wrote {len(words)} words to {out}")


if __name__ == "__main__":
    main()
