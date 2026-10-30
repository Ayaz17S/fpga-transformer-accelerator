"""Generate whole-model test vectors: full token steps through GptFx.

Each sequence feeds 16 given tokens (teacher forcing) at positions 0..15, so
the KV cache fills exactly as it does in the hardware. Writes (all in data/):
    model_tokens.mem   SEQS x 16 input tokens
    model_x.mem        SEQS x 16 steps x 16   residual stream after the MLP block
    model_logits.mem   SEQS x 16 steps x 27   logits
    model_next.mem     SEQS x 16 steps        argmax token
    gen_len.mem        27 generated lengths, one per start token 0..26
    gen_tokens.mem     27 x 16 generated tokens (unused slots 0)

Usage (from the project root, after gen_placeholder_weights.py and gen_luts.py):
    python scripts/gen_model_vectors.py
"""

import pathlib
import random

from fixedpoint import (CTX, VOCAB, GptFx, exp_lut, generate_fx, read_mem,
                        rsqrt_lut, write_mem)

DATA = pathlib.Path(__file__).resolve().parent.parent / "data"
SEQS = 6


def main() -> None:
    rng = random.Random(6)
    weights = read_mem(DATA / "weights_all.mem")

    tokens, xs, logits, nexts = [], [], [], []
    saturated = total = 0
    for _ in range(SEQS):
        model = GptFx(weights, rsqrt_lut(), exp_lut())   # fresh KV cache per sequence
        for pos in range(CTX):
            tok = rng.randrange(VOCAB)
            out = model.step(tok, pos)
            tokens.append(tok)
            xs += out["x"]
            logits += out["logits"]
            nexts.append(out["next"])
            saturated += sum(abs(v) >= 32767 for v in out["x"] + out["logits"])
            total += len(out["x"]) + len(out["logits"])

    write_mem(DATA / "model_tokens.mem", tokens)
    write_mem(DATA / "model_x.mem", xs)
    write_mem(DATA / "model_logits.mem", logits)
    write_mem(DATA / "model_next.mem", nexts)
    print(f"wrote {SEQS} sequences x {CTX} steps; "
          f"{saturated}/{total} final values saturated; "
          f"distinct next tokens: {len(set(nexts))}")

    # ---------------- generation from every possible start token ----------------
    gen_len, gen_tokens = [], []
    for start in range(VOCAB):
        seq = generate_fx(GptFx(weights, rsqrt_lut(), exp_lut()), start)
        gen_len.append(len(seq))
        gen_tokens += seq + [0] * (CTX - len(seq))
    write_mem(DATA / "gen_len.mem", gen_len)
    write_mem(DATA / "gen_tokens.mem", gen_tokens)
    stopped = sum(n < CTX for n in gen_len)
    print(f"wrote generation for {VOCAB} start tokens; "
          f"{stopped} stopped on BOS, {VOCAB - stopped} filled the context")


if __name__ == "__main__":
    main()
