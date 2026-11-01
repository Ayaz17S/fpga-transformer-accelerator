"""Run the microGPT accelerator from Python on the PYNQ-Z2.

Copy this file together with overlay/gpt_overlay.bit and overlay/gpt_overlay.hwh
into the same folder on the board (for example ~/jupyter_notebooks/gpt/), then:

    sudo python3 gpt_overlay.py            # generate from BOS
    sudo python3 gpt_overlay.py --all      # every start token, with timing

or import GptAccelerator from a Jupyter notebook.

Every call is plain register access to the AXI4-Lite interface in
rtl/axi_gpt.sv; the offsets below must match its register map.
"""

import argparse
import os
import time

from pynq import Overlay

CTRL, STATUS, START_TOKEN, GEN_LEN, CYCLES, ID, TOKENS = 0x00, 0x04, 0x08, 0x0C, 0x10, 0x14, 0x40
ID_VALUE = 0x6D475054  # "mGPT"
BOS = 26
CLOCK_HZ = 100e6       # FCLK_CLK0


def decode(tokens):
    """Tokens 0..25 -> 'a'..'z'. This follows the placeholder vocabulary;
    switch to the ML side's character table once trained weights are in."""
    return "".join(chr(ord("a") + t) for t in tokens)


class GptAccelerator:
    def __init__(self, bitfile=None):
        here = os.path.dirname(os.path.abspath(__file__))
        self.overlay = Overlay(bitfile or os.path.join(here, "gpt_overlay.bit"))
        # the block-design cell is named "gpt"; fall back to the only non-PS IP
        ips = [name for name in self.overlay.ip_dict if not name.startswith("ps7")]
        self.ip = getattr(self.overlay, "gpt", None) or getattr(self.overlay, ips[0])
        found = self.ip.read(ID)
        if found != ID_VALUE:
            raise RuntimeError(f"ID register reads 0x{found:08x}, expected 0x{ID_VALUE:08x}")

    def generate(self, start_token=BOS):
        """Run one greedy generation on the FPGA. Returns (tokens, cycles)."""
        self.ip.write(START_TOKEN, start_token)
        self.ip.write(CTRL, 1)
        while not self.ip.read(STATUS) & 0b10:  # wait for STATUS.done
            pass
        length = self.ip.read(GEN_LEN)
        tokens = [self.ip.read(TOKENS + 4 * i) for i in range(length)]
        return tokens, self.ip.read(CYCLES)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--all", action="store_true", help="generate from every start token")
    args = parser.parse_args()

    acc = GptAccelerator()
    starts = range(BOS + 1) if args.all else [BOS]
    for start in starts:
        t0 = time.perf_counter()
        tokens, cycles = acc.generate(start)
        wall_us = (time.perf_counter() - t0) * 1e6
        hw_us = cycles / CLOCK_HZ * 1e6
        per_token = hw_us / max(len(tokens), 1)
        print(f"start {start:2d}: {decode(tokens):16s} {len(tokens):2d} tokens  "
              f"hw {hw_us:8.1f} us ({per_token:5.1f} us/token)  wall {wall_us:8.1f} us")


if __name__ == "__main__":
    main()
