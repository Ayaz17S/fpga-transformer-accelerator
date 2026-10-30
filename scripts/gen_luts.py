"""Write the shared lookup tables that both the RTL and the Python model load.

Usage (from the project root):
    python scripts/gen_luts.py
"""

import pathlib

from fixedpoint import exp_lut, rsqrt_lut, write_mem

DATA = pathlib.Path(__file__).resolve().parent.parent / "data"


def main() -> None:
    for name, lut in (("rsqrt_lut.mem", rsqrt_lut()), ("exp_lut.mem", exp_lut())):
        write_mem(DATA / name, lut)
        print(f"wrote {len(lut)} entries to {DATA / name}")


if __name__ == "__main__":
    main()
