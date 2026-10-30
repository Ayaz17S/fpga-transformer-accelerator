"""Write the shared lookup tables that both the RTL and the Python model load.

Usage (from the project root):
    python scripts/gen_luts.py
"""

import pathlib

from fixedpoint import rsqrt_lut, write_mem

DATA = pathlib.Path(__file__).resolve().parent.parent / "data"


def main() -> None:
    lut = rsqrt_lut()
    write_mem(DATA / "rsqrt_lut.mem", lut)
    print(f"wrote {len(lut)} entries to {DATA / 'rsqrt_lut.mem'}")


if __name__ == "__main__":
    main()
