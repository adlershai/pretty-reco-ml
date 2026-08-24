"""Minimal embeddings worker entrypoint.

Image vectorization is the first ML capability. Encoding is not
implemented in this setup milestone.
"""

from __future__ import annotations

import argparse
import sys

__version__ = "0.0.0"


def main() -> int:
    parser = argparse.ArgumentParser(description="Pretty Reco ML embeddings worker")
    parser.add_argument(
        "--version",
        action="store_true",
        help="Print worker and Python versions, then exit",
    )
    args = parser.parse_args()
    if args.version:
        print(f"pretty-reco-ml {__version__}")
        print(f"Python {sys.version.split()[0]}")
        return 0
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
