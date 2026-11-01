#!/usr/bin/env python3
"""Prepare binned FedGrant trace CSVs from the five raw Mahimahi .up files."""

from __future__ import annotations

import argparse
from pathlib import Path

from fedgrant.traces.binning import TRACE_FILES, process_directory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="data/traces/raw", help="directory containing the five .up files")
    parser.add_argument("--output", default="data/traces", help="output directory")
    args = parser.parse_args()
    source = Path(args.source)
    missing = [name for name in TRACE_FILES if not (source / f"{name}.up").is_file()]
    if missing:
        names = ", ".join(f"{name}.up" for name in missing)
        parser.error(
            f"missing raw traces in {source}: {names}; run `make trace-fetch` first "
            "or pass --source DIR containing all five licensed .up files"
        )
    summary = process_directory(args.source, args.output)
    print(f"wrote {len(summary)} trace splits to {args.output}")


if __name__ == "__main__":
    main()
