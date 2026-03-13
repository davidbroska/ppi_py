#!/usr/bin/env python3
"""Thin wrapper that forwards to the R implementation."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


def main() -> int:
    rscript = shutil.which("Rscript")
    if rscript is None:
        raise RuntimeError("Rscript is required to run prepare_kluger_data.R.")

    script_path = Path(__file__).with_suffix(".R")
    result = subprocess.run([rscript, str(script_path), *sys.argv[1:]], check=False)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
