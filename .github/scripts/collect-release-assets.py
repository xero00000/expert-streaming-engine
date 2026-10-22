#!/usr/bin/env python3
"""Collect the requested version and runtime variant with source provenance."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.release_assets import collect, collect_main

if __name__ == "__main__":
    collect_main()
