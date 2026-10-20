#!/usr/bin/env python3
"""Validate the complete package matrix and generate update/checksum files."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.release_assets import prepare_main

if __name__ == "__main__":
    prepare_main()
