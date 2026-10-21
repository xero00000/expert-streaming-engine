#!/usr/bin/env python3
"""Bundle native/Studio MSVC runtime DLLs from the installed VS redist."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.windows_msvc import redist_directory, stage


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-bin", action="append", required=True, type=Path)
    parser.add_argument("--gui-directory", required=True, type=Path)
    parser.add_argument("--redist-directory", type=Path)
    args = parser.parse_args()
    report = stage(args.runtime_bin, args.gui_directory, redist_directory(args.redist_directory))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
