"""Regenerate offline outputs from the audited snapshot, with no API calls."""
import argparse
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--without-mixed", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    for name in ("results", "figures"):
        (root / name).mkdir(exist_ok=True)
    command = [sys.executable, str(root / "analysis.py"), "--from-clean-data"]
    if not args.without_mixed:
        command.append("--mixed")
    subprocess.run(command, cwd=root, check=True)


if __name__ == "__main__":
    main()
