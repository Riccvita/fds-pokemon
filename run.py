"""Command line entry point.

Usage:
    python run.py --data DATA_DIR --out OUT_DIR [--cv]

DATA_DIR must contain train.jsonl, test.jsonl and type_chart_gen2.csv.
"""

import argparse
import json
import os
import sys

# Make the `pokemon` package importable when running from the repository root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pokemon.main import build_all, cross_validate, make_submissions


def main():
    """Build features, optionally run the CV and write the three submissions."""
    parser = argparse.ArgumentParser(description="Pokemon battle prediction pipeline")
    parser.add_argument("--data", required=True, help="folder with the competition files")
    parser.add_argument("--out", default="submissions", help="output folder for the CSV files")
    parser.add_argument("--cv", action="store_true", help="also run the 5-fold nested CV")
    args = parser.parse_args()

    D = build_all(args.data)
    print("feature matrix:", D["A"].shape)

    if args.cv:
        print(json.dumps(cross_validate(D), indent=2))

    print(make_submissions(D, args.out))


if __name__ == "__main__":
    main()
