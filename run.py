"""CLI for submissions and optional, separately stored validation protocols."""

import argparse
import json
from pathlib import Path

from pokemon.features import read_jsonl
from pokemon.main import (VARIANTS, build_all, compare_preparation, cross_validate,
                          make_submissions, repeat_validation)
from pokemon.protocols import REPEAT_SEEDS, audit_battles, file_digest, provenance


def main():
    parser = argparse.ArgumentParser(description="Pokemon battle prediction pipeline")
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--out", default="submissions", type=Path,
                        help="Kaggle submissions only")
    parser.add_argument("--cv-out", default="validation_runs/original", type=Path,
                        help="diagnostic artifacts, separate from Kaggle CSVs")
    parser.add_argument("--cv", action="store_true", help="five-fold validation")
    parser.add_argument("--validation-only", action="store_true", help="validate without fitting submissions")
    parser.add_argument("--reference-oof", type=Path, help="previous validation_predictions.csv (legacy probability CSV also accepted)")
    parser.add_argument("--repeat-cv", action="store_true", help="fixed outer seeds 42, 123, 2026")
    parser.add_argument("--group-by-p1", action="store_true", help="hold out entire P1 compositions")
    parser.add_argument("--preparation", choices=("global", "fold", "compare"), default="global")
    parser.add_argument("--outer-seed", type=int, default=42)
    parser.add_argument("--nested-cv", action="store_true", help="select fixed finalists using inner CV only")
    parser.add_argument("--finalists", nargs="+", choices=tuple(VARIANTS), default=list(VARIANTS))
    parser.add_argument("--selection-folds", type=int, default=3)
    parser.add_argument("--audit-only", action="store_true", help="inspect original data without building features or fitting")
    args = parser.parse_args()
    if args.cv_out.resolve() == args.out.resolve():
        parser.error("--cv-out and --out must be different directories")
    if args.repeat_cv and args.outer_seed != 42:
        parser.error("--repeat-cv uses fixed seeds; do not override --outer-seed")
    if args.reference_oof and (args.repeat_cv or args.preparation == "compare"):
        parser.error("Compare previous files one split/protocol at a time; omit --reference-oof here")
    if args.audit_only:
        train = read_jsonl(args.data / "train.jsonl")
        test_path = args.data / "test.jsonl"
        test = read_jsonl(test_path) if test_path.exists() else []
        audit, _ = audit_battles(train, test)
        audit["provenance"] = provenance({"input_sha256": {
            name: file_digest(args.data / name) for name in ("train.jsonl", "test.jsonl", "type_chart_gen2.csv")
            if (args.data / name).exists()}})
        args.cv_out.mkdir(parents=True, exist_ok=True)
        (args.cv_out / "data_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
        print("Audit:", args.cv_out / "data_audit.json")
        return
    do_cv = (args.cv or args.validation_only or args.repeat_cv or args.group_by_p1
             or args.nested_cv or args.preparation != "global")
    if args.reference_oof and not do_cv:
        parser.error("--reference-oof requires a validation mode")
    data = build_all(args.data)
    print("Feature matrix:", data["A"].shape)
    if do_cv:
        options = dict(split_mode="p1_grouped" if args.group_by_p1 else "random",
                       model_seed=42, te_seed=0, nested=args.nested_cv,
                       finalists=args.finalists, selection_splits=args.selection_folds)
        if args.preparation == "compare":
            # In repeated comparisons each seed gets its own paired artifacts.
            for seed in REPEAT_SEEDS if args.repeat_cv else (args.outer_seed,):
                folder = args.cv_out / f"seed_{seed}" if args.repeat_cv else args.cv_out
                print(json.dumps(compare_preparation(data, folder, outer_seed=seed, **options), indent=2))
        elif args.repeat_cv:
            print(json.dumps(repeat_validation(data, args.cv_out, preparation=args.preparation, **options), indent=2))
        else:
            report, _ = cross_validate(data, output_dir=args.cv_out,
                                       reference_oof=args.reference_oof, return_details=True,
                                       outer_seed=args.outer_seed, preparation=args.preparation, **options)
            print(json.dumps(report, indent=2))
    if not args.validation_only:
        print(make_submissions(data, args.out))


if __name__ == "__main__":
    main()
